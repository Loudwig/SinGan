from torchvision.transforms.functional import resize, to_tensor, to_pil_image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

import ot
from time import time
from torchvision import transforms

from models import *

# convert [-1,1] image to PIL image (to_pil_image requires [0,1] image.
def pil_from_minus1to1(t):
    """
    Convertit un tenseur [-1,1] (CHW ou 1,C,H,W) en PIL Image.
    """
    t = t.squeeze(0) if t.dim() == 4 else t          # B×C×H×W -> C×H×W
    t = ((t.clamp(-1, 1) + 1) / 2)                   # [-1,1]  ->  [0,1]
    return to_pil_image(t.cpu())


def extract_patches(img, patch_size, stride):
    
    img_batch = img.unsqueeze(0)  # (1, C, H, W)
    patches = torch.nn.functional.unfold(img_batch, kernel_size=patch_size, stride=stride)
    patches = patches.squeeze(0).T  # (num_patches, C*patch_size*patch_size)
   
    return patches

# get gaussian param of patches
def get_gaussian_params(patches):
    # patches: Tensor (N, D)
    
    mu = patches.mean(dim=0)
    centered = patches - mu
    Sigma = (centered.T @ centered) / (patches.shape[0] - 1)
    
    return mu, Sigma

# sqrt of a matrix. Need double to be sufficiently precise
def matrix_sqrt_eig(mat, eps=1e-10):
    """
    Symmetric square root of an SPD matrix via eigen-decomposition.
    """
    # 1) Enforce perfect symmetry
    mat = (mat + mat.T) / 2

    # 2) Eigen-decomposition
    vals, vecs = torch.linalg.eigh(mat)
    # 3) Clamp to avoid tiny negatives, then sqrt
    vals = torch.clamp(vals, min=eps)
    sqrt_mat = vecs @ torch.diag(torch.sqrt(vals)) @ vecs.T

    # 4) Re-symmetrize result
    sqrt_mat = (sqrt_mat + sqrt_mat.T) / 2
    
    return sqrt_mat


# Distance de wasserstein entre deux distribution gaussienne.
def wasserstein_2_gaussian_eig(mu1, Sigma1, mu2, Sigma2, eps=1e-12, debug=False):
    """
    Computes W2^2 between two Gaussians (mu1, Sigma1) and (mu2, Sigma2),
    using a robust eigen-based sqrt.
    """
    # Mean term
    diff_mu_sq = torch.norm(mu1 - mu2)**2
    if debug:
        print("moyenne diff =", diff_mu_sq.item())

    # double sinon pas assez précis. 
    A = Sigma1 + eps * torch.eye(Sigma1.size(0), device=Sigma1.device,dtype=torch.float64)
    S1 = matrix_sqrt_eig(A, eps=eps)

    # Inner product √(S1 Σ2 S1)
    prod = S1 @ Sigma2 @ S1
    S2 = matrix_sqrt_eig(prod, eps=eps)

    # Trace term
    trace_term = torch.trace(Sigma1 + Sigma2 - 2 * S2)
    if debug:
        print("trace term =", trace_term.item())

    return (diff_mu_sq + trace_term) /len(mu1)



@torch.no_grad()
def generate_multiscale(imgs_ref,Generators,sigma_n,N,device,start_scale = None):
    
    if start_scale == None : 
        start_scale = N-1
    """
    start_scale = 0  → on ne renouvelle le bruit qu’à la fine scale
    start_scale = N-1→ on renouvelle le bruit à toutes les échelles
    """
    # Génération totale
    if start_scale == N-1 : 
        h, w = imgs_ref[-1].shape[2:]
        gen_image = [torch.zeros((1, 3, h, w), device=device)]

         # 2. on remonte coarse → fine
        for i in range(N):
            k = N - 1 - i                    
        
            prev = gen_image[-1]
            if prev.shape[2:] != imgs_ref[k].shape[2:]:
                prev = F.interpolate(prev, size=imgs_ref[k].shape[2:],
                                    mode='bilinear', align_corners=False)

            z = torch.randn_like(prev) * sigma_n[k]
            

            x_k = Generators[k](z + prev) + prev
            gen_image.append(x_k)

        

    else : 
        low = imgs_ref[start_scale+1] 
        up = F.interpolate(low, size=imgs_ref[start_scale].shape[2:],
                                    mode='bilinear', align_corners=False)
        gen_image = [up]

         # 2. on remonte coarse → fine
        for scale in range(start_scale,-1,-1):
                            
            prev = gen_image[-1]
            if prev.shape[2:] != imgs_ref[scale].shape[2:]:
                prev = F.interpolate(prev, size=imgs_ref[scale].shape[2:],
                                    mode='bilinear', align_corners=False)

            z = torch.randn_like(prev) * sigma_n[scale]
            x_k = Generators[scale](z + prev) + prev
            gen_image.append(x_k)

    
    gen_image = gen_image[1:].copy()
    gen_image.reverse()
    return gen_image
        
    
# --- (3) fonction helper : recrée un Generator adapté au state_dict -----------------
def build_generator_from_state(sd, device="cpu"):
    for key in sd.keys():
        if key.endswith(".weight") and sd[key].dim() == 4:
            n_hidden = sd[key].shape[0]
            break
    else:
        raise KeyError("Impossible d'inférer n_hidden depuis le state_dict.")
    G = Generator(n_hidden).to(device)
    G.load_state_dict(sd, strict=True)
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G


def preprocess_features(features):
    """
    Prend features de forme (B, C, H, W)
    et renvoie un tenseur (B * H * W, C),
    i.e. un nuage de B*H*W vecteurs C-dimensionnels.
    """
    B, C, H, W = features.shape
    # 1) permute pour déplacer C en dernière dimension
    x = features.permute(0, 2, 3, 1)    # [B, H, W, C]
    # 2) flatten batch+spatial en une seule dimension
    return x.reshape(B * H * W, C)      # [B*H*W, C]

def load_inception():
  model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
  model.eval()
  model.avgpool = nn.Identity()
  model.dropout = nn.Identity()
  model.fc = nn.Identity()
  return model

def preprocess_image(file_path):
  preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  input_image1 = Image.open(file_path).convert('RGB')
  input_tensor1 = preprocess(input_image1)
  input_batch1 = input_tensor1.unsqueeze(0)

  return input_batch1

def sliced_wasserstein_image(model, file_path1, file_path2, n_projections=100, verbose=False):
  n_points = 64

  input_batch1 = preprocess_image(file_path1)
  input_batch2 = preprocess_image(file_path2)

  nuage_A = model(input_batch1)[0].view(2048, n_points).T.detach().numpy()
  nuage_B = model(input_batch2)[0].view(2048, n_points).T.detach().numpy()

  a = np.ones((n_points,)) / n_points
  b = np.ones((n_points,)) / n_points

  t1 = time()
  res = ot.sliced_wasserstein_distance(nuage_A,nuage_B,a, b, n_projections)
  t2=time()

  if verbose:
    print(f"Executed in {t2-t1:.4f}s")

  return res

def wasserstein_image(model, file_path1, file_path2, verbose=False):
  n_points = 64

  input_batch1 = preprocess_image(file_path1)
  input_batch2 = preprocess_image(file_path2)

  nuage_A = model(input_batch1)[0].view(2048, n_points).T.detach().numpy()
  nuage_B = model(input_batch2)[0].view(2048, n_points).T.detach().numpy()

  a = np.ones((n_points,)) / n_points
  b = np.ones((n_points,)) / n_points

  t1 = time()
  M = ot.dist(nuage_A, nuage_B, metric='euclidean')
  res = ot.emd2(a, b, M)
  t2=time()

  if verbose:
    print(f"Executed in {t2-t1:.4f}s")

  return res

def preprocess_image_for_lpips(file_path, image_size=256):
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                   
        transforms.Normalize((0.5, 0.5, 0.5),    
                             (0.5, 0.5, 0.5)),
    ])

    image = Image.open(file_path).convert('RGB')
    tensor = preprocess(image).unsqueeze(0)
    return tensor
