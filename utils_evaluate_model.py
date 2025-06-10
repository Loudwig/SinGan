from torchvision.transforms.functional import resize, to_tensor, to_pil_image
import numpy as np
import matplotlib.pyplot as plt
import time
import torchsummary

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms.functional import resize, to_tensor, to_pil_image
from PIL import Image
from tqdm import tqdm
import os, datetime ,json
from models import *

# convert [-1,1] image to PIL image (to_pil_image requires [0,1] image.
def pil_from_minus1to1(t):
    """
    Convertit un tenseur [-1,1] (CHW ou 1,C,H,W) en PIL Image.
    """
    t = t.squeeze(0) if t.dim() == 4 else t          # B×C×H×W → C×H×W
    t = ((t.clamp(-1, 1) + 1) / 2)                   # [-1,1] → [0,1]
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


def sliced_wasserstein_distance_features(real_features, fake_features, num_projections=128):
    # real_features shape: (B, C_feat, H_feat, W_feat)
    # Reshape to (B * H_feat * W_feat, C_feat) where C_feat is the dimension of the feature vector
    real_flattened = real_features.permute(0, 2, 3, 1).reshape(-1, real_features.shape[1])
    fake_flattened = fake_features.permute(0, 2, 3, 1).reshape(-1, fake_features.shape[1])

    # Now real_flattened and fake_flattened have shape (num_samples, feature_dim)
    # where num_samples = B * H_feat * W_feat and feature_dim = C_feat

    feature_dim = real_flattened.shape[1] # This is C_feat

    projections = torch.randn(feature_dim, num_projections).to(real_features.device)
    projections /= torch.sqrt(torch.sum(projections**2, dim=0, keepdim=True))

    real_projections = torch.matmul(real_flattened, projections)
    fake_projections = torch.matmul(fake_flattened, projections)

    sorted_real_projections, _ = torch.sort(real_projections, dim=0)
    sorted_fake_projections, _ = torch.sort(fake_projections, dim=0)

    return torch.mean(torch.abs(sorted_real_projections - sorted_fake_projections))