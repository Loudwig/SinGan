import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_tensor, to_pil_image


def extract_patches_batched(imgs, patch_size, stride):
    """
    Extracts sliding patches from a batch of images.

    Args:
        imgs (torch.Tensor): Input images, shape (B, C, H, W).
        patch_size (int or tuple): The size of the sliding patches (e.g., 32 for 32x32 patches).
        stride (int or tuple): The stride of the sliding patches.

    Returns:
        torch.Tensor: Flattened patches, shape (B * num_patches_per_image, C * patch_size * patch_size).
                      Each row is a flattened patch.
    """
    # unfold directly handles the batch dimension (N)
    # Input shape: (N, C, H, W)
    # Output shape: (N, C * K_h * K_w, L)
    #   where L is the total number of sliding blocks (patches) per image
    patches_unfolded = F.unfold(imgs, kernel_size=patch_size, stride=stride)

    # Reshape the output to (N * L, C * K_h * K_w)
    # N is batch size, L is num_patches_per_image, C*K_h*K_w is patch_dim
    # We want to stack all patches from all images in the batch into a single long list.
    # patches_unfolded.transpose(1, 2) changes (N, C*K*K, L) to (N, L, C*K*K)
    # .reshape(-1, C*K*K) then flattens across N and L, resulting in (N*L, C*K*K)
    
    # Get the patch dimension (C * K_h * K_w)
    patch_dim = patches_unfolded.shape[1] 
    
    # Transpose and reshape to get (B * num_patches_per_image, C * patch_size * patch_size)
    patches_flat = patches_unfolded.transpose(1, 2).reshape(-1, patch_dim) 
    return patches_flat



def sliced_wasserstein_distance(real_patches_flat, fake_patches_flat, num_projections=128):
    """
    Computes the Sliced Wasserstein Distance between two batches of *flattened* patches.

    Args:
        real_patches_flat (torch.Tensor): Tensor of real patches, shape (num_real_patches, patch_dim)
        fake_patches_flat (torch.Tensor): Tensor of fake patches, shape (num_fake_patches, patch_dim)
        num_projections (int): Number of random 1D projections.
    """
    # Verify that the patch dimensions match
    if real_patches_flat.shape[1] != fake_patches_flat.shape[1]:
        raise ValueError("The flattened patch dimensions of real and fake patches must match.")

    patch_dimension = real_patches_flat.shape[1] # This is C * patch_size * patch_size

    # Generate random projection vectors (each column is a projection direction)
    projections = torch.randn(patch_dimension, num_projections).to(real_patches_flat.device)

    # Normalize each projection vector to have unit length
    projections /= torch.sqrt(torch.sum(projections**2, dim=0, keepdim=True))

    # Project the patches onto the random lines
    # (num_patches, patch_dim) @ (patch_dim, num_projections) -> (num_patches, num_projections)
    real_projections = torch.matmul(real_patches_flat, projections)
    fake_projections = torch.matmul(fake_patches_flat, projections)

    # Sort the projected values along each projection (column-wise)
    sorted_real_projections, _ = torch.sort(real_projections, dim=0)
    sorted_fake_projections, _ = torch.sort(fake_projections, dim=0)

    # Compute the average L1 distance between the sorted projected samples
    return torch.mean(torch.abs(sorted_real_projections - sorted_fake_projections))
