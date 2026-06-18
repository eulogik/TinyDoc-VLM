import math
import torch
import torch.nn as nn

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> torch.Tensor:
    """
    Generate 2D sinusoidal positional embeddings.
    
    Args:
        embed_dim: Dimension of the embedding (must be even)
        grid_size: Height/Width of the grid (assumed square)
        cls_token: If True, prepends a zero embedding for the class token
        
    Returns:
        pos_embed: shape (grid_size * grid_size, embed_dim) or (1 + grid_size * grid_size, embed_dim)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    
    # Create coordinate grid
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)  # shape (2, grid_size, grid_size)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token:
        # Prepend a zero embedding for CLS token
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
        
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    """
    Helper function to generate embeddings from a coordinate grid.
    
    Args:
        embed_dim: Dimension of the embedding
        grid: shape (2, 1, grid_h, grid_w) where index 0 is Y, index 1 is X
        
    Returns:
        emb: shape (grid_h * grid_w, embed_dim)
    """
    assert embed_dim % 2 == 0
    
    # Use half of dimensions for X, half for Y
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # Y coords
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # X coords
    
    emb = torch.cat([emb_h, emb_w], dim=1)  # shape (grid_h * grid_w, embed_dim)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    Generate 1D sinusoidal positional embeddings for a coordinate tensor.
    
    Args:
        embed_dim: Dimension of the embedding
        pos: Coordinate tensor of shape (1, H, W) or (L,)
        
    Returns:
        emb: shape (H * W, embed_dim) or (L, embed_dim)
    """
    assert embed_dim % 2 == 0
    
    # omega = 1 / (10000 ** (2i / d))
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)  # shape (embed_dim // 2,)
    
    pos = pos.reshape(-1)  # Flatten spatial dims to 1D sequence
    out = torch.outer(pos, omega)  # shape (seq_len, embed_dim // 2)
    
    emb_sine = torch.sin(out)
    emb_cosine = torch.cos(out)
    
    emb = torch.cat([emb_sine, emb_cosine], dim=1)  # shape (seq_len, embed_dim)
    return emb
