import math
import torch
import torch.nn as nn
from .configuration import TinyDocVLMConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class PixelShuffleTokenCompressor(nn.Module):
    """
    Performs space-to-depth token compression on Vision Transformer patch sequences.
    Groups scale_factor x scale_factor patches and projects to decoder hidden dimension.
    """
    def __init__(self, config: TinyDocVLMConfig, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.config = config
        self.scale_factor = config.pixel_shuffle_scale
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        # After space-to-depth, channel dimension becomes encoder_dim * scale_factor^2
        compressed_dim = encoder_dim * (self.scale_factor ** 2)
        
        # MLP projection to map visual tokens to language model dimension
        self.projection = nn.Sequential(
            nn.Linear(compressed_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim)
        )
        self.norm = RMSNorm(decoder_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: shape (batch_size, num_tiles, num_patches, encoder_dim)
            
        Returns:
            compressed_features: shape (batch_size, num_tiles, num_compressed_patches, decoder_dim)
        """
        batch_size, num_tiles, num_patches, encoder_dim = visual_features.shape
        
        # Determine spatial dimensions assuming a square grid of patches
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"Number of patches ({num_patches}) must be a perfect square to apply 2D pixel shuffle."
            )
            
        if grid_size % self.scale_factor != 0:
            raise ValueError(
                f"Grid size ({grid_size}) must be divisible by pixel_shuffle_scale ({self.scale_factor})."
            )

        # Reshape to 2D spatial grid: (batch_size * num_tiles, grid_size, grid_size, encoder_dim)
        x = visual_features.view(batch_size * num_tiles, grid_size, grid_size, encoder_dim)
        
        # Apply space-to-depth: (batch_size * num_tiles, H//s, s, W//s, s, C)
        s = self.scale_factor
        x = x.view(batch_size * num_tiles, grid_size // s, s, grid_size // s, s, encoder_dim)
        
        # Permute: (batch_size * num_tiles, H//s, W//s, s, s, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        
        # Reshape to flatten the spatial groups into the channel dimension:
        # (batch_size * num_tiles, (H//s) * (W//s), s * s * C)
        new_patches = (grid_size // s) ** 2
        x = x.view(batch_size * num_tiles, new_patches, s * s * encoder_dim)
        
        # Project and normalize
        x = self.projection(x)
        x = self.norm(x)
        
        # Reshape back to batch: (batch_size, num_tiles, new_patches, decoder_dim)
        x = x.view(batch_size, num_tiles, new_patches, self.decoder_dim)
        return x
