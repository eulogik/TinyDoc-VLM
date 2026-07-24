import torch
from torch import nn
from transformers import SiglipVisionModel

from .configuration import TinyDocVLMConfig


class SigLIPVisionEncoder(nn.Module):
    """
    Wrapper around HuggingFace's SiglipVisionModel.
    Handles encoding of multiple image tiles and thumbnails.
    """
    def __init__(self, config: TinyDocVLMConfig):
        super().__init__()
        self.config = config
        
        # Load from config or create model
        vision_config = config.vision_config
        self.encoder = SiglipVisionModel(vision_config)
        self.hidden_size = vision_config.hidden_size
        self.patch_size = getattr(vision_config, "patch_size", 16)
        self.image_size = getattr(vision_config, "image_size", 384)

    def resize_pos_embeddings(self, target_grid_size: int):
        """
        Interpolate the learned positional embeddings to a new square grid
        (target_grid_size x target_grid_size). Call this after loading a
        pretrained SigLIP checkpoint whose resolution differs from the current
        config (e.g. loading a 384px-pretrained encoder into a 768px model).
        """
        pos_embed = self.encoder.embeddings.position_embedding
        old_num = pos_embed.weight.shape[0]
        old_grid = int(old_num ** 0.5)
        if old_grid * old_grid != old_num or old_grid == target_grid_size:
            return
        new_num = target_grid_size * target_grid_size
        # (1, old_num, dim) -> (1, dim, old_grid, old_grid)
        weight = pos_embed.weight.unsqueeze(0).permute(0, 2, 1).view(1, -1, old_grid, old_grid)
        weight = torch.nn.functional.interpolate(
            weight, size=(target_grid_size, target_grid_size), mode="bicubic", align_corners=False
        )
        weight = weight.view(1, -1, new_num).permute(0, 2, 1).squeeze(0)
        pos_embed.weight.data = weight

    def forward(
        self, 
        pixel_values: torch.Tensor, 
        interpolate_pos_encoding: bool = True
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: shape (batch_size, num_tiles, channels, height, width) 
                          or (batch_size * num_tiles, channels, height, width)
            interpolate_pos_encoding: whether to interpolate positional embeddings if resolution changes
            
        Returns:
            visual_features: shape (batch_size, num_tiles, num_patches, hidden_size)
        """
        # If input has shape (batch_size, num_tiles, channels, height, width)
        if len(pixel_values.shape) == 5:
            batch_size, num_tiles, channels, height, width = pixel_values.shape
            # Flatten batch and tiles for vision encoder
            pixel_values = pixel_values.view(batch_size * num_tiles, channels, height, width)
        else:
            # Assumed to be already flattened (batch_size * num_tiles, channels, height, width)
            batch_size = 1
            num_tiles = pixel_values.shape[0]
            channels, height, width = pixel_values.shape[1:]

        # Run through SigLIP Vision Model
        outputs = self.encoder(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding
        )
        
        # Last hidden state: (batch_size * num_tiles, num_patches, hidden_size)
        # For SigLIP-B/16 with 384x384 input: num_patches = (384/16)^2 = 24^2 = 576
        last_hidden_state = outputs.last_hidden_state
        
        # Reshape back to batch format
        num_patches = last_hidden_state.shape[1]
        last_hidden_state = last_hidden_state.view(batch_size, num_tiles, num_patches, self.hidden_size)
        
        return last_hidden_state
