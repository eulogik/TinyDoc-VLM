import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional, Tuple
from transformers import SiglipVisionModel, PreTrainedModel, AutoConfig
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
        # We can initialize from config. If we are running pretraining, we load weights.
        # During runtime we might load a pretrained siglip model.
        self.encoder = SiglipVisionModel(vision_config)
        self.hidden_size = vision_config.hidden_size
        
        # Add special region classification or auxiliary layers if needed in future
        # For now, just a wrapper around the SigLIP vision encoder

    def forward(
        self, 
        pixel_values: torch.Tensor, 
        interpolate_pos_encoding: bool = False
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
