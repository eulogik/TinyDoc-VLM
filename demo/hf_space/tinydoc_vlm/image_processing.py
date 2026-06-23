import numpy as np
from PIL import Image
from typing import Optional, List
import torch
import torchvision.transforms as T
from transformers.image_processing_base import ImageProcessingMixin

class TinyDocImageProcessor(ImageProcessingMixin):
    """
    Image processor for TinyDoc-VLM.
    Handles resizing, normalization, and optional tiling (splitting) of document images.
    """
    def __init__(
        self,
        image_size: int = 384,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        tiling_mode: str = "auto",  # "none", "auto" (split if large)
        **kwargs,
    ):
        self.image_size = image_size
        self.mean = mean or [0.5, 0.5, 0.5]
        self.std = std or [0.5, 0.5, 0.5]
        self.tiling_mode = tiling_mode

        super().__init__(**kwargs)
        
        # Base torchvision transforms for single tile
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

    def preprocess(
        self, 
        image: Image.Image, 
        return_tensors: str = "pt"
    ) -> torch.Tensor:
        """
        Preprocesses a PIL Image into a multi-tile float tensor.
        
        Returns shape: (num_tiles, 3, image_size, image_size)
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        w, h = image.size
        
        if self.tiling_mode == "none" or (w <= self.image_size and h <= self.image_size):
            # No tiling needed: resize to image_size x image_size and return single tile
            resized = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            tile_tensor = self.transform(resized)  # shape (3, image_size, image_size)
            # Add tile batch dimension: shape (1, 3, image_size, image_size)
            return tile_tensor.unsqueeze(0)
            
        # Tiling mode 'auto': split high-res image into a grid of image_size x image_size tiles,
        # plus a downscaled overview thumbnail.
        
        # Calculate how many tiles we need
        cols = int(np.ceil(w / self.image_size))
        rows = int(np.ceil(h / self.image_size))
        
        # Limit grid size to prevent excessive memory usage (max 2x2 grid = 4 tiles)
        cols = min(cols, 2)
        rows = min(rows, 2)
        
        # Target size for the tiling grid
        target_w = cols * self.image_size
        target_h = rows * self.image_size
        
        # Resize original image to fit the target grid shape (maintaining proportions via padding)
        resized_full = self._resize_and_pad(image, target_w, target_h)
        
        tiles = []
        
        # 1. Generate thumbnail/overview of the full image
        thumbnail = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        tiles.append(self.transform(thumbnail))
        
        # 2. Extract tiles from the grid
        for r in range(rows):
            for c in range(cols):
                box = (
                    c * self.image_size,
                    r * self.image_size,
                    (c + 1) * self.image_size,
                    (r + 1) * self.image_size
                )
                tile = resized_full.crop(box)
                tiles.append(self.transform(tile))
                
        # Stack tiles along a new dimension: shape (num_tiles, 3, image_size, image_size)
        # where num_tiles = 1 (overview) + rows * cols
        stacked_tiles = torch.stack(tiles, dim=0)
        return stacked_tiles

    def _resize_and_pad(self, img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """
        Resizes and pads an image to target dimensions while maintaining aspect ratio.
        """
        # Calculate aspect ratio
        w, h = img.size
        ratio = min(target_w / w, target_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        # Create a new padded background image
        padded = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        # Center the resized image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded.paste(resized, (x_offset, y_offset))
        
        return padded
