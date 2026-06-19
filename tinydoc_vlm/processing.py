from PIL import Image
from typing import Dict, Any, Union, Optional, List, Tuple
import torch
from transformers import AutoTokenizer, ProcessorMixin

from .image_processing import TinyDocImageProcessor

class TinyDocVLMProcessor(ProcessorMixin):
    """
    Processor coordinating TinyDocImageProcessor and AutoTokenizer.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, config=None, **kwargs):
        if image_processor is None:
            image_processor = TinyDocImageProcessor()
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

        self.config = config
            
        super().__init__(image_processor, tokenizer, **kwargs)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        
        # Define image special token and ID
        self.image_token = "<image>"
        # Ensure image token is added to the tokenizer
        if self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.image_token]})
            
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

    def __call__(
        self,
        text: Union[str, List[str]],
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        """
        Preprocesses text and images into tensors for the model.
        
        Args:
            text: A string or list of strings to tokenize.
            images: A PIL image or list of PIL images to process.
            
        Returns:
            inputs: A dictionary containing 'input_ids', 'attention_mask', and 'pixel_values' (if images are provided).
        """
        # 1. Process images if present
        pixel_values = None
        num_tiles_list = []
        
        if images is not None:
            if not isinstance(images, list):
                images = [images]
                
            processed_images = []
            for img in images:
                # Shape: (num_tiles, 3, image_size, image_size)
                tile_tensor = self.image_processor.preprocess(img)
                processed_images.append(tile_tensor)
                num_tiles_list.append(tile_tensor.shape[0])
                
            # If batching multiple images with varying number of tiles, pad them to the max tiles
            max_tiles = max(num_tiles_list)
            padded_images = []
            for tile_tensor in processed_images:
                num_tiles = tile_tensor.shape[0]
                if num_tiles < max_tiles:
                    # Pad with zeros along the tile dimension
                    padding_shape = (max_tiles - num_tiles, 3, self.image_processor.image_size, self.image_processor.image_size)
                    padding_tensor = torch.zeros(padding_shape, dtype=tile_tensor.dtype, device=tile_tensor.device)
                    tile_tensor = torch.cat([tile_tensor, padding_tensor], dim=0)
                padded_images.append(tile_tensor)
                
            # Shape: (batch_size, max_tiles, 3, image_size, image_size)
            pixel_values = torch.stack(padded_images, dim=0)

        # 2. Process text and expand image tokens
        # Each image token needs to be expanded to match the compressed sequence length of its corresponding image.
        # Number of tokens per tile = (image_size / patch_size / scale_factor) ** 2
        # For default 384 size, 16 patch, scale 3: (384/16/3)^2 = 8^2 = 64 tokens.
        scale = getattr(self.config, "pixel_shuffle_scale", 3) if self.config else 3
        patch_size = getattr(self.config, "patch_size", 16) if self.config else 16
        tokens_per_tile = (self.image_processor.image_size // patch_size // scale) ** 2

        if isinstance(text, list):
            expanded_texts = []
            for idx, t in enumerate(text):
                # Calculate number of expansion tokens needed for this sample's image
                if idx < len(num_tiles_list):
                    num_tiles = num_tiles_list[idx]
                    total_visual_tokens = num_tiles * tokens_per_tile
                else:
                    total_visual_tokens = 0
                
                expanded_t = self._expand_image_tokens(t, total_visual_tokens)
                expanded_texts.append(expanded_t)
            processed_text = expanded_texts
        else:
            total_visual_tokens = num_tiles_list[0] * tokens_per_tile if num_tiles_list else 0
            processed_text = self._expand_image_tokens(text, total_visual_tokens)

        # 3. Tokenize text
        tokenized = self.tokenizer(
            processed_text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

        inputs = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values

        # Expose image_token_id in the output so the model can use it
        inputs["image_token_id"] = self.image_token_id

        return inputs

    def _expand_image_tokens(self, text: str, total_tokens: int) -> str:
        """
        Replaces '<image>' with `<image> * total_tokens`.
        """
        expansion = self.image_token * total_tokens
        return text.replace(self.image_token, expansion)
