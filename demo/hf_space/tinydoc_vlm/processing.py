"""
TinyDocVLMProcessor — standalone processor (does not inherit ProcessorMixin to avoid
strict type-checking issues in transformers<4.45). Provides the same public API.
"""
import os
import json
from PIL import Image
from typing import Dict, Any, Union, Optional, List

import torch
from transformers import AutoTokenizer

from .image_processing import TinyDocImageProcessor


class TinyDocVLMProcessor:
    """
    Coordinates TinyDocImageProcessor (image tiling + normalisation) and the
    SmolLM2 tokenizer extended with document-special tokens.

    Usage:
        processor = TinyDocVLMProcessor()
        inputs = processor(text=["Extract fields: <image>"], images=[pil_img])
        # inputs → {"input_ids", "attention_mask", "pixel_values", "image_token_id"}
    """

    # Class-level attrs used by some HF utilities (save_pretrained etc.)
    image_processor_class = "TinyDocImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Optional[TinyDocImageProcessor] = None,
        tokenizer=None,
        config=None,
        **kwargs,
    ):
        self.image_processor = image_processor or TinyDocImageProcessor()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct"
        )
        self.config = config

        # Ensure <image> special token exists
        self.image_token = "<image>"
        if self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.image_token]}
            )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

    # ------------------------------------------------------------------
    # Core __call__
    # ------------------------------------------------------------------

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

        Returns:
            dict with keys: input_ids, attention_mask, pixel_values (optional),
                            image_token_id.
        """
        # ---- 1. Image processing ------------------------------------------------
        pixel_values = None
        num_tiles_list: List[int] = []

        if images is not None:
            if not isinstance(images, list):
                images = [images]

            processed: List[torch.Tensor] = []
            for img in images:
                tile_tensor = self.image_processor.preprocess(img)  # (T, 3, H, W)
                processed.append(tile_tensor)
                num_tiles_list.append(tile_tensor.shape[0])

            # Pad to max tiles so we can stack into a single tensor
            max_tiles = max(num_tiles_list)
            padded: List[torch.Tensor] = []
            sz = self.image_processor.image_size
            for tile_tensor in processed:
                T = tile_tensor.shape[0]
                if T < max_tiles:
                    pad = torch.zeros(
                        (max_tiles - T, 3, sz, sz),
                        dtype=tile_tensor.dtype,
                        device=tile_tensor.device,
                    )
                    tile_tensor = torch.cat([tile_tensor, pad], dim=0)
                padded.append(tile_tensor)

            pixel_values = torch.stack(padded, dim=0)  # (B, max_tiles, 3, H, W)

        # ---- 2. Expand <image> tokens in text ----------------------------------
        scale = (
            getattr(self.config, "pixel_shuffle_scale", 3) if self.config else 3
        )
        patch_size = (
            getattr(self.config, "patch_size", 16) if self.config else 16
        )
        sz = self.image_processor.image_size
        tokens_per_tile = (sz // patch_size // scale) ** 2

        if isinstance(text, list):
            expanded: List[str] = []
            for idx, t in enumerate(text):
                if idx < len(num_tiles_list):
                    total_vis = num_tiles_list[idx] * tokens_per_tile
                else:
                    total_vis = 0
                expanded.append(self._expand_image_tokens(t, total_vis))
            processed_text = expanded
        else:
            total_vis = num_tiles_list[0] * tokens_per_tile if num_tiles_list else 0
            processed_text = self._expand_image_tokens(text, total_vis)

        # ---- 3. Tokenise -------------------------------------------------------
        enc = self.tokenizer(
            processed_text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

        inputs: Dict[str, Any] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values
        inputs["image_token_id"] = self.image_token_id

        return inputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expand_image_tokens(self, text: str, total_tokens: int) -> str:
        """Replaces the single '<image>' placeholder with `total_tokens` copies."""
        expansion = self.image_token * total_tokens
        return text.replace(self.image_token, expansion)

    # ------------------------------------------------------------------
    # save / from_pretrained stubs (for HF Hub compatibility)
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.image_processor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        # Write a minimal processor_config.json
        cfg = {
            "processor_class": "TinyDocVLMProcessor",
            "image_token": self.image_token,
            "image_token_id": self.image_token_id,
        }
        with open(os.path.join(save_directory, "processor_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        image_processor = TinyDocImageProcessor.from_pretrained(
            pretrained_model_name_or_path
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        return cls(image_processor=image_processor, tokenizer=tokenizer, **kwargs)
