import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .image_processing import TinyDocImageProcessor


class DocumentDataset(Dataset):
    """
    Dataset for document understanding training.
    Supports loading from a JSON manifest file or from individual samples.

    Manifest format (JSONL):
        {"image_path": "path/to/image.png", "text": "Extract: <image>", "labels": {...}}
    """
    def __init__(
        self,
        data_root: Union[str, Path],
        manifest_path: Optional[Union[str, Path]] = None,
        image_processor: Optional[TinyDocImageProcessor] = None,
        max_seq_length: int = 2048,
        stage: int = 1,
        samples: Optional[List[Dict]] = None,
    ):
        self.data_root = Path(data_root)
        self.image_processor = image_processor or TinyDocImageProcessor()
        self.max_seq_length = max_seq_length
        self.stage = stage

        if samples is not None:
            self.samples = samples
        elif manifest_path:
            with open(manifest_path) as f:
                self.samples = [json.loads(line) for line in f if line.strip()]
        else:
            self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image_path = self.data_root / sample["image_path"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor.preprocess(image)

        text = sample.get("text", "<image>")
        labels = sample.get("labels", {})

        return {
            "pixel_values": pixel_values,
            "text": text,
            "labels": labels,
            "metadata": sample.get("metadata", {}),
        }


def collate_fn(batch: List[Dict], tokenizer, image_token_id: int, max_length: int = 2048) -> Dict:
    """
    Collate function for DocumentDataset.
    Handles variable-length text, variable-number tiles, and label padding.
    """
    texts = [item["text"] for item in batch]
    images = [item.get("pixel_values") for item in batch]

    max_tiles = max(pv.shape[0] for pv in images)
    image_size = images[0].shape[-1]
    padded_pixel_values = []
    for pv in images:
        num_tiles = pv.shape[0]
        if num_tiles < max_tiles:
            pad = torch.zeros(max_tiles - num_tiles, 3, image_size, image_size, dtype=pv.dtype)
            pv = torch.cat([pv, pad], dim=0)
        padded_pixel_values.append(pv)

    pixel_values = torch.stack(padded_pixel_values, dim=0)

    tokenized = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    labels = tokenized["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "pixel_values": pixel_values,
        "labels": labels,
    }
