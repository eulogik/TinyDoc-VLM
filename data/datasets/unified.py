"""
Unified dataset interface for TinyDoc-VLM training.
Maps all document understanding datasets to a common conversation format.

Common format:
{
    "image": PIL.Image,
    "conversations": [
        {"role": "user", "content": "<image>\nExtract all fields as JSON."},
        {"role": "assistant", "content": "{\"vendor\": \"...\", ...}"}
    ],
    "metadata": {"source": "docvqa", "task": "qa", "language": "en"}
}
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from PIL import Image
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample dataclass (plain dict with type alias for readability)
# ---------------------------------------------------------------------------
Sample = Dict  # {"image": PIL.Image, "conversations": [...], "metadata": {...}}


def make_sample(
    image: Image.Image,
    user_text: str,
    assistant_text: str,
    source: str,
    task: str,
    language: str = "en",
    extra_meta: Optional[Dict] = None,
) -> Sample:
    """Utility to build a normalised sample dict."""
    meta = {"source": source, "task": task, "language": language}
    if extra_meta:
        meta.update(extra_meta)
    return {
        "image": image,
        "conversations": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": meta,
    }


# ---------------------------------------------------------------------------
# DocVQA
# ---------------------------------------------------------------------------
class DocVQADataset(Dataset):
    """
    DocVQA visual question answering dataset.

    Expected directory layout:
        data_dir/
            train_v1.0.json  (or val_v1.0.json / test_v1.0.json)
            documents/       (PNG / JPG images)

    JSON schema: {"data": [{"questionId": ..., "question": ...,
                             "answers": [...], "image": "filename.png"}]}
    """

    PROMPTS = [
        "<image>\n{question}",
        "<image>\nAnswer the following question about this document: {question}",
        "Look at this document <image> and answer: {question}",
    ]

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split

        ann_file = self.data_dir / f"{split}_v1.0.json"
        if not ann_file.exists():
            ann_file = self.data_dir / f"{split}.json"
        if not ann_file.exists():
            logger.warning(f"DocVQA annotation file not found at {ann_file}")
            self.samples: List[Dict] = []
            return

        with open(ann_file) as f:
            data = json.load(f)
        self.samples = data.get("data", data) if isinstance(data, dict) else data

        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Sample]:
        item = self.samples[idx]
        image_name = item.get("image", item.get("image_name", ""))
        image_path = self.data_dir / "documents" / image_name

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"DocVQA image not found: {image_path} ({e})")
            return None

        question = item["question"]
        # Use first answer for training; evaluation uses all answers
        answers = item.get("answers", item.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]
        answer = answers[0] if answers else ""

        prompt = random.choice(self.PROMPTS).format(question=question)
        return make_sample(
            image=image,
            user_text=prompt,
            assistant_text=answer,
            source="docvqa",
            task="qa",
            extra_meta={"question_id": item.get("questionId"), "all_answers": answers},
        )


# ---------------------------------------------------------------------------
# FUNSD (Form Understanding in Noisy Scanned Documents)
# ---------------------------------------------------------------------------
class FUNSDDataset(Dataset):
    """
    FUNSD entity linking / form understanding dataset.

    Expected layout:
        data_dir/
            training_data/ (or testing_data/)
                annotations/  *.json
                images/       *.png

    Task: extract key-value pairs from form images.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        split_dir = "training_data" if split == "train" else "testing_data"
        self.ann_dir = self.data_dir / split_dir / "annotations"
        self.img_dir = self.data_dir / split_dir / "images"

        if not self.ann_dir.exists():
            logger.warning(f"FUNSD annotations dir not found: {self.ann_dir}")
            self.ann_files: List[Path] = []
            return

        self.ann_files = sorted(self.ann_dir.glob("*.json"))
        if max_samples:
            self.ann_files = self.ann_files[:max_samples]

    def __len__(self) -> int:
        return len(self.ann_files)

    def __getitem__(self, idx: int) -> Optional[Sample]:
        ann_file = self.ann_files[idx]
        image_path = self.img_dir / (ann_file.stem + ".png")

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError):
            return None

        with open(ann_file) as f:
            ann = json.load(f)

        # Collect all entity key-value pairs
        kv_pairs: Dict[str, str] = {}
        for form_item in ann.get("form", []):
            label = form_item.get("label", "other")
            text = form_item.get("text", "").strip()
            if label not in ("other", "") and text:
                kv_pairs[label] = text

        assistant_text = json.dumps(kv_pairs, ensure_ascii=False) if kv_pairs else "{}"
        return make_sample(
            image=image,
            user_text="<image>\nExtract all form fields as JSON key-value pairs.",
            assistant_text=assistant_text,
            source="funsd",
            task="extraction",
            extra_meta={"file": ann_file.stem},
        )


# ---------------------------------------------------------------------------
# CORD (Consolidated Receipt Dataset)
# ---------------------------------------------------------------------------
class CORDDataset(Dataset):
    """
    CORD receipt key-information extraction dataset.

    Expected layout:
        data_dir/
            train/ (or dev/ or test/)
                image/  *.png
                json/   *.json
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        split_map = {"train": "train", "val": "dev", "test": "test"}
        split_dir = self.data_dir / split_map.get(split, split)

        self.img_dir = split_dir / "image"
        self.ann_dir = split_dir / "json"

        if not self.img_dir.exists():
            logger.warning(f"CORD image dir not found: {self.img_dir}")
            self.image_files: List[Path] = []
            return

        self.image_files = sorted(self.img_dir.glob("*.png"))
        if max_samples:
            self.image_files = self.image_files[:max_samples]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Optional[Sample]:
        img_path = self.image_files[idx]
        ann_path = self.ann_dir / (img_path.stem + ".json")

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            return None

        if not ann_path.exists():
            return None

        with open(ann_path) as f:
            ann = json.load(f)

        # Flatten CORD's nested structure into a simple dict
        valid_line = ann.get("valid_line", [])
        extracted: Dict[str, str] = {}
        for line in valid_line:
            category = line.get("category", "")
            words = " ".join(w.get("quad", {}).get("text", w.get("text", "")) for w in line.get("words", []))
            if category and words.strip():
                extracted[category] = words.strip()

        # Also include summary totals
        gt = ann.get("gt_parse", {})
        if gt:
            extracted.update({k: str(v) for k, v in gt.items() if v})

        assistant_text = json.dumps(extracted, ensure_ascii=False)
        return make_sample(
            image=image,
            user_text="<image>\nExtract all receipt information as JSON.",
            assistant_text=assistant_text,
            source="cord",
            task="extraction",
        )


# ---------------------------------------------------------------------------
# SROIE (Scanned Receipts OCR and Information Extraction)
# ---------------------------------------------------------------------------
class SROIEDataset(Dataset):
    """
    SROIE receipt key-information extraction (4 fields: company, date, address, total).

    Expected layout:
        data_dir/
            0325updated.task2train/ (or test/)
                *.jpg
                *.txt  (JSON annotation)
    """

    FIELDS = ["company", "date", "address", "total"]

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        split_dir_name = "0325updated.task2train" if split == "train" else "task2-test"
        self.split_dir = self.data_dir / split_dir_name

        if not self.split_dir.exists():
            # Try alternate structure
            self.split_dir = self.data_dir / split
            if not self.split_dir.exists():
                logger.warning(f"SROIE data dir not found: {self.split_dir}")
                self.samples: List[Path] = []
                return

        self.samples = sorted(self.split_dir.glob("*.jpg")) + sorted(self.split_dir.glob("*.png"))
        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Sample]:
        img_path = self.samples[idx]
        ann_path = img_path.with_suffix(".txt")

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            return None

        extracted: Dict[str, str] = {}
        if ann_path.exists():
            try:
                with open(ann_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                # SROIE annotation files are JSON-formatted key-value
                ann = json.loads(content)
                extracted = {k: str(ann.get(k, "")) for k in self.FIELDS}
            except (json.JSONDecodeError, Exception):
                pass

        assistant_text = json.dumps(extracted, ensure_ascii=False)
        return make_sample(
            image=image,
            user_text="<image>\nExtract company name, date, address and total amount from this receipt as JSON.",
            assistant_text=assistant_text,
            source="sroie",
            task="extraction",
        )


# ---------------------------------------------------------------------------
# PubTabNet (Table structure recognition)
# ---------------------------------------------------------------------------
class PubTabNetDataset(Dataset):
    """
    PubTabNet table structure recognition dataset.
    Converts table images to HTML markup.

    Expected layout:
        data_dir/
            PubTabNet_2.0.0.jsonl  (annotation file)
            images/
                *.png
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        ann_path = self.data_dir / "PubTabNet_2.0.0.jsonl"

        if not ann_path.exists():
            logger.warning(f"PubTabNet annotation file not found: {ann_path}")
            self.samples: List[Dict] = []
            return

        self.samples = []
        with open(ann_path) as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if item.get("split", "train") == split:
                        self.samples.append(item)
                except json.JSONDecodeError:
                    continue

        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Sample]:
        item = self.samples[idx]
        image_path = self.data_dir / "images" / item["filename"]

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError):
            return None

        # Build HTML from the annotation tokens + structure
        html_struct = item.get("html", {})
        structure = html_struct.get("structure", {}).get("tokens", [])
        cells = html_struct.get("cells", [])

        html = self._reconstruct_html(structure, cells)
        return make_sample(
            image=image,
            user_text="<image>\nConvert this table to HTML markup.",
            assistant_text=html,
            source="pubtabnet",
            task="table_extraction",
        )

    @staticmethod
    def _reconstruct_html(structure: List[str], cells: List[Dict]) -> str:
        """Reconstruct HTML table from PubTabNet structure tokens and cell content."""
        cell_idx = 0
        html_parts = ["<table>"]
        for token in structure:
            if token == "<td>":
                if cell_idx < len(cells):
                    cell_tokens = cells[cell_idx].get("tokens", [])
                    cell_text = "".join(cell_tokens)
                    html_parts.append(f"<td>{cell_text}</td>")
                    cell_idx += 1
                else:
                    html_parts.append("<td></td>")
            else:
                html_parts.append(token)
        html_parts.append("</table>")
        return "".join(html_parts)


# ---------------------------------------------------------------------------
# Synthetic manifest dataset (already exists)
# ---------------------------------------------------------------------------
class SyntheticDocDataset(Dataset):
    """
    Loads pre-generated synthetic documents from manifest.jsonl.
    Converts each sample into the unified conversation format.
    """

    QA_PROMPTS = [
        "<image>\nExtract all structured information from this document as JSON.",
        "<image>\nWhat information can you extract from this document?",
        "<image>\nIdentify all key fields in this document.",
    ]

    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_root: Union[str, Path],
        max_samples: Optional[int] = None,
        use_qa_pairs: bool = True,
    ):
        self.data_root = Path(data_root)
        self.use_qa_pairs = use_qa_pairs

        with open(manifest_path) as f:
            raw = [json.loads(line) for line in f if line.strip()]

        # Filter out failed renders
        self.samples = [s for s in raw if s.get("image_path") and not s.get("error")]
        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Sample]:
        item = self.samples[idx]
        image_path = self.data_root / item["image_path"]

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Synthetic image not found: {image_path} ({e})")
            return None

        # Use a QA pair if available and randomly selected
        qa_pairs = item.get("qa_pairs", [])
        if self.use_qa_pairs and qa_pairs and random.random() > 0.5:
            pair = random.choice(qa_pairs)
            user_text = f"<image>\n{pair['question']}"
            assistant_text = str(pair["answer"])
            task = "qa"
        else:
            # Default: extract as JSON
            metadata = {k: v for k, v in item.get("metadata", {}).items()
                        if not isinstance(v, list)}
            user_text = random.choice(self.QA_PROMPTS)
            assistant_text = json.dumps(metadata, ensure_ascii=False)
            task = "extraction"

        return make_sample(
            image=image,
            user_text=user_text,
            assistant_text=assistant_text,
            source="synthetic",
            task=task,
            extra_meta={"doc_type": item.get("doc_type")},
        )


# ---------------------------------------------------------------------------
# Unified Dataset (mix of all sources)
# ---------------------------------------------------------------------------
class UnifiedDocDataset(Dataset):
    """
    Mixes multiple document datasets with configurable proportions.

    Usage:
        dataset = UnifiedDocDataset(
            sources={
                "synthetic": (SyntheticDocDataset(...), 0.40),
                "docvqa":    (DocVQADataset(...),       0.20),
                "cord":      (CORDDataset(...),         0.15),
                "funsd":     (FUNSDDataset(...),        0.10),
                "sroie":     (SROIEDataset(...),        0.10),
                "pubtabnet": (PubTabNetDataset(...),    0.05),
            }
        )
    """

    def __init__(
        self,
        sources: Dict[str, tuple],  # name -> (Dataset, weight)
        total_samples: int = 100_000,
        seed: int = 42,
    ):
        self.datasets: List[Dataset] = []
        self.index_map: List[tuple] = []  # (dataset_idx, sample_idx)

        rng = random.Random(seed)
        names = list(sources.keys())
        datasets_list = [sources[n][0] for n in names]
        weights = [sources[n][1] for n in names]

        # Normalize weights
        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]

        self.datasets = datasets_list
        counts = [max(1, int(total_samples * w)) for w in norm_weights]

        for ds_idx, (ds, count) in enumerate(zip(datasets_list, counts)):
            n = len(ds)
            if n == 0:
                continue
            # Sample with replacement if count > n
            indices = [rng.randint(0, n - 1) for _ in range(count)]
            for s_idx in indices:
                self.index_map.append((ds_idx, s_idx))

        rng.shuffle(self.index_map)
        logger.info(
            f"UnifiedDocDataset: {len(self.index_map)} samples from "
            f"{len([d for d in datasets_list if len(d) > 0])} sources"
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Optional[Sample]:
        ds_idx, s_idx = self.index_map[idx]
        return self.datasets[ds_idx][s_idx]


def collate_unified(batch: List[Optional[Sample]], tokenizer, image_processor, max_length: int = 2048) -> Dict:
    """
    Collate function for UnifiedDocDataset.
    Converts PIL images and conversation texts into tensors.
    """
    import torch

    # Filter None samples
    batch = [s for s in batch if s is not None]
    if not batch:
        return {}

    # Process images
    pixel_values_list = []
    num_tiles_list = []
    for sample in batch:
        tile_tensor = image_processor.preprocess(sample["image"])
        pixel_values_list.append(tile_tensor)
        num_tiles_list.append(tile_tensor.shape[0])

    max_tiles = max(num_tiles_list)
    sz = image_processor.image_size
    padded = []
    for pv in pixel_values_list:
        T = pv.shape[0]
        if T < max_tiles:
            pad = torch.zeros(max_tiles - T, 3, sz, sz, dtype=pv.dtype)
            pv = torch.cat([pv, pad], dim=0)
        padded.append(pv)
    pixel_values = torch.stack(padded, dim=0)

    # Build prompt texts from conversations
    scale = 3
    patch_size = 16
    tokens_per_tile = (sz // patch_size // scale) ** 2
    image_token = "<image>"

    texts = []
    label_texts = []
    for sample, n_tiles in zip(batch, num_tiles_list):
        convs = sample["conversations"]
        user_msg = convs[0]["content"] if convs else ""
        asst_msg = convs[1]["content"] if len(convs) > 1 else ""

        # Expand <image> tokens
        total_visual = n_tiles * tokens_per_tile
        user_expanded = user_msg.replace(image_token, image_token * total_visual)
        texts.append(user_expanded + " " + asst_msg)
        label_texts.append(asst_msg)

    # Tokenize
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "pixel_values": pixel_values,
        "labels": labels,
    }
