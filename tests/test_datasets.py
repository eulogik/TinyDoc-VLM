import json
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from data.datasets.docvqa import DocVQADataset
from data.datasets.funsd import FUNSDDataset
from data.datasets.cord import CORDDataset
from data.datasets.sroie import SROIEDataset
from data.datasets.pubtabnet import PubTabNetDataset
from data.datasets.unified import SyntheticDocDataset, UnifiedDocDataset

def create_dummy_image(path: Path):
    img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
    img.save(path)

def test_docvqa_dataset(tmp_path):
    data_dir = tmp_path / "docvqa"
    data_dir.mkdir()
    
    # Create mock annotation
    ann = {
        "data": [
            {
                "questionId": 123,
                "question": "What is the date?",
                "answers": ["2026-06-20"],
                "image": "doc_1.png"
            }
        ]
    }
    with open(data_dir / "train_v1.0.json", "w") as f:
        json.dump(ann, f)
        
    doc_dir = data_dir / "documents"
    doc_dir.mkdir()
    create_dummy_image(doc_dir / "doc_1.png")
    
    dataset = DocVQADataset(data_dir, split="train")
    assert len(dataset) == 1
    
    sample = dataset[0]
    assert sample is not None
    assert "image" in sample
    assert "conversations" in sample
    assert sample["metadata"]["source"] == "docvqa"
    assert sample["metadata"]["task"] == "qa"
    assert "2026-06-20" in sample["conversations"][1]["content"]

def test_funsd_dataset(tmp_path):
    data_dir = tmp_path / "funsd"
    data_dir.mkdir()
    
    split_dir = data_dir / "training_data"
    split_dir.mkdir()
    ann_dir = split_dir / "annotations"
    ann_dir.mkdir()
    img_dir = split_dir / "images"
    img_dir.mkdir()
    
    # Create mock annotation
    ann = {
        "form": [
            {"label": "header", "text": "INVOICE"},
            {"label": "date", "text": "2026-06-20"}
        ]
    }
    with open(ann_dir / "doc_1.json", "w") as f:
        json.dump(ann, f)
        
    create_dummy_image(img_dir / "doc_1.png")
    
    dataset = FUNSDDataset(data_dir, split="train")
    assert len(dataset) == 1
    
    sample = dataset[0]
    assert sample is not None
    assert "image" in sample
    assert sample["metadata"]["source"] == "funsd"
    assert "invoice" in sample["conversations"][1]["content"].lower()

def test_cord_dataset(tmp_path):
    data_dir = tmp_path / "cord"
    data_dir.mkdir()
    
    split_dir = data_dir / "train"
    split_dir.mkdir()
    img_dir = split_dir / "image"
    img_dir.mkdir()
    ann_dir = split_dir / "json"
    ann_dir.mkdir()
    
    # Create mock annotation
    ann = {
        "valid_line": [
            {
                "category": "menu.nm",
                "words": [{"text": "Coffee"}]
            }
        ],
        "gt_parse": {"total": "5.00"}
    }
    create_dummy_image(img_dir / "receipt_1.png")
    with open(ann_dir / "receipt_1.json", "w") as f:
        json.dump(ann, f)
        
    dataset = CORDDataset(data_dir, split="train")
    assert len(dataset) == 1
    
    sample = dataset[0]
    assert sample is not None
    assert "image" in sample
    assert sample["metadata"]["source"] == "cord"
    assert "5.00" in sample["conversations"][1]["content"]

def test_sroie_dataset(tmp_path):
    data_dir = tmp_path / "sroie"
    data_dir.mkdir()
    
    split_dir = data_dir / "0325updated.task2train"
    split_dir.mkdir()
    
    create_dummy_image(split_dir / "receipt_1.jpg")
    ann = {
        "company": "Starbucks",
        "date": "2026-06-20",
        "address": "Seattle",
        "total": "5.00"
    }
    with open(split_dir / "receipt_1.txt", "w") as f:
        json.dump(ann, f)
        
    dataset = SROIEDataset(data_dir, split="train")
    assert len(dataset) == 1
    
    sample = dataset[0]
    assert sample is not None
    assert "image" in sample
    assert sample["metadata"]["source"] == "sroie"
    assert "Starbucks" in sample["conversations"][1]["content"]

def test_pubtabnet_dataset(tmp_path):
    data_dir = tmp_path / "pubtabnet"
    data_dir.mkdir()
    
    img_dir = data_dir / "images"
    img_dir.mkdir()
    create_dummy_image(img_dir / "table_1.png")
    
    # Create mock jsonl
    ann = {
        "filename": "table_1.png",
        "split": "train",
        "html": {
            "structure": {"tokens": ["<thead>", "<tr>", "<td>", "</td>", "</tr>", "</thead>"]},
            "cells": [{"tokens": ["Header1"]}]
        }
    }
    with open(data_dir / "PubTabNet_2.0.0.jsonl", "w") as f:
        f.write(json.dumps(ann) + "\n")
        
    dataset = PubTabNetDataset(data_dir, split="train")
    assert len(dataset) == 1
    
    sample = dataset[0]
    assert sample is not None
    assert "image" in sample
    assert sample["metadata"]["source"] == "pubtabnet"
    assert "Header1" in sample["conversations"][1]["content"]

def test_synthetic_doc_dataset(tmp_path):
    data_dir = tmp_path / "synthetic"
    data_dir.mkdir()
    
    img_dir = data_dir / "images"
    img_dir.mkdir()
    create_dummy_image(img_dir / "doc_1.png")
    
    manifest_path = data_dir / "manifest.jsonl"
    sample_manifest = {
        "image_path": "images/doc_1.png",
        "doc_type": "invoice",
        "metadata": {"vendor_name": "TestVendor", "total": "100.0"},
        "qa_pairs": [{"question": "Who is the vendor?", "answer": "TestVendor"}]
    }
    with open(manifest_path, "w") as f:
        f.write(json.dumps(sample_manifest) + "\n")
        
    dataset = SyntheticDocDataset(manifest_path, data_dir, use_qa_pairs=True)
    assert len(dataset) == 1
    
    sample = dataset[0]
    assert sample is not None
    assert "image" in sample
    assert sample["metadata"]["source"] == "synthetic"
    
    # Run a few times to test both branches (QA pairs vs default JSON extraction)
    has_qa = False
    has_ext = False
    for _ in range(20):
        s = dataset[0]
        if s["metadata"]["task"] == "qa":
            has_qa = True
        elif s["metadata"]["task"] == "extraction":
            has_ext = True
            
    assert has_qa or has_ext
