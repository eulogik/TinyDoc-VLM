import torch
import torch.nn as nn
from typing import Dict

class JSONHead(nn.Module):
    """
    Structured JSON generation head.
    Projects decoder hidden states to schema-constrained JSON token logits.
    In practice, this works as a specialized classifier over JSON structural tokens
    plus a pointer network for field values.
    """
    def __init__(self, hidden_size: int, num_json_tokens: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_json_tokens),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class KVHead(nn.Module):
    """
    Key-Value extraction head.
    Produces key-value pairs from decoder hidden states with confidence scores.
    Uses two separate projections: one for key detection, one for value extraction.
    """
    def __init__(self, hidden_size: int, num_keys: int = 128):
        super().__init__()
        self.key_classifier = nn.Linear(hidden_size, num_keys)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.confidence = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        key_logits = self.key_classifier(hidden_states)
        value_embeds = self.value_proj(hidden_states)
        conf = torch.sigmoid(self.confidence(hidden_states))
        return {
            "key_logits": key_logits,
            "value_embeds": value_embeds,
            "confidence": conf,
        }


class TableHead(nn.Module):
    """
    Table generation head.
    Outputs HTML/Markdown table tokens via a specialized projection.
    Can also output structured row/cell coordinates.
    """
    def __init__(self, hidden_size: int, num_table_tokens: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_table_tokens),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class OCRHead(nn.Module):
    """
    OCR text generation head.
    Decodes visual features into character-level sequences.
    Useful for reading-order text extraction.
    """
    def __init__(self, hidden_size: int, vocab_size: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class QAHead(nn.Module):
    """
    Question-answering head.
    Standard LM head for natural language answers.
    """
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class MultiTaskOutputHeads(nn.Module):
    """
    Container for all structured output heads.
    Routes decoder hidden states to the appropriate task-specific head(s).
    During training, all heads can be trained jointly with task-specific losses.
    During inference, only the requested head is used.
    """
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_json_tokens: int = 256,
        num_keys: int = 128,
        num_table_tokens: int = 128,
        ocr_vocab_size: int = 256,
    ):
        super().__init__()
        self.json_head = JSONHead(hidden_size, num_json_tokens)
        self.kv_head = KVHead(hidden_size, num_keys)
        self.table_head = TableHead(hidden_size, num_table_tokens)
        self.ocr_head = OCRHead(hidden_size, ocr_vocab_size)
        self.qa_head = QAHead(hidden_size, vocab_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        task: str = "qa",
    ) -> Dict[str, torch.Tensor]:
        if task == "json":
            return {"logits": self.json_head(hidden_states)}
        elif task == "kv":
            return self.kv_head(hidden_states)
        elif task == "table":
            return {"logits": self.table_head(hidden_states)}
        elif task == "ocr":
            return {"logits": self.ocr_head(hidden_states)}
        elif task == "qa":
            return {"logits": self.qa_head(hidden_states)}
        else:
            raise ValueError(f"Unknown task: {task}")
