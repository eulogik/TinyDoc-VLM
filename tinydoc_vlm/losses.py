import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def ce_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index)


def kv_loss(kv_outputs: Dict[str, torch.Tensor], key_labels: torch.Tensor, confidence_labels: torch.Tensor) -> torch.Tensor:
    key_ce = F.cross_entropy(kv_outputs["key_logits"].view(-1, kv_outputs["key_logits"].size(-1)), key_labels.view(-1))
    conf_bce = F.binary_cross_entropy(kv_outputs["confidence"].view(-1), confidence_labels.view(-1))
    return key_ce + conf_bce


class CombinedLoss(nn.Module):
    """
    Combines multiple task-specific losses for multi-stage training.
    Stage 1: layout + OCR + region
    Stage 2: QA + JSON + KV + table
    Stage 3: standard LM
    """
    def __init__(self, stage: int = 1):
        super().__init__()
        self.stage = stage

    def forward(
        self,
        lm_logits: torch.Tensor,
        lm_labels: torch.Tensor,
        head_outputs: Optional[Dict[str, torch.Tensor]] = None,
        head_labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        total_loss = torch.tensor(0.0, device=lm_logits.device)

        lm_loss = ce_loss(lm_logits, lm_labels)
        losses["lm_loss"] = lm_loss

        if self.stage == 1:
            total_loss = lm_loss
        elif self.stage == 2:
            total_loss = lm_loss
            if head_outputs and head_labels:
                if "json_logits" in head_outputs and "json_labels" in head_labels:
                    json_loss = ce_loss(head_outputs["json_logits"], head_labels["json_labels"])
                    losses["json_loss"] = json_loss
                    total_loss = total_loss + json_loss

                if "kv" in head_outputs and "kv_key_labels" in head_labels:
                    kv = kv_loss(head_outputs["kv"], head_labels["kv_key_labels"], head_labels["kv_conf_labels"])
                    losses["kv_loss"] = kv
                    total_loss = total_loss + kv

                if "table_logits" in head_outputs and "table_labels" in head_labels:
                    table_loss = ce_loss(head_outputs["table_logits"], head_labels["table_labels"])
                    losses["table_loss"] = table_loss
                    total_loss = total_loss + table_loss
        else:
            total_loss = lm_loss

        losses["loss"] = total_loss
        return losses
