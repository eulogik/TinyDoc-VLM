import os
import math
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Union, Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler

from .modeling import TinyDocVLMForConditionalGeneration
from .processing import TinyDocVLMProcessor
from .losses import CombinedLoss
from .data import DocumentDataset, collate_fn

logger = logging.getLogger(__name__)


class TrainerConfig:
    def __init__(
        self,
        output_dir: str = "checkpoints",
        num_epochs: int = 3,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1e-4,
        min_learning_rate: float = 1e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        max_seq_length: int = 2048,
        stage: int = 1,
        use_fp16: bool = True,
        save_every_steps: int = 1000,
        eval_every_steps: int = 500,
        log_every_steps: int = 10,
        gradient_checkpointing: bool = True,
    ):
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.max_seq_length = max_seq_length
        self.stage = stage
        self.use_fp16 = use_fp16
        self.save_every_steps = save_every_steps
        self.eval_every_steps = eval_every_steps
        self.log_every_steps = log_every_steps
        self.gradient_checkpointing = gradient_checkpointing

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})


class TinyDocVLMTrainer:
    """
    Trainer for TinyDoc-VLM across all 3 training stages.
    Supports FSDP, mixed precision, gradient checkpointing, and checkpointing.
    """
    def __init__(
        self,
        model: TinyDocVLMForConditionalGeneration,
        processor: TinyDocVLMProcessor,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or TrainerConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        os.makedirs(self.config.output_dir, exist_ok=True)

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(self.device)

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_params = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_params, lr=self.config.learning_rate, betas=(0.9, 0.95), eps=1e-8)

        total_steps = len(self.train_dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps) * self.config.num_epochs
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.05, end_factor=1.0, total_iters=self.config.warmup_steps)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps - self.config.warmup_steps), eta_min=self.config.min_learning_rate)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.config.warmup_steps])

        self.scaler = GradScaler(enabled=self.config.use_fp16)
        self.loss_fn = CombinedLoss(stage=self.config.stage)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.processor.tokenizer, self.processor.image_token_id, self.config.max_seq_length),
            num_workers=4,
            pin_memory=True,
        )

        if self.eval_dataset:
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=lambda batch: collate_fn(batch, self.processor.tokenizer, self.processor.image_token_id, self.config.max_seq_length),
                num_workers=4,
                pin_memory=True,
            )

    def train_step(self, batch: Dict) -> Dict:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)
        labels = batch["labels"].to(self.device)
        task = batch.get("task", None)

        with autocast(enabled=self.config.use_fp16):
            if task and self.config.stage == 2:
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=labels,
                    task=task,
                )
                lm_loss = outputs["lm_outputs"].loss
                head_outputs = outputs["head_outputs"]
                loss_dict = self.loss_fn(
                    lm_logits=outputs["lm_outputs"].logits,
                    lm_labels=labels,
                    head_outputs=head_outputs,
                    head_labels=batch.get("head_labels", None),
                )
                loss = loss_dict["loss"]
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        if self.config.use_fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {"loss": loss.item()}

    def train_epoch(self) -> Dict:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulation_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(self.train_loader):
            step_loss = self.train_step(batch)
            accumulation_loss += step_loss["loss"]

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_fp16:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                if self.config.use_fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

                avg_loss = accumulation_loss / self.config.gradient_accumulation_steps
                total_loss += avg_loss
                num_batches += 1
                accumulation_loss = 0.0

                if self.global_step % self.config.log_every_steps == 0:
                    elapsed = time.time() - start_time
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {self.epoch+1} | Step {self.global_step} | Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | Elapsed: {elapsed:.1f}s"
                    )

                if self.global_step % self.config.eval_every_steps == 0 and self.eval_loader:
                    eval_loss = self.evaluate()
                    logger.info(f"Eval loss: {eval_loss:.4f}")
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_checkpoint("best")

                if self.global_step % self.config.save_every_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

        avg_epoch_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        logger.info(f"Epoch {self.epoch+1} complete. Avg loss: {avg_epoch_loss:.4f}. Elapsed: {elapsed:.1f}s")

        return {"loss": avg_epoch_loss, "epoch": self.epoch + 1, "steps": self.global_step}

    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast(enabled=self.config.use_fp16):
                    outputs = self.model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        return total_loss / max(num_batches, 1)

    def train(self):
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Eval samples: {len(self.eval_dataset) if self.eval_dataset else 0}")
        logger.info(f"Config: {self.config.to_dict()}")

        self.save_checkpoint("init")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            metrics = self.train_epoch()
            self.save_checkpoint(f"epoch_{epoch+1}")

        logger.info("Training complete.")

    def save_checkpoint(self, tag: str):
        output_dir = Path(self.config.output_dir) / tag
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(output_dir))
        self.processor.tokenizer.save_pretrained(str(output_dir))

        trainer_state = {
            "global_step": self.global_step,
            "epoch": self.epoch + 1,
            "best_eval_loss": self.best_eval_loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
        }
        torch.save(trainer_state, output_dir / "trainer_state.pt")
        logger.info(f"Checkpoint saved to {output_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        checkpoint_dir = Path(checkpoint_dir)
        self.model = TinyDocVLMForConditionalGeneration.from_pretrained(str(checkpoint_dir))
        self.model.to(self.device)

        trainer_state = torch.load(checkpoint_dir / "trainer_state.pt", map_location=self.device)
        self.global_step = trainer_state["global_step"]
        self.epoch = trainer_state.get("epoch", 1) - 1
        self.best_eval_loss = trainer_state.get("best_eval_loss", float("inf"))
        self.optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
        self.scheduler.load_state_dict(trainer_state["scheduler_state_dict"])
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")
