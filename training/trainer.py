"""
training/trainer.py — Self-contained ``Trainer`` that handles the full
training lifecycle: optimiser, cosine LR schedule, mixed-precision
gradient accumulation, checkpointing, and optional HF Hub upload.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm

from config import CFG
from utils.hub import push_to_hub
from utils.vram import vram_summary


class Trainer:
    """
    Encapsulates the BLIP fine-tuning loop.

    Usage
    -----
    >>> trainer = Trainer(model, processor, train_loader, val_loader, device)
    >>> history = trainer.fit()
    """

    def __init__(
        self,
        model: BlipForConditionalGeneration,
        processor: BlipProcessor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model        = model
        self.processor    = processor
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device

        # fp16 only makes sense on CUDA; silently disable it on CPU
        self._use_amp = CFG.train.fp16 and device.type == "cuda"

        self.optimizer, self.scheduler = self._build_optimizer_and_scheduler()

        # GradScaler must match the actual device type; disabled on CPU
        self.scaler = torch.amp.GradScaler(device.type, enabled=self._use_amp)

        self.best_val_loss: float = float("inf")
        self.history: List[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> List[dict]:
        """
        Run the full training loop for ``CFG.train.epochs`` epochs.

        Returns
        -------
        history : list of dicts with keys ``epoch``, ``train_loss``, ``val_loss``.
        """
        _vram_probe(self.model, self.train_loader, self.device, self._use_amp)

        for epoch in range(1, CFG.train.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss   = self._val_epoch()

            record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            self.history.append(record)

            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss {train_loss:.4f} | "
                f"Val Loss {val_loss:.4f}"
            )
            print(vram_summary())

            self._maybe_save_and_push(epoch, val_loss)

        self._save_history()
        print(f"\nTraining complete. Best Val Loss: {self.best_val_loss:.4f}")
        return self.history

    # ------------------------------------------------------------------
    # Private — training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}/{CFG.train.epochs} [train]",
            leave=False,
        )

        for step, batch in pbar:
            loss = self._forward_step(batch)
            self.scaler.scale(loss).backward()
            total_loss += loss.item() * CFG.train.accum_steps  # undo the /accum division

            if (step + 1) % CFG.train.accum_steps == 0:
                self._optimizer_step()

            # VRAM display is CUDA-only
            vram_str = (
                f"{torch.cuda.memory_allocated(0) / 1e9:.1f}GB"
                if self.device.type == "cuda"
                else "cpu"
            )
            pbar.set_postfix(
                loss=f"{total_loss / (step + 1):.4f}",
                lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                vram=vram_str,
            )

        # Flush any remaining accumulated gradients at end of epoch
        remaining = (step + 1) % CFG.train.accum_steps
        if remaining != 0:
            self._optimizer_step()

        return total_loss / len(self.train_loader)

    # ------------------------------------------------------------------
    # Private — validation epoch
    # ------------------------------------------------------------------

    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                with torch.amp.autocast(self.device.type, enabled=self._use_amp):
                    outputs = self.model(
                        pixel_values   = batch["pixel_values"].to(self.device),
                        input_ids      = batch["input_ids"].to(self.device),
                        attention_mask = batch["attention_mask"].to(self.device),
                        labels         = batch["labels"].to(self.device),
                    )
                total_loss += outputs.loss.item()

        return total_loss / len(self.val_loader)

    # ------------------------------------------------------------------
    # Private — single forward + scaled loss
    # ------------------------------------------------------------------

    def _forward_step(self, batch: dict) -> torch.Tensor:
        with torch.amp.autocast(self.device.type, enabled=self._use_amp):
            outputs = self.model(
                pixel_values   = batch["pixel_values"].to(self.device),
                input_ids      = batch["input_ids"].to(self.device),
                attention_mask = batch["attention_mask"].to(self.device),
                labels         = batch["labels"].to(self.device),
            )
            return outputs.loss / CFG.train.accum_steps

    # ------------------------------------------------------------------
    # Private — gradient update
    # ------------------------------------------------------------------

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad()

    # ------------------------------------------------------------------
    # Private — checkpointing + Hub push
    # ------------------------------------------------------------------

    def _maybe_save_and_push(self, epoch: int, val_loss: float) -> None:
        if val_loss >= self.best_val_loss:
            return

        self.best_val_loss = val_loss
        ckpt_dir = CFG.best_ckpt_dir

        self.model.save_pretrained(ckpt_dir)
        self.processor.save_pretrained(ckpt_dir)
        print(f"  ✓ New best Val Loss: {val_loss:.4f} — saved to {ckpt_dir}")

        if CFG.hub.push_every_epoch:
            push_to_hub(
                folder_path    = ckpt_dir,
                commit_message = f"Best checkpoint — epoch {epoch}, val_loss={val_loss:.4f}",
            )

    # ------------------------------------------------------------------
    # Private — history persistence
    # ------------------------------------------------------------------

    def _save_history(self) -> None:
        path = Path(CFG.history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {path}")

    # ------------------------------------------------------------------
    # Private — optimiser + scheduler construction
    # ------------------------------------------------------------------

    def _build_optimizer_and_scheduler(self):
        """Build AdamW with differential LRs for vision vs decoder params."""
        vision_params = [
            p for p in self.model.vision_model.parameters() if p.requires_grad
        ]
        decoder_params = [
            p for p in self.model.text_decoder.parameters() if p.requires_grad
        ]

        param_groups = [
            {"params": vision_params,  "lr": CFG.train.lr / 10, "name": "vision"},
            {"params": decoder_params, "lr": CFG.train.lr,       "name": "decoder"},
        ]

        optimizer = AdamW(param_groups, weight_decay=CFG.train.weight_decay)

        total_steps  = (len(self.train_loader) // CFG.train.accum_steps) * CFG.train.epochs
        warmup_steps = int(total_steps * CFG.train.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = warmup_steps,
            num_training_steps = total_steps,
        )

        print(
            f"Vision LR: {CFG.train.lr / 10:.1e}  |  "
            f"Decoder LR: {CFG.train.lr:.1e}  |  "
            f"Total steps: {total_steps}  |  Warmup: {warmup_steps}"
        )

        return optimizer, scheduler


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _vram_probe(
    model: BlipForConditionalGeneration,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> None:
    """
    Run a single forward pass to confirm the batch fits in VRAM, then
    release the memory.  Skipped entirely on CPU-only machines.
    """
    if device.type != "cuda":
        print("No CUDA device — skipping VRAM probe.")
        return

    print("VRAM before forward pass:")
    print(vram_summary())

    sample = next(iter(loader))
    with torch.no_grad(), torch.amp.autocast(device.type, enabled=use_amp):
        model(
            pixel_values   = sample["pixel_values"].to(device),
            input_ids      = sample["input_ids"].to(device),
            attention_mask = sample["attention_mask"].to(device),
            labels         = sample["labels"].to(device),
        )

    print(f"VRAM after forward pass (batch_size={CFG.train.batch_size}):")
    print(vram_summary())

    del sample
    torch.cuda.empty_cache()
