#!/usr/bin/env python3
"""
scripts/train.py — End-to-end training entry point.

Usage
-----
    python scripts/train.py

Override configuration by editing ``config.py`` or by passing environment
variables before the call, e.g.::

    EPOCHS=8 python scripts/train.py   # (requires extending Config.__post_init__)
"""

from __future__ import annotations

import random
import sys
import os

# Make the project root importable when the script is invoked directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from config import CFG
from data.download import download_and_extract
from data.dataset import build_loaders
from models.blip import build_model
from training.trainer import Trainer
from utils.hub import init_hub
from utils.vram import vram_summary


def set_seed(seed: int) -> None:
    """Fix all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    # ── Reproducibility ──────────────────────────────────────────────────────
    set_seed(CFG.train.seed)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # ── Hardware ─────────────────────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  {props.total_memory / 1e9:.1f} GB")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Data ─────────────────────────────────────────────────────────────────
    download_and_extract()

    # ── Model ────────────────────────────────────────────────────────────────
    model, processor = build_model(device)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_loaders(processor)
    print(f"Effective batch size: {CFG.train.effective_batch_size}")

    # ── Hub initialisation ───────────────────────────────────────────────────
    if CFG.hub.push_every_epoch:
        init_hub()

    # ── Training ─────────────────────────────────────────────────────────────
    trainer = Trainer(model, processor, train_loader, val_loader, device)
    history = trainer.fit()

    print("\nFinal training history:")
    for record in history:
        print(f"  Epoch {record['epoch']:02d} — "
              f"train_loss={record['train_loss']:.4f}, "
              f"val_loss={record['val_loss']:.4f}")


if __name__ == "__main__":
    main()
