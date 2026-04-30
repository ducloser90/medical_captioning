"""
data/dataset.py — PyTorch Dataset for the Rocov2 medical captioning task,
plus a convenience factory that returns train / val / test DataLoaders.
"""

from __future__ import annotations

import os
from typing import Dict

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import BlipProcessor

from config import CFG


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MedCapDataset(Dataset):
    """
    Loads (image, caption) pairs from a CSV and returns tokenised tensors
    compatible with ``BlipForConditionalGeneration``.

    Parameters
    ----------
    df:          DataFrame with at minimum an image-ID column and a caption column.
    img_dir:     Directory that contains ``<id>.jpg`` files.
    processor:   Instantiated ``BlipProcessor``.
    max_length:  Maximum token length for caption truncation / padding.
    img_col:     Column name for image IDs.    Defaults to ``CFG.data.img_col``.
    caption_col: Column name for captions.    Defaults to ``CFG.data.caption_col``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        processor: BlipProcessor,
        max_length: int,
        img_col: str | None = None,
        caption_col: str | None = None,
    ) -> None:
        self.df          = df.reset_index(drop=True)
        self.img_dir     = img_dir
        self.processor   = processor
        self.max_length  = max_length
        self.img_col     = img_col     or CFG.data.img_col
        self.caption_col = caption_col or CFG.data.caption_col

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row[self.img_col]}.jpg")
        caption  = str(row[self.caption_col])

        image = self._load_image(img_path)

        enc = self.processor(
            images         = image,
            text           = caption,
            padding        = "max_length",
            truncation     = True,
            max_length     = self.max_length,
            return_tensors = "pt",
        )

        input_ids    = enc["input_ids"].squeeze(0)
        pixel_values = enc["pixel_values"].squeeze(0)
        attn_mask    = enc["attention_mask"].squeeze(0)

        # Replace pad tokens with -100 so they are ignored by the loss
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values"  : pixel_values,
            "input_ids"     : input_ids,
            "attention_mask": attn_mask,
            "labels"        : labels,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _load_image(path: str) -> Image.Image:
        """Open image, falling back to a grey placeholder on failure."""
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", (384, 384), color=128)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_loaders(
    processor: BlipProcessor,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Read the three CSV splits and return (train_loader, val_loader, test_loader).

    Uses settings from the global ``CFG`` object.
    """
    train_df = pd.read_csv(CFG.data.train_csv)
    val_df   = pd.read_csv(CFG.data.val_csv)
    test_df  = pd.read_csv(CFG.data.test_csv)

    print(
        f"Dataset sizes — Train: {len(train_df)} | "
        f"Val: {len(val_df)} | Test: {len(test_df)}"
    )
    print(f"Columns: {train_df.columns.tolist()}")

    _report_sample(train_df, CFG.data.train_img_dir)

    train_ds = MedCapDataset(train_df, CFG.data.train_img_dir, processor, CFG.model.max_length)
    val_ds   = MedCapDataset(val_df,   CFG.data.val_img_dir,   processor, CFG.model.max_length)
    test_ds  = MedCapDataset(test_df,  CFG.data.test_img_dir,  processor, CFG.model.max_length)

    # pin_memory requires CUDA; silently disable on CPU-only machines
    _pin = torch.cuda.is_available()

    loader_kwargs = dict(
        batch_size  = CFG.train.batch_size,
        num_workers = CFG.train.num_workers,
        pin_memory  = _pin,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def _report_sample(df: pd.DataFrame, img_dir: str) -> None:
    """Log a single sample for a quick sanity-check."""
    sample_id  = df[CFG.data.img_col].iloc[0]
    sample_cap = str(df[CFG.data.caption_col].iloc[0])[:120]
    sample_img = MedCapDataset._load_image(os.path.join(img_dir, f"{sample_id}.jpg"))
    print(f"Sample image: {sample_img.size}, mode={sample_img.mode}")
    print(f"Caption: {sample_cap}")
