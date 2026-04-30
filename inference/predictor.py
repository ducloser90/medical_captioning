"""
inference/predictor.py — Generate captions from a trained BLIP checkpoint.

Supports:
  * Batched generation over a test DataFrame (writes predictions_only.csv
    AND ground_truth_only.csv so the evaluator can run immediately).
  * Single-image inference from a PIL Image or a file path.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
from tqdm import tqdm

from config import CFG


class Predictor:
    """
    Wraps a trained BLIP checkpoint and exposes caption generation methods.

    Parameters
    ----------
    checkpoint_dir : Path to a saved ``BlipForConditionalGeneration`` checkpoint.
                     Defaults to ``CFG.best_ckpt_dir``.
    device         : torch.device.  Auto-detected (CUDA → CPU) by default.
    batch_size     : Images processed per forward pass.  Defaults to
                     ``CFG.train.batch_size``.
    """

    def __init__(
        self,
        checkpoint_dir: str | None = None,
        device: Optional[torch.device] = None,
        batch_size: int | None = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir or CFG.best_ckpt_dir
        self.device         = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size     = batch_size or CFG.train.batch_size

        print(f"Loading checkpoint from {self.checkpoint_dir} …")
        self.processor = BlipProcessor.from_pretrained(self.checkpoint_dir)
        self.model     = BlipForConditionalGeneration.from_pretrained(self.checkpoint_dir)
        self.model.eval()
        self.model.to(self.device)
        print(f"Model ready on {self.device}.")

    # ------------------------------------------------------------------
    # Public: batch prediction → both submission CSVs
    # ------------------------------------------------------------------

    def predict_dataset(
        self,
        df: pd.DataFrame,
        img_dir: str,
        predictions_csv: str,
        ground_truth_csv: str,
        max_new_tokens: int = 64,
        num_beams: int = 4,
    ) -> tuple[str, str]:
        """
        Generate captions for every image in ``df`` and write two CSVs:

        * ``predictions_csv``  — model-generated captions  (predictions_only.csv)
        * ``ground_truth_csv`` — reference captions from the test split
                                  (ground_truth_only.csv)

        Both CSVs have exactly two columns: ``ID, Caption``.

        Parameters
        ----------
        df               : Test-split DataFrame with ID and Caption columns.
        img_dir          : Directory containing ``<id>.jpg`` images.
        predictions_csv  : Output path for model predictions.
        ground_truth_csv : Output path for ground-truth captions.
        max_new_tokens   : Maximum tokens to generate per caption.
        num_beams        : Beam-search width.

        Returns
        -------
        (predictions_csv, ground_truth_csv) — paths to the two written files.
        """
        img_ids   = df[CFG.data.img_col].astype(str).tolist()
        gt_caps   = df[CFG.data.caption_col].astype(str).tolist()

        pred_caps = self._generate_captions(
            img_ids        = img_ids,
            img_dir        = img_dir,
            max_new_tokens = max_new_tokens,
            num_beams      = num_beams,
        )

        # ── Write predictions_only.csv ────────────────────────────────────────
        _write_id_caption_csv(
            path    = predictions_csv,
            ids     = img_ids,
            caps    = pred_caps,
            label   = "predictions",
        )

        # ── Write ground_truth_only.csv ───────────────────────────────────────
        _write_id_caption_csv(
            path    = ground_truth_csv,
            ids     = img_ids,
            caps    = gt_caps,
            label   = "ground truth",
        )

        return predictions_csv, ground_truth_csv

    # ------------------------------------------------------------------
    # Public: single-image inference
    # ------------------------------------------------------------------

    def predict_image(
        self,
        image: Image.Image | str | Path,
        max_new_tokens: int = 64,
        num_beams: int = 4,
    ) -> str:
        """
        Generate a caption for a single image.

        Parameters
        ----------
        image          : A PIL Image or a path to a JPEG file.
        max_new_tokens : Maximum tokens to generate.
        num_beams      : Beam-search width.

        Returns
        -------
        str — The generated caption.
        """
        if isinstance(image, (str, Path)):
            image = _load_image(str(image))

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                num_beams      = num_beams,
            )

        return self.processor.decode(ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Private — batched generation
    # ------------------------------------------------------------------

    def _generate_captions(
        self,
        img_ids: List[str],
        img_dir: str,
        max_new_tokens: int,
        num_beams: int,
    ) -> List[str]:
        """
        Run batched beam-search generation over all images.

        Returns a list of captions in the same order as ``img_ids``.
        """
        all_captions: List[str] = []

        for batch_start in tqdm(
            range(0, len(img_ids), self.batch_size),
            desc="Generating captions",
        ):
            batch_ids   = img_ids[batch_start : batch_start + self.batch_size]
            batch_imgs  = [
                _load_image(os.path.join(img_dir, f"{img_id}.jpg"))
                for img_id in batch_ids
            ]

            # Processor handles batching + padding natively
            inputs = self.processor(
                images         = batch_imgs,
                return_tensors = "pt",
                padding        = True,
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens = max_new_tokens,
                    num_beams      = num_beams,
                )

            captions = self.processor.batch_decode(
                output_ids, skip_special_tokens=True
            )
            all_captions.extend(captions)

        return all_captions


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _load_image(path: str) -> Image.Image:
    """Open image, falling back to a grey placeholder on failure."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (384, 384), color=128)


def _write_id_caption_csv(
    path: str,
    ids: List[str],
    caps: List[str],
    label: str,
) -> None:
    """Write an ``ID, Caption`` CSV to ``path``, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Caption"])
        writer.writerows(zip(ids, caps))
    print(f"  ✓ {label} CSV saved to {path} ({len(ids)} rows).")
