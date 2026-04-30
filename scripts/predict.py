#!/usr/bin/env python3
"""
scripts/predict.py — Run inference on the test split and write both
competition CSVs that the evaluator expects.

Output files
------------
  predictions_only.csv  — model-generated captions   (CFG.eval.submission_file_path)
  ground_truth_only.csv — reference test captions     (CFG.eval.ground_truth_path)

Both are written to the same directory so ``scripts/evaluate.py`` can be
run immediately afterwards without any path changes.

Usage
-----
    python scripts/predict.py

    # Override checkpoint or output directory
    python scripts/predict.py \\
        --ckpt /path/to/checkpoint \\
        --out-dir /path/to/output/dir
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from config import CFG
from inference.predictor import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate test-set captions and write evaluation CSVs."
    )
    parser.add_argument(
        "--ckpt",
        default=CFG.best_ckpt_dir,
        help="Path to a trained BLIP checkpoint directory.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.dirname(CFG.eval.submission_file_path),
        help=(
            "Directory where predictions_only.csv and ground_truth_only.csv "
            "will be written. Defaults to the directory of "
            "CFG.eval.submission_file_path."
        ),
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64,
        help="Maximum tokens to generate per caption.",
    )
    parser.add_argument(
        "--num-beams", type=int, default=4,
        help="Beam width for generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Resolve output paths (always use the canonical eval filenames) ────────
    out_dir          = os.path.abspath(args.out_dir)
    predictions_csv  = os.path.join(out_dir, "predictions_only.csv")
    ground_truth_csv = os.path.join(out_dir, "ground_truth_only.csv")

    os.makedirs(out_dir, exist_ok=True)

    # ── Update CFG so the evaluator points at the same files ─────────────────
    CFG.eval.submission_file_path = predictions_csv
    CFG.eval.ground_truth_path    = ground_truth_csv

    # ── Load test split ───────────────────────────────────────────────────────
    test_df = pd.read_csv(CFG.data.test_csv)
    print(f"Test split: {len(test_df)} images")
    print(f"Writing CSVs to: {out_dir}")

    # ── Run inference ─────────────────────────────────────────────────────────
    predictor = Predictor(checkpoint_dir=args.ckpt)

    pred_path, gt_path = predictor.predict_dataset(
        df               = test_df,
        img_dir          = CFG.data.test_img_dir,
        predictions_csv  = predictions_csv,
        ground_truth_csv = ground_truth_csv,
        max_new_tokens   = args.max_tokens,
        num_beams        = args.num_beams,
    )

    print("\nDone. To evaluate, run:")
    print(f"  python scripts/evaluate.py --gt {gt_path} --pred {pred_path}")


if __name__ == "__main__":
    main()
