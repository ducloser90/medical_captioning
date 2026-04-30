#!/usr/bin/env python3
"""
scripts/evaluate.py — Score a submission file against the ground truth.

Usage
-----
    # Use paths from config.py
    python scripts/evaluate.py

    # Override paths via CLI flags
    python scripts/evaluate.py \\
        --gt   /path/to/ground_truth.csv \\
        --pred /path/to/predictions.csv
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CFG
from evaluation.evaluator import CaptionEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate medical caption predictions.")
    parser.add_argument(
        "--gt",
        default=CFG.eval.ground_truth_path,
        help="Path to ground-truth CSV.",
    )
    parser.add_argument(
        "--pred",
        default=CFG.eval.submission_file_path,
        help="Path to predictions CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluator = CaptionEvaluator(ground_truth_path=args.gt)
    results   = evaluator.evaluate(submission_file_path=args.pred)

    print("\n── Evaluation Results ──────────────────────────────")
    print(f"  BERTScore (primary)   : {results['score']:.4f}")
    print(f"  ROUGE-1   (secondary) : {results['score_secondary']:.4f}")


if __name__ == "__main__":
    main()
