"""
tests/test_core.py — Lightweight unit tests that do NOT require a GPU
or the real dataset. All heavy dependencies are mocked.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Make the project root importable when tests are discovered via pytest / unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):

    def test_effective_batch_size(self) -> None:
        from config import Config
        cfg = Config()
        self.assertEqual(
            cfg.train.effective_batch_size,
            cfg.train.batch_size * cfg.train.accum_steps,
        )

    def test_paths_are_strings(self) -> None:
        from config import CFG
        self.assertIsInstance(CFG.data.train_csv, str)
        self.assertIsInstance(CFG.best_ckpt_dir, str)
        self.assertIsInstance(CFG.history_path, str)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestMedCapDataset(unittest.TestCase):

    def _make_dataset(self):
        import pandas as pd
        from PIL import Image
        from data.dataset import MedCapDataset

        mock_processor = MagicMock()
        mock_enc = {
            "input_ids":      MagicMock(squeeze=lambda _: MagicMock(clone=lambda: MagicMock())),
            "pixel_values":   MagicMock(squeeze=lambda _: MagicMock()),
            "attention_mask": MagicMock(squeeze=lambda _: MagicMock()),
        }
        # Make tokenizer.pad_token_id accessible
        mock_processor.tokenizer.pad_token_id = 0
        mock_processor.return_value = mock_enc

        df = pd.DataFrame({"ID": ["img1", "img2"], "Caption": ["cap1", "cap2"]})
        return MedCapDataset(df, "/nonexistent", mock_processor, max_length=32), df

    def test_len(self) -> None:
        ds, df = self._make_dataset()
        self.assertEqual(len(ds), len(df))

    def test_fallback_image(self) -> None:
        from PIL import Image
        from data.dataset import MedCapDataset
        img = MedCapDataset._load_image("/definitely/does/not/exist.jpg")
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (384, 384))


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

class TestCaptionEvaluator(unittest.TestCase):

    def _write_csv(self, rows: list[tuple], path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def test_load_csv_no_header(self) -> None:
        from evaluation.evaluator import CaptionEvaluator

        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = os.path.join(tmpdir, "gt.csv")
            self._write_csv([("img1", "a caption"), ("img2", "another")], gt_path)

            # Patch heavy scorers
            with patch("evaluation.evaluator.evaluate.load", return_value=MagicMock()):
                ev = CaptionEvaluator(ground_truth_path=gt_path)

            self.assertIn("img1", ev.gt)
            self.assertIn("img2", ev.gt)

    def test_clean_lowercases_and_strips_punct(self) -> None:
        from evaluation.evaluator import CaptionEvaluator

        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = os.path.join(tmpdir, "gt.csv")
            self._write_csv([("x", "dummy")], gt_path)

            with patch("evaluation.evaluator.evaluate.load", return_value=MagicMock()):
                ev = CaptionEvaluator(ground_truth_path=gt_path)

            result = ev.clean("Hello, World! 42")
            self.assertEqual(result, "hello world number")

    def test_load_predictions_duplicate_raises(self) -> None:
        from evaluation.evaluator import CaptionEvaluator

        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path   = os.path.join(tmpdir, "gt.csv")
            pred_path = os.path.join(tmpdir, "pred.csv")

            self._write_csv([("img1", "a"), ("img2", "b")], gt_path)
            self._write_csv([("img1", "x"), ("img1", "y")], pred_path)  # duplicate

            with patch("evaluation.evaluator.evaluate.load", return_value=MagicMock()):
                ev = CaptionEvaluator(ground_truth_path=gt_path)

            with self.assertRaises(ValueError):
                ev._load_predictions(pred_path)

    def test_load_predictions_unknown_id_raises(self) -> None:
        from evaluation.evaluator import CaptionEvaluator

        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path   = os.path.join(tmpdir, "gt.csv")
            pred_path = os.path.join(tmpdir, "pred.csv")

            self._write_csv([("img1", "a")], gt_path)
            self._write_csv([("UNKNOWN", "x")], pred_path)

            with patch("evaluation.evaluator.evaluate.load", return_value=MagicMock()):
                ev = CaptionEvaluator(ground_truth_path=gt_path)

            with self.assertRaises(ValueError):
                ev._load_predictions(pred_path)


# ---------------------------------------------------------------------------
# VRAM utility tests
# ---------------------------------------------------------------------------

class TestVramSummary(unittest.TestCase):

    def test_no_gpu_returns_string(self) -> None:
        from utils.vram import vram_summary
        result = vram_summary()
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
