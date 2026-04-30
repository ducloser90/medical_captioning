"""
evaluation/evaluator.py — ImageCLEFmedical Caption Prediction evaluator.

Primary metric   : BERTScore  (microsoft/deberta-xlarge-mnli)
Secondary metric : ROUGE-1

The evaluator is a direct refactor of the original competition script.
All logic is preserved; only the structure has been cleaned up.
"""

from __future__ import annotations

import csv
import re
import string
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import evaluate
import numpy as np
import transformers
from bert_score import score as bert_score_fn
import bert_score.utils as _bsu
from packaging import version
from tqdm import tqdm

from config import CFG


# ---------------------------------------------------------------------------
# Monkey-patch bert_score to enforce truncation
# (prevents OverflowError in the fast-tokenizer Rust backend)
# ---------------------------------------------------------------------------

def _safe_sent_encode(tokenizer, sent: str) -> List[int]:
    if version.parse(transformers.__version__) >= version.parse("4.0.0"):
        return tokenizer.encode(
            sent,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
        )
    return tokenizer.encode(sent, add_special_tokens=True)


_bsu.sent_encode = _safe_sent_encode


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class CaptionEvaluator:
    """
    Evaluate predicted captions against a ground-truth CSV.

    Parameters
    ----------
    ground_truth_path : Path to the ground-truth CSV (``ID,Caption`` rows).
    device            : ``"cuda"`` or ``"cpu"``.  Auto-detected by default.
    bert_batch_size   : Batch size for the BERTScore pass.
    """

    def __init__(
        self,
        ground_truth_path: str | None = None,
        device: Optional[str] = None,
        bert_batch_size: int | None = None,
    ) -> None:
        import torch

        self.ground_truth_path = ground_truth_path or CFG.eval.ground_truth_path
        self.device            = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self.bert_batch_size   = bert_batch_size or CFG.eval.bert_batch_size
        self.max_len           = CFG.eval.max_text_len
        self.case_sensitive    = CFG.eval.case_sensitive

        self.gt = self._load_csv(self.ground_truth_path)
        print(f"Ground truth loaded — {len(self.gt)} samples.")

        print("Loading ROUGE scorer …")
        self._rouge = evaluate.load("rouge")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, submission_file_path: str) -> Dict[str, float]:
        """
        Score a submission file.

        Parameters
        ----------
        submission_file_path : CSV with ``ID,Caption`` rows (no header required).

        Returns
        -------
        dict with keys ``score`` (BERTScore) and ``score_secondary`` (ROUGE-1).
        """
        predictions = self._load_predictions(submission_file_path)

        bert  = self._compute_bertscore(predictions)
        rouge = self._compute_rouge(predictions)

        print(f"BERTScore (primary)  : {bert:.4f}")
        print(f"ROUGE-1   (secondary): {rouge:.4f}")

        return {"score": bert, "score_secondary": rouge}

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """Normalise text to match competition pre-processing."""
        if text is None:
            return ""
        text = str(text)[: self.max_len]
        if not self.case_sensitive:
            text = text.lower()
        text = re.sub(r"\d+", "number", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.strip()

    # ------------------------------------------------------------------
    # BERTScore
    # ------------------------------------------------------------------

    def _compute_bertscore(self, predictions: Dict[str, str]) -> float:
        warnings.filterwarnings("ignore")

        keys       = list(predictions.keys())
        candidates = [self.clean(predictions[k]) for k in keys]
        references = [self.clean(self.gt[k])      for k in keys]

        scores: List[Optional[float]] = [None] * len(keys)
        real_cands, real_refs, real_idx = [], [], []

        # Trivial case: both empty → perfect score
        for i, (c, r) in enumerate(zip(candidates, references)):
            if len(c) == 0 and len(r) == 0:
                scores[i] = 1.0
            else:
                real_cands.append(c)
                real_refs.append(r)
                real_idx.append(i)

        if real_cands:
            scores = self._run_bertscore(scores, real_cands, real_refs, real_idx)

        return float(np.mean(scores))

    def _run_bertscore(
        self,
        scores: List,
        candidates: List[str],
        references: List[str],
        indices: List[int],
    ) -> List:
        """Attempt full-batch BERTScore; fall back to per-item on failure."""
        try:
            _, _, F = bert_score_fn(
                cands      = candidates,
                refs       = references,
                model_type = CFG.eval.bert_model,
                batch_size = self.bert_batch_size,
                device     = self.device,
                verbose    = True,
            )
            for i, f in zip(indices, F.tolist()):
                scores[i] = float(f)

        except Exception as exc:
            print(f"  [WARN] Full-batch BERTScore failed ({exc}), trying per-item …")
            for i, c, r in tqdm(
                zip(indices, candidates, references),
                total=len(indices),
                desc="BERTScore per-item fallback",
            ):
                try:
                    _, _, F = bert_score_fn(
                        cands=[c], refs=[r],
                        model_type=CFG.eval.bert_model,
                        batch_size=1,
                        device=self.device,
                        verbose=False,
                    )
                    scores[i] = float(F[0])
                except Exception as exc2:
                    print(f"    [SKIP] item {i} failed ({exc2}), assigning 0.0")
                    scores[i] = 0.0

        return scores

    # ------------------------------------------------------------------
    # ROUGE-1
    # ------------------------------------------------------------------

    def _compute_rouge(self, predictions: Dict[str, str]) -> float:
        warnings.filterwarnings("ignore")

        keys       = list(predictions.keys())
        candidates = [self.clean(predictions[k]) for k in keys]
        references = [self.clean(self.gt[k])      for k in keys]

        scores: List[Tuple[int, float]] = []
        non_empty_cands, non_empty_refs, non_empty_idx = [], [], []

        for i, (c, r) in enumerate(zip(candidates, references)):
            if len(c) == 0 and len(r) == 0:
                scores.append((i, 1.0))
            else:
                non_empty_cands.append(c)
                non_empty_refs.append(r)
                non_empty_idx.append(i)

        if non_empty_cands:
            try:
                result = self._rouge.compute(
                    predictions    = non_empty_cands,
                    references     = non_empty_refs,
                    use_aggregator = False,
                )
                for i, score in zip(non_empty_idx, result["rouge1"]):
                    scores.append((i, float(score)))
            except Exception as exc:
                print(f"  [WARN] ROUGE batch failed ({exc}), assigning 0.0 for affected items.")
                for i in non_empty_idx:
                    scores.append((i, 0.0))

        scores.sort(key=lambda x: x[0])
        return float(np.mean([s for _, s in scores]))

    # ------------------------------------------------------------------
    # CSV loading helpers
    # ------------------------------------------------------------------

    def _load_csv(self, path: str) -> Dict[str, str]:
        """Read ``ID,Caption`` CSV into a dict; tolerates missing header."""
        pairs = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            first  = next(reader)
            if "ID" not in first[0]:          # no header — process the first row too
                pairs[first[0]] = first[1]
            for row in reader:
                if len(row) >= 2:
                    pairs[row[0]] = row[1]
        return pairs

    def _load_predictions(self, path: str) -> Dict[str, str]:
        """
        Read predictions CSV, validating against the ground-truth keys.

        Raises
        ------
        Exception  on format errors, unknown IDs, duplicates, or count mismatches.
        """
        gt_ids   = set(self.gt.keys())
        seen     = []
        pairs    = {}
        line_cnt = 0

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or "ID" in row[0]:
                    continue
                line_cnt += 1

                if len(row) < 2:
                    raise ValueError(
                        f"Wrong format at line {line_cnt}: "
                        "each row must be <imageID>,<caption>"
                    )

                img_id, caption = row[0], row[1]

                if img_id not in gt_ids:
                    raise ValueError(f"Unknown image ID '{img_id}' at line {line_cnt}.")
                if img_id in seen:
                    raise ValueError(f"Duplicate image ID '{img_id}' at line {line_cnt}.")

                seen.append(img_id)
                pairs[img_id] = caption

        if len(seen) != len(gt_ids):
            raise ValueError(
                f"Submission has {len(seen)} entries but ground truth has {len(gt_ids)}."
            )

        return pairs
