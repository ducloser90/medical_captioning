"""
config.py — Central configuration for the medical captioning project.

All paths are derived from the project's working directory at runtime, so
the project runs without modification on Kaggle, Colab, or any local machine.

Override any value after import:

    from config import CFG
    CFG.data.data_root = "/my/custom/dataset"
    CFG.hub.token = os.environ["HF_TOKEN"]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Runtime root detection
# ---------------------------------------------------------------------------

def _detect_working_dir() -> str:
    """
    Return the best working directory for storing data and checkpoints.

    Priority:
    1. /kaggle/working  — if running on Kaggle
    2. /content         — if running on Colab
    3. os.getcwd()      — everywhere else
    """
    for candidate in ("/kaggle/working", "/content"):
        if os.path.isdir(candidate):
            return candidate
    return os.getcwd()


# Resolved once at import time and used by all default_factory lambdas.
_WORK_DIR: str = _detect_working_dir()


def _work(*parts: str) -> str:
    """Absolute path joined under the detected working directory."""
    return os.path.join(_WORK_DIR, *parts)


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Paths and column names for the Rocov2 dataset."""

    # Root of the extracted dataset — always under the detected working dir.
    data_root: str = field(default_factory=lambda: _work("Rocov2_Dataset"))

    # CSV column identifiers
    img_col: str = "ID"
    caption_col: str = "Caption"

    # Google Drive source
    gdrive_file_id: str = "1z1JdOCpYoeZxXz5ZfM1JYGlSnHgOyM6s"

    # ── Derived paths (always consistent with data_root) ─────────────────────

    @property
    def zip_path(self) -> str:
        """ZIP file always sits next to data_root's parent directory."""
        return os.path.join(
            os.path.dirname(os.path.abspath(self.data_root)),
            "Rocov2_Dataset.zip",
        )

    @property
    def extract_to(self) -> str:
        """Directory into which the ZIP is extracted."""
        return os.path.dirname(os.path.abspath(self.data_root))

    @property
    def train_img_dir(self) -> str:
        return os.path.join(self.data_root, "train", "images")

    @property
    def val_img_dir(self) -> str:
        return os.path.join(self.data_root, "validation", "images")

    @property
    def test_img_dir(self) -> str:
        return os.path.join(self.data_root, "test", "images")

    @property
    def train_csv(self) -> str:
        return os.path.join(self.data_root, "train", "captions.csv")

    @property
    def val_csv(self) -> str:
        return os.path.join(self.data_root, "validation", "captions.csv")

    @property
    def test_csv(self) -> str:
        return os.path.join(self.data_root, "test", "captions.csv")

    def is_extracted(self) -> bool:
        """True only when all three expected CSV sentinel files exist."""
        return all(
            os.path.isfile(p)
            for p in (self.train_csv, self.val_csv, self.test_csv)
        )


@dataclass
class ModelConfig:
    """BLIP model and ViT fine-tuning settings."""

    model_name: str = "Salesforce/blip-image-captioning-large"
    max_length: int = 128

    # How many of the *last* ViT transformer blocks to unfreeze.
    # Set to 0 to freeze the entire vision encoder.
    unfreeze_vit_blocks: int = 4

    grad_checkpointing: bool = True


@dataclass
class TrainConfig:
    """Optimiser, scheduler, and hardware settings."""

    batch_size: int = 32
    accum_steps: int = 2          # Effective batch = batch_size * accum_steps
    lr: float = 5e-5
    weight_decay: float = 1e-2
    epochs: int = 4
    warmup_ratio: float = 0.1
    fp16: bool = True
    num_workers: int = 4
    seed: int = 42

    # Checkpoints land under the detected working directory.
    output_dir: str = field(default_factory=lambda: _work("blip_large_rocov2"))

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accum_steps


_HUB_PLACEHOLDER_TOKEN   = "YOUR_HF_TOKEN"
_HUB_PLACEHOLDER_REPO_ID = "YOUR_HF_REPO_ID"


@dataclass
class HubConfig:
    """Hugging Face Hub push settings."""

    repo_id: str = _HUB_PLACEHOLDER_REPO_ID
    # Token is automatically read from HF_TOKEN env-var when not set explicitly.
    token: str = field(
        default_factory=lambda: os.environ.get("HF_TOKEN", _HUB_PLACEHOLDER_TOKEN)
    )
    push_every_epoch: bool = True
    private: bool = False

    @property
    def is_configured(self) -> bool:
        """
        True only when both token and repo_id look like real values.
        When False, all Hub operations are skipped without error.
        """
        token_ok = bool(self.token)   and self.token   != _HUB_PLACEHOLDER_TOKEN
        repo_ok  = bool(self.repo_id) and self.repo_id != _HUB_PLACEHOLDER_REPO_ID
        return token_ok and repo_ok


@dataclass
class EvalConfig:
    """Evaluation script settings."""

    bert_model: str = "microsoft/deberta-xlarge-mnli"
    bert_batch_size: int = 32
    max_text_len: int = 512
    case_sensitive: bool = False

    # Override these paths before calling CaptionEvaluator.
    ground_truth_path: str = field(
        default_factory=lambda: _work("ground_truth_only.csv")
    )
    submission_file_path: str = field(
        default_factory=lambda: _work("predictions_only.csv")
    )


# ---------------------------------------------------------------------------
# Top-level composite config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Top-level config object — compose all sub-configs here."""

    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hub:   HubConfig   = field(default_factory=HubConfig)
    eval:  EvalConfig  = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        # Create output dir, but never crash if the filesystem is read-only
        # or the path is invalid (e.g. during unit tests).
        try:
            os.makedirs(self.train.output_dir, exist_ok=True)
        except OSError as exc:
            print(f"[config] Warning: could not create output_dir: {exc}")

    # ── Convenience shortcuts ─────────────────────────────────────────────────

    @property
    def best_ckpt_dir(self) -> str:
        return os.path.join(self.train.output_dir, "best_model")

    @property
    def history_path(self) -> str:
        return os.path.join(self.train.output_dir, "history.json")


# ---------------------------------------------------------------------------
# Module-level singleton — import this in other modules:
#   from config import CFG
# ---------------------------------------------------------------------------
CFG = Config()
