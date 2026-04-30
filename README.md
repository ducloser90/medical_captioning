# Medical Image Captioning — BLIP Fine-tuning on Rocov2

Fine-tuning [`Salesforce/blip-image-captioning-large`](https://huggingface.co/Salesforce/blip-image-captioning-large) on the **Rocov2** medical captioning dataset, evaluated with the **ImageCLEFmedical** protocol (BERTScore + ROUGE-1).

Runs without modification on **Kaggle**, **Google Colab**, and any local machine. No manual path editing required.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Step 0 — Clone and Install](#step-0--clone-and-install)
4. [Step 1 — Configure](#step-1--configure)
5. [Step 2 — Train](#step-2--train)
6. [Step 3 — Generate Predictions](#step-3--generate-predictions)
7. [Step 4 — Evaluate](#step-4--evaluate)
8. [Step 5 — Run Tests](#step-5--run-tests)
9. [Platform-Specific Notes](#platform-specific-notes)
10. [CLI Reference](#cli-reference)
11. [Configuration Reference](#configuration-reference)
12. [Model Architecture](#model-architecture)
13. [Key Design Decisions](#key-design-decisions)

---

## Project Structure

```
medical_captioning/
├── config.py                  # Single source of truth for all settings
│
├── data/
│   ├── download.py            # Google Drive download + ZIP extraction
│   └── dataset.py             # MedCapDataset + build_loaders()
│
├── models/
│   └── blip.py                # build_model(): load, freeze, unfreeze, checkpointing
│
├── training/
│   └── trainer.py             # Trainer: optimiser, scheduler, train/val loop, HF push
│
├── evaluation/
│   └── evaluator.py           # CaptionEvaluator: BERTScore (primary) + ROUGE-1
│
├── inference/
│   └── predictor.py           # Predictor: batched generation + single-image API
│
├── utils/
│   ├── vram.py                # GPU memory monitoring
│   └── hub.py                 # Hugging Face Hub helpers (safe offline fallback)
│
├── scripts/
│   ├── train.py               # Entry-point: full training pipeline
│   ├── predict.py             # Entry-point: generate test predictions + write CSVs
│   └── evaluate.py            # Entry-point: score predictions against ground truth
│
├── tests/
│   └── test_core.py           # Unit tests (no GPU required)
│
├── requirements.txt
└── README.md
```

---

## Requirements

| Requirement | Notes |
|---|---|
| Python ≥ 3.9 | `|` union type hints are used throughout |
| PyTorch ≥ 2.1 | For `torch.amp.GradScaler` API |
| CUDA GPU (optional) | Training works on CPU but will be very slow |
| ~20 GB disk space | Dataset ZIP + extracted files + checkpoint |
| Internet access | First run only — downloads dataset and model weights |

---

## Step 0 — Clone and Install

```bash
git clone https://github.com/ducloser90/medical_captioning.git
cd medical_captioning
pip install -r requirements.txt
```

On **Kaggle / Colab**, add `!` before each command and run in a notebook cell:

```python
!pip install -r requirements.txt
```

---

## Step 1 — Configure

All settings live in `config.py`. The project auto-detects your environment:

| Environment | Working directory auto-detected as |
|---|---|
| Kaggle | `/kaggle/working` |
| Google Colab | `/content` |
| Local / other | Current working directory (`os.getcwd()`) |

Data, checkpoints, and output CSVs are all written under this directory automatically.

### Hugging Face Hub (optional)

Hub uploads are **completely optional**. If you skip this, training still runs and saves checkpoints locally.

**Option A — environment variable (recommended)**

```bash
export HF_TOKEN="hf_..."          # set once in your shell / notebook secrets
```

**Option B — edit `config.py` directly**

```python
# config.py  ↓  HubConfig section
repo_id = "your-username/your-repo-name"
token   = "hf_..."
```

If neither is set, Hub pushes are silently skipped and a warning is printed. Training is never interrupted.

---

## Step 2 — Train

```bash
python scripts/train.py
```

This single command runs the full pipeline in order:

1. **Downloads** the Rocov2 dataset from Google Drive into the working directory (skipped if already present).
2. **Extracts** the ZIP and validates the three CSV sentinel files (skipped if already extracted).
3. **Loads** `Salesforce/blip-image-captioning-large` from the Hugging Face Hub.
4. **Freezes** the full model, then selectively unfreezes the last 4 ViT blocks and the entire text decoder.
5. **Trains** for 4 epochs with:
   - Cosine LR schedule with warm-up
   - Mixed precision (fp16, CUDA only — disabled automatically on CPU)
   - Gradient accumulation (effective batch size = 64)
   - Differential learning rates: vision encoder at `lr/10`, decoder at `lr`
6. **Saves** the best checkpoint (lowest validation loss) to `<working_dir>/blip_large_rocov2/best_model/`.
7. **Pushes** the checkpoint to the Hugging Face Hub after each improvement (if configured).

Expected output at the end of each epoch:

```
Epoch 01 | Train Loss 2.3145 | Val Loss 2.1823
Epoch 02 | Train Loss 1.9871 | Val Loss 1.9102
...
Training complete. Best Val Loss: 1.8744
Training history saved to <working_dir>/blip_large_rocov2/history.json
```

---

## Step 3 — Generate Predictions

```bash
python scripts/predict.py
```

This loads the best checkpoint and runs batched beam-search inference over the entire test split. It writes **two files** to the same output directory:

| File | Contents | Used by |
|---|---|---|
| `predictions_only.csv` | Model-generated captions | `evaluate.py` as `--pred` |
| `ground_truth_only.csv` | Reference captions from the test split | `evaluate.py` as `--gt` |

Both files have exactly two columns: `ID, Caption`.

At the end, the script prints the exact evaluate command to run next:

```
Done. To evaluate, run:
  python scripts/evaluate.py --gt .../ground_truth_only.csv --pred .../predictions_only.csv
```

### Custom output directory

```bash
python scripts/predict.py --out-dir /path/to/output/dir
```

### Custom checkpoint

```bash
python scripts/predict.py --ckpt /path/to/checkpoint
```

### Both together

```bash
python scripts/predict.py \
    --ckpt   /path/to/checkpoint \
    --out-dir /path/to/output/dir
```

---

## Step 4 — Evaluate

```bash
python scripts/evaluate.py
```

If you used a custom `--out-dir` in Step 3, pass the paths explicitly:

```bash
python scripts/evaluate.py \
    --gt   /path/to/ground_truth_only.csv \
    --pred /path/to/predictions_only.csv
```

Expected output:

```
── Evaluation Results ──────────────────────────────
  BERTScore (primary)   : 0.6073
  ROUGE-1   (secondary) : 0.2272
```

**BERTScore** (primary) uses `microsoft/deberta-xlarge-mnli` and is downloaded automatically on first run (~900 MB). **ROUGE-1** uses the `evaluate` library and needs no extra downloads.

---

## Step 5 — Run Tests

Unit tests cover Config, Dataset, Evaluator edge cases, and VRAM utilities. No GPU is required.

```bash
pytest tests/ -v
```

Expected output:

```
tests/test_core.py::TestConfig::test_effective_batch_size   PASSED
tests/test_core.py::TestConfig::test_paths_are_strings      PASSED
tests/test_core.py::TestMedCapDataset::test_len             PASSED
tests/test_core.py::TestMedCapDataset::test_fallback_image  PASSED
...
9 passed in 0.42s
```

---

## Platform-Specific Notes

### Kaggle

```python
# Cell 1 — install
!pip install -r /kaggle/input/<your-dataset>/requirements.txt

# Cell 2 — set HF token via Kaggle Secrets (recommended) or inline
import os
os.environ["HF_TOKEN"] = "hf_..."   # or use kaggle_secrets

# Cell 3 — train
!python /kaggle/input/<your-dataset>/scripts/train.py
```

Kaggle provides two T4 GPUs. The project uses only `cuda:0` by default — no DataParallel needed.

### Google Colab

```python
# Mount Drive if your dataset is there (optional)
from google.colab import drive
drive.mount("/content/drive")

# Install
!pip install -r requirements.txt

# Set token
import os
os.environ["HF_TOKEN"] = "hf_..."   # or use Colab Secrets

# Train
!python scripts/train.py
```

If you run out of Colab RAM during the BERTScore evaluation step, reduce `CFG.eval.bert_batch_size` from 32 to 8.

### Local machine (CPU only)

The project runs fully on CPU. fp16 and VRAM probing are disabled automatically. Expect training to be ~10–20× slower than on a GPU. For quick local testing, reduce epochs and batch size:

```python
# At the top of your notebook or script, before importing anything else:
from config import CFG
CFG.train.epochs     = 1
CFG.train.batch_size = 4
CFG.train.accum_steps = 1
```

---

## CLI Reference

### `scripts/train.py`

```
python scripts/train.py
```

No required arguments. All settings come from `config.py`.

---

### `scripts/predict.py`

```
python scripts/predict.py [--ckpt PATH] [--out-dir DIR] [--max-tokens N] [--num-beams N]
```

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | `CFG.best_ckpt_dir` | Path to a trained checkpoint directory |
| `--out-dir` | Same dir as `CFG.eval.submission_file_path` | Where to write both output CSVs |
| `--max-tokens` | `64` | Maximum tokens to generate per caption |
| `--num-beams` | `4` | Beam-search width |

Always writes two files: `predictions_only.csv` and `ground_truth_only.csv`.

---

### `scripts/evaluate.py`

```
python scripts/evaluate.py [--gt PATH] [--pred PATH]
```

| Argument | Default | Description |
|---|---|---|
| `--gt` | `CFG.eval.ground_truth_path` | Path to `ground_truth_only.csv` |
| `--pred` | `CFG.eval.submission_file_path` | Path to `predictions_only.csv` |

---

## Configuration Reference

All fields are in `config.py` under their respective dataclass.

### `DataConfig`

| Field | Default | Description |
|---|---|---|
| `data_root` | `<work_dir>/Rocov2_Dataset` | Root of the extracted dataset |
| `img_col` | `"ID"` | CSV column for image IDs |
| `caption_col` | `"Caption"` | CSV column for captions |
| `gdrive_file_id` | *(see config)* | Google Drive file ID for the dataset ZIP |

### `ModelConfig`

| Field | Default | Description |
|---|---|---|
| `model_name` | `"Salesforce/blip-image-captioning-large"` | HF model ID |
| `max_length` | `128` | Max caption token length |
| `unfreeze_vit_blocks` | `4` | Last N ViT blocks to unfreeze (0 = fully frozen) |
| `grad_checkpointing` | `True` | Enable gradient checkpointing on text decoder |

### `TrainConfig`

| Field | Default | Description |
|---|---|---|
| `batch_size` | `32` | Per-GPU batch size |
| `accum_steps` | `2` | Gradient accumulation steps (effective batch = 64) |
| `lr` | `5e-5` | Base learning rate for the text decoder |
| `weight_decay` | `1e-2` | AdamW weight decay |
| `epochs` | `4` | Total training epochs |
| `warmup_ratio` | `0.1` | Fraction of total steps used for LR warm-up |
| `fp16` | `True` | Mixed precision (auto-disabled on CPU) |
| `num_workers` | `4` | DataLoader worker processes |
| `seed` | `42` | Global random seed |
| `output_dir` | `<work_dir>/blip_large_rocov2` | Checkpoint and history output directory |

### `HubConfig`

| Field | Default | Description |
|---|---|---|
| `repo_id` | `"YOUR_HF_REPO_ID"` | `username/repo-name` on Hugging Face |
| `token` | `$HF_TOKEN` env var | Hugging Face write token |
| `push_every_epoch` | `True` | Push checkpoint after each val-loss improvement |
| `private` | `False` | Create repo as private |

### `EvalConfig`

| Field | Default | Description |
|---|---|---|
| `bert_model` | `"microsoft/deberta-xlarge-mnli"` | BERTScore backbone |
| `bert_batch_size` | `32` | Batch size for BERTScore (reduce if OOM) |
| `ground_truth_path` | `<work_dir>/ground_truth_only.csv` | Reference CSV |
| `submission_file_path` | `<work_dir>/predictions_only.csv` | Predictions CSV |

---

## Model Architecture

```
BlipForConditionalGeneration
├── vision_model (ViT-Large)
│   ├── encoder.layers[0..19]     ← frozen
│   ├── encoder.layers[20..23]    ← trainable  (last 4 blocks)
│   └── post_layernorm            ← trainable
└── text_decoder (BERT-style)     ← fully trainable
    └── bert.encoder              ← gradient checkpointing enabled
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Auto-detected working directory** | `config.py` probes `/kaggle/working` → `/content` → `os.getcwd()` at import time, so zero path editing is needed on any platform. |
| **Properties instead of fields for derived paths** | `zip_path`, `extract_to`, and all `*_img_dir` / `*_csv` paths are `@property` on `DataConfig`. Changing `data_root` updates everything automatically. |
| **`is_extracted()` sentinel check** | Checks for all three CSVs rather than just `os.path.exists(data_root)`, catching partial extractions. |
| **Hub safe fallback** | `HubConfig.is_configured` validates token and repo_id against placeholder strings. A module-level `_hub_ok` flag is flipped `False` on first failure, making all subsequent Hub calls instant no-ops. Training is never interrupted. |
| **`fp16` and `GradScaler` guarded by `device.type`** | `self._use_amp = CFG.train.fp16 and device.type == "cuda"` — fp16 is silently disabled on CPU instead of crashing. |
| **`pin_memory` guarded by `cuda.is_available()`** | Avoids crashes on CPU-only machines with no penalty on GPU. |
| **Selective ViT unfreeze** | Freezing the vision backbone and only unfreezing the last 4 blocks reduces VRAM by ~40% and prevents catastrophic forgetting of visual features. |
| **Differential learning rates** | Vision encoder at `lr/10`, text decoder at `lr` — preserves pre-trained visual representations while allowing the decoder to adapt. |
| **Both CSVs written in one predict run** | `predict.py` writes `predictions_only.csv` and `ground_truth_only.csv` together so the evaluator can run immediately with no manual file preparation. |
| **`bert_score` monkey-patch** | Upstream `sent_encode` omits `truncation=True`, causing a Rust-backend `OverflowError` on long captions. The patch is applied at `evaluator.py` import time. |
