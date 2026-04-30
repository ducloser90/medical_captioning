"""
data/download.py — Download and extract the Rocov2 dataset from Google Drive.
"""

from __future__ import annotations

import os
import zipfile

from config import CFG


def download_and_extract(
    file_id: str | None = None,
    zip_path: str | None = None,
    extract_to: str | None = None,
) -> str:
    """
    Download the Rocov2 ZIP from Google Drive (if absent) and extract it.

    Parameters
    ----------
    file_id:    Google Drive file ID.  Defaults to ``CFG.data.gdrive_file_id``.
    zip_path:   Local destination for the ZIP.  Defaults to ``CFG.data.zip_path``.
    extract_to: Directory to extract into.  Defaults to ``CFG.data.extract_to``.

    Returns
    -------
    str — Absolute path to the extracted dataset root.

    Raises
    ------
    RuntimeError  if the download succeeds but extraction validation fails.
    """
    file_id    = file_id    or CFG.data.gdrive_file_id
    zip_path   = os.path.abspath(zip_path   or CFG.data.zip_path)
    extract_to = os.path.abspath(extract_to or CFG.data.extract_to)

    # ── Guard: already fully extracted? ──────────────────────────────────────
    if CFG.data.is_extracted():
        print(f"Dataset already extracted at {CFG.data.data_root}, skipping.")
        _print_tree(CFG.data.data_root, max_depth=2)
        return CFG.data.data_root

    # ── Ensure parent directories exist before writing the ZIP ───────────────
    zip_parent = os.path.dirname(zip_path)
    if zip_parent:
        os.makedirs(zip_parent, exist_ok=True)
    os.makedirs(extract_to, exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────────
    if not os.path.isfile(zip_path):
        _download(file_id, zip_path)
    else:
        print(f"ZIP already present at {zip_path}, skipping download.")

    # ── Extract ───────────────────────────────────────────────────────────────
    _extract(zip_path, extract_to)

    # ── Validate ──────────────────────────────────────────────────────────────
    if not CFG.data.is_extracted():
        raise RuntimeError(
            f"Extraction appeared to succeed but sentinel CSVs are missing "
            f"under {CFG.data.data_root}. The ZIP content may differ from the "
            f"expected layout (train/captions.csv, validation/captions.csv, "
            f"test/captions.csv)."
        )

    _print_tree(CFG.data.data_root, max_depth=2)
    return CFG.data.data_root


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download(file_id: str, zip_path: str) -> None:
    """Download from Google Drive, with a clear error if gdown is unavailable."""
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for dataset download. "
            "Install it with: pip install gdown"
        )

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading dataset from Google Drive → {zip_path}")
    try:
        gdown.download(url, zip_path, quiet=False, fuzzy=True)
    except Exception as exc:
        # Clean up a partial download so a retry starts fresh.
        if os.path.isfile(zip_path):
            os.remove(zip_path)
        raise RuntimeError(
            f"Dataset download failed: {exc}\n"
            "Check your internet connection and that the Google Drive file is "
            "publicly accessible."
        ) from exc

    if not os.path.isfile(zip_path) or os.path.getsize(zip_path) == 0:
        raise RuntimeError(
            f"Download appeared to complete but {zip_path} is missing or empty."
        )


def _extract(zip_path: str, extract_to: str) -> None:
    """Extract ZIP, removing the archive on success to free disk space."""
    print(f"Extracting {zip_path} → {extract_to} …")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"{zip_path} is not a valid ZIP file (possibly a partial download). "
            f"Delete it and re-run to trigger a fresh download."
        ) from exc
    print("Extraction complete.")


def _print_tree(root: str, max_depth: int = 2) -> None:
    """Print a compact directory tree for quick sanity-checks."""
    if not os.path.isdir(root):
        print(f"[download] Warning: {root} does not exist, skipping tree print.")
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        depth = dirpath.replace(root, "").count(os.sep)
        if depth > max_depth:
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath)}/")
        for fname in filenames[:3]:
            print(f"{indent}  {fname}")
