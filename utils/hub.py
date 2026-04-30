"""
utils/hub.py — Hugging Face Hub integration with safe offline fallback.

Rules:
  - If HubConfig.is_configured is False, every function is a no-op.
  - Network errors never propagate — they are logged and Hub is disabled
    for the rest of the process so training is never interrupted.
  - init_hub() is idempotent; safe to call repeatedly.
"""

from __future__ import annotations

import warnings

from config import CFG

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_hf_api = None          # huggingface_hub.HfApi instance, set after init
_hub_ok: bool = True    # flipped to False on first failure; disables all ops


def _is_available() -> bool:
    """Return True when Hub is both configured and has not previously failed."""
    if not _hub_ok:
        return False
    if not CFG.hub.is_configured:
        warnings.warn(
            "[hub] HF Hub is not configured (token or repo_id is a placeholder). "
            "Set CFG.hub.token and CFG.hub.repo_id, or export HF_TOKEN, to enable "
            "Hub uploads. Training will continue without pushing.",
            stacklevel=3,
        )
        return False
    return True


def init_hub():
    """
    Authenticate with the HF Hub and ensure the target repository exists.

    Safe to call multiple times — authentication and repo creation are both
    idempotent.  Silently disables Hub and returns None if anything fails.

    Returns
    -------
    HfApi | None
    """
    global _hf_api, _hub_ok

    if _hf_api is not None:
        return _hf_api  # already initialised

    if not _is_available():
        return None

    try:
        from huggingface_hub import HfApi, create_repo, login

        login(token=CFG.hub.token, add_to_git_credential=False)
        create_repo(
            repo_id   = CFG.hub.repo_id,
            repo_type = "model",
            exist_ok  = True,
            private   = CFG.hub.private,
        )
        _hf_api = HfApi()
        print(f"[hub] HF repo ready: https://huggingface.co/{CFG.hub.repo_id}")
        return _hf_api

    except Exception as exc:
        _hub_ok = False
        warnings.warn(
            f"[hub] Hub initialisation failed ({exc}). "
            "Hub uploads will be skipped for this run. Training continues normally.",
            stacklevel=2,
        )
        return None


def push_to_hub(
    folder_path: str,
    commit_message: str = "Model checkpoint",
) -> None:
    """
    Upload ``folder_path`` to ``CFG.hub.repo_id``.

    Completely safe to call at any time — silently skips if Hub is not
    configured, not initialised, or has previously encountered an error.

    Parameters
    ----------
    folder_path    : Local directory whose contents will be uploaded.
    commit_message : Commit message shown on the HF Hub.
    """
    global _hub_ok

    if not _is_available():
        return

    api = _hf_api or init_hub()
    if api is None:
        return  # init failed — already warned

    try:
        print(f"  [hub] Pushing to {CFG.hub.repo_id} …")
        api.upload_folder(
            folder_path    = folder_path,
            repo_id        = CFG.hub.repo_id,
            repo_type      = "model",
            commit_message = commit_message,
        )
        print(f"  [hub] ✓ Pushed: https://huggingface.co/{CFG.hub.repo_id}")

    except Exception as exc:
        _hub_ok = False
        warnings.warn(
            f"  [hub] Push failed ({exc}). "
            "Hub uploads disabled for the rest of this run. Training continues.",
            stacklevel=2,
        )
