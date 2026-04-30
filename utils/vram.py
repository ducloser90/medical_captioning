"""
utils/vram.py — GPU memory reporting utilities.
"""

from __future__ import annotations

import torch


def vram_summary() -> str:
    """
    Return a multi-line string with allocated / reserved / total VRAM per GPU.

    Example output
    --------------
    GPU 0: 8.3 GB allocated / 9.0 GB reserved / 40.0 GB total
    """
    lines = []
    for i in range(torch.cuda.device_count()):
        alloc  = torch.cuda.memory_allocated(i) / 1e9
        reserv = torch.cuda.memory_reserved(i)  / 1e9
        total  = torch.cuda.get_device_properties(i).total_memory / 1e9
        lines.append(
            f"  GPU {i}: {alloc:.1f} GB allocated / "
            f"{reserv:.1f} GB reserved / {total:.1f} GB total"
        )
    return "\n".join(lines) if lines else "  No CUDA GPUs detected."
