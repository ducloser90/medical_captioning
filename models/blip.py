"""
models/blip.py — Load ``BlipForConditionalGeneration``, selectively unfreeze
layers, and optionally enable gradient checkpointing.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint
from transformers import BlipForConditionalGeneration, BlipProcessor

from config import CFG


def build_model(
    device: torch.device,
) -> tuple[BlipForConditionalGeneration, BlipProcessor]:
    """
    Instantiate BLIP-large, freeze the backbone, then selectively re-enable
    gradients on the text decoder and the last N ViT blocks.

    Parameters
    ----------
    device: Target CUDA / CPU device.

    Returns
    -------
    (model, processor) — both ready for training.
    """
    processor = BlipProcessor.from_pretrained(CFG.model.model_name)
    model     = BlipForConditionalGeneration.from_pretrained(CFG.model.model_name)

    _freeze_all(model)
    _unfreeze_text_decoder(model)
    _unfreeze_vit_blocks(model, CFG.model.unfreeze_vit_blocks)

    if CFG.model.grad_checkpointing:
        _enable_grad_checkpointing(model)

    _log_param_counts(model)

    model = model.to(device)
    print(f"Model loaded on {device}.")
    return model, processor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _freeze_all(model: BlipForConditionalGeneration) -> None:
    """Freeze every parameter in the model."""
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_text_decoder(model: BlipForConditionalGeneration) -> None:
    """Allow gradients on all text-decoder parameters."""
    for param in model.text_decoder.parameters():
        param.requires_grad = True


def _unfreeze_vit_blocks(
    model: BlipForConditionalGeneration,
    n_blocks: int,
) -> None:
    """
    Unfreeze the last ``n_blocks`` ViT transformer blocks and the
    post-LayerNorm.  No-op when ``n_blocks == 0``.
    """
    if n_blocks <= 0:
        return

    vit_layers   = model.vision_model.encoder.layers
    total_blocks = len(vit_layers)

    for block in vit_layers[total_blocks - n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    for param in model.vision_model.post_layernorm.parameters():
        param.requires_grad = True

    print(f"Unfroze last {n_blocks}/{total_blocks} ViT blocks + post-LayerNorm.")


def _enable_grad_checkpointing(model: BlipForConditionalGeneration) -> None:
    """Enable gradient checkpointing on the text decoder BERT encoder."""
    encoder = model.text_decoder.bert.encoder
    encoder.gradient_checkpointing = True
    encoder._gradient_checkpointing_func = checkpoint
    print("Gradient checkpointing enabled on text decoder.")


def _log_param_counts(model: BlipForConditionalGeneration) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Parameters — Total: {total / 1e6:.1f} M | "
        f"Trainable: {trainable / 1e6:.1f} M ({trainable / total * 100:.1f} %)"
    )
