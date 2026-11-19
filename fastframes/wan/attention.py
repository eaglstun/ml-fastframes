# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modifications Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see ACKNOWLEDGEMENTS.
# Modified for FastFrames: attention uses PyTorch SDPA with the cuDNN backend
# (fastest on B200; PyTorch would otherwise auto-select FLASH).
import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import mps_flash_attention

__all__ = ["attention", "cross_attention", "set_cross_attention_backend"]

_cross_attention_backend = "cudnn"
# MPS only: "flash" (Metal kernel in mps_flash_attention.py, default) or "sdpa" (PyTorch, per head).
_MPS_ATTN = os.environ.get("FASTFRAMES_MPS_ATTN", "flash").lower()
if _MPS_ATTN not in ("flash", "sdpa"):
    raise ValueError(f"FASTFRAMES_MPS_ATTN must be 'flash' or 'sdpa', got {_MPS_ATTN!r}")


def attention(q, k, v, causal=False, dropout_p=0.0, backend="cudnn"):
    """BF16 multi-head attention on ``[B, L, H, D]`` tensors.

    ``backend="auto"`` leaves the kernel choice to PyTorch. The result is
    ``[B, L, H, D]`` in the dtype of ``q``.
    """
    out_dtype = q.dtype
    if (q.device.type == "mps" and _MPS_ATTN == "flash" and not causal and dropout_p == 0.0
            and q.shape[1] >= mps_flash_attention.MIN_QUERY_ROWS and mps_flash_attention.is_available(q.shape[-1])):
        # One Metal kernel with online softmax, straight on the [B, L, H, D] layout: ~2x faster than
        # MPS SDPA and nothing is cached per shape.  FASTFRAMES_MPS_ATTN=sdpa selects the path below.
        return mps_flash_attention.flash_attention(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)).to(out_dtype)
    q = q.transpose(1, 2).to(torch.bfloat16)
    k = k.transpose(1, 2).to(torch.bfloat16)
    v = v.transpose(1, 2).to(torch.bfloat16)
    if q.device.type == "mps":
        # MPS caches one MPSGraph per input shape together with its FP32 score buffer, and
        # the self-attention prefix grows every frame (21 shapes): all heads at once retains
        # ~68 GiB, one head per call ~6 GiB, and is also slightly faster.
        out = torch.cat([F.scaled_dot_product_attention(q[:, h:h + 1], k[:, h:h + 1], v[:, h:h + 1],
                                                        is_causal=causal, dropout_p=dropout_p)
                         for h in range(q.shape[1])], dim=1)
    elif backend == "auto" or not q.is_cuda:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=dropout_p)
    else:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=dropout_p)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def set_cross_attention_backend(name):
    """Kernel of the text cross-attention: ``"cudnn"`` (default, used by all released
    inference) or ``"auto"`` (PyTorch's own dispatch, as in the original Wan2.1-14B
    ODE, DMD and KV-refinement training runs)."""
    global _cross_attention_backend
    if name not in ("cudnn", "auto"):
        raise ValueError(f"unknown cross-attention backend {name!r}")
    _cross_attention_backend = name


def cross_attention(q, k, v):
    return attention(q, k, v, backend=_cross_attention_backend)
