"""Weight-only int8 linear layers for the DiT blocks (Apple Silicon memory saving).

Each weight row gets one absmax scale; activations stay in bf16. A layer converts its int8
weight to bf16 for the GEMM and scales the output rows (exact: the scale is per output row), so
only one layer's weight is ever resident in bf16, about 6% slower than a bf16 GEMM on MPS.

``torch._weight_int8pack_mm`` would skip the conversion but is not used: on MPS (torch 2.9.1)
about 1 call in 50 returns a corrupted 16x8 output tile, so runs neither repeat nor match.

A 14B checkpoint is quantized while it loads (:func:`quantize_state`), so the bf16 weights of
the whole model are never resident at once.
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Int8Linear", "quantize_blocks", "quantize_state"]

# Attention and FFN projections of the transformer blocks; everything else stays as loaded.
_BLOCK_LINEAR = re.compile(r"blocks\.\d+\.(?:(?:self_attn|cross_attn)\.[qkvo]|ffn\.[02])\.(weight|bias)$")


def quantize_weight(weight):
    """``[out, in]`` float weight -> (int8 weight, bf16 per-row scale)."""
    scale = weight.float().abs().amax(dim=1).clamp_min(1e-8) / 127
    return (weight.float() / scale[:, None]).round_().clamp_(-127, 127).to(torch.int8), scale.to(torch.bfloat16)


class Int8Linear(nn.Module):
    """``nn.Linear`` with an int8 ``[out, in]`` weight and a bf16 per-row scale.

    Outputs bf16, as ``nn.Linear`` does under the pipeline's bf16 autocast.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight, scale = quantize_weight(linear.weight.detach())
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().to(torch.bfloat16))
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x):
        y = F.linear(x.to(torch.bfloat16), self.weight.to(torch.bfloat16)) * self.scale
        return y if self.bias is None else y + self.bias

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, int8"


def quantize_blocks(model):
    """Swap the attention and FFN linears of every transformer block for ``Int8Linear``.

    Embeddings, time projections, modulation and the head stay in their original dtype. On a
    ``meta`` model this only sets up the int8 layout, for :func:`quantize_state` to fill.
    """
    for block in model.blocks:
        for attn in (block.self_attn, block.cross_attn):
            for name in ("q", "k", "v", "o"):
                setattr(attn, name, Int8Linear(getattr(attn, name)))
        for i in (0, 2):
            block.ffn[i] = Int8Linear(block.ffn[i])
    device = next(model.parameters()).device
    if device.type == "mps":
        torch.mps.empty_cache()
    return model


def quantize_state(name, tensor):
    """Per-tensor loader hook: a block linear's weight becomes ``weight`` (int8) + ``scale``."""
    match = _BLOCK_LINEAR.search(name)
    if match is None:
        return {name: tensor}
    if match.group(1) == "bias":
        return {name: tensor.to(torch.bfloat16)}
    weight, scale = quantize_weight(tensor)
    return {name: weight, name[:-len("weight")] + "scale": scale}
