#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Checkpoint loading for Wan backbones, FastFrames generators and their add-on networks."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .device import default_device
from .wan.causal_model import CausalWanModel
from .wan.int8 import quantize_blocks
from .wan.model import WAN_CONFIGS, WanModel

__all__ = ["load_state_dict", "transform_state", "build_wan", "load_wan_base"]

_PREFIXES = ("model._fsdp_wrapped_module.", "_fsdp_wrapped_module.", "model.")


def load_state_dict(path, key=None, device="cpu", transform=None):
    """Tensor state from ``.safetensors`` (a file, or the ``.safetensors.index.json`` of a sharded
    file) or a torch pickle (optionally nested under ``key``).

    ``device`` places ``.safetensors`` tensors one by one; loading to CPU and then moving the model
    holds two copies at once, which a 14B DiT (28.6 GB) can't afford in 64 GB of unified memory.
    ``transform(name, tensor) -> {name: tensor}``, if given, is applied to each tensor as it is
    read (e.g. :func:`~fastframes.wan.int8.quantize_state`), so the untransformed state is never
    resident all at once."""
    path = Path(path)
    if path.name.endswith(".safetensors.index.json"):
        files = [path.parent / name for name in sorted(set(json.loads(path.read_text())["weight_map"].values()))]
    elif path.suffix == ".safetensors":
        files = [path]
    else:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if key is not None:
            state = state[key]
        return transform_state(((_strip(k), v) for k, v in state.items()), transform)
    if transform is None:
        from safetensors.torch import load_file
        state = {}
        for file in files:
            state.update(load_file(str(file), device=str(device)))
        return {_strip(k): v for k, v in state.items()}
    from safetensors import safe_open

    def tensors():
        for file in files:
            with safe_open(str(file), framework="pt", device=str(device)) as f:
                for name in f.keys():
                    yield _strip(name), f.get_tensor(name)
    return transform_state(tensors(), transform)


def transform_state(items, transform):
    """``{name: tensor}`` from ``(name, tensor)`` pairs, through ``transform`` if given."""
    if transform is None:
        return dict(items)
    state = {}
    for name, tensor in items:
        state.update(transform(name, tensor))
    return state


def _strip(name):
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def build_wan(name, state_dict, *, causal=True, dtype=torch.bfloat16, device=None, int8=False):
    """Instantiate a (causal) Wan2.1 transformer without materialising random weights.

    ``int8`` builds the blocks with :class:`~fastframes.wan.int8.Int8Linear` layers; the state
    must then come from :func:`load_state_dict` with ``transform=quantize_state``."""
    device = device or default_device()
    cls = CausalWanModel if causal else WanModel
    with torch.device("meta"):
        model = cls(WAN_CONFIGS[name])
        if int8:
            quantize_blocks(model)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def load_wan_base(wan_dir, *, causal=False, dtype=torch.bfloat16, device=None):
    """Load the released Wan2.1 weights (``diffusion_pytorch_model*.safetensors``)."""
    from safetensors.torch import load_file
    wan_dir = Path(wan_dir)
    index = wan_dir / "diffusion_pytorch_model.safetensors.index.json"
    files = (sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists()
             else ["diffusion_pytorch_model.safetensors"])
    state = {}
    for name in files:
        state.update(load_file(str(wan_dir / name)))
    config = json.loads((wan_dir / "config.json").read_text())
    name = "Wan2.1-T2V-14B" if config["dim"] == 5120 else "Wan2.1-T2V-1.3B"
    return build_wan(name, state, causal=causal, dtype=dtype, device=device)
