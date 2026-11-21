#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Loading the upstream LingBot-World model.

Imports a local clone at ``UPSTREAM_COMMIT``, selected by ``--lingbot-root`` or
``LINGBOT_WORLD_ROOT``. See docs/installation.md#lingbot for source and weight setup.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from fastframes.device import default_device
from fastframes.wan.mps_conv3d import patch_conv3d_class

__all__ = ["UPSTREAM_URL", "UPSTREAM_COMMIT", "import_upstream", "Upstream", "load_upstream"]

UPSTREAM_URL = "https://github.com/robbyant/lingbot-world-v2"
UPSTREAM_COMMIT = "1895d300d8ac936401689b26389f51cbd36530eb"
ROOT_ENV = "LINGBOT_WORLD_ROOT"


def _git_head(root: Path):
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def import_upstream(root=None):
    """Import the upstream ``wan`` package from a LingBot-World clone."""
    root = Path(root or os.environ.get(ROOT_ENV, "")).expanduser()
    if not (root / "wan" / "modules" / "model_fast.py").is_file():
        raise FileNotFoundError(
            f"no LingBot-World clone at '{root}'; pass --lingbot-root or set {ROOT_ENV}")
    root = root.resolve()
    loaded = sys.modules.get("wan")
    if loaded is not None:
        if root not in Path(loaded.__file__).resolve().parents:
            raise RuntimeError(f"another 'wan' package is already imported from {loaded.__file__}")
        return loaded
    head = _git_head(root)
    if head is not None and head != UPSTREAM_COMMIT:
        warnings.warn(f"LingBot-World clone is at {head}; FastFrames-LingBot was verified at {UPSTREAM_COMMIT}")
    sys.path.insert(0, str(root))
    if torch.cuda.is_available():
        return importlib.import_module("wan")
    # wan/modules/t5.py evaluates ``device=torch.cuda.current_device()`` as a default argument at
    # import time, which raises without CUDA. Callers always pass the device explicitly.
    current_device = torch.cuda.current_device
    torch.cuda.current_device = lambda: "cpu"
    try:
        wan = importlib.import_module("wan")
    finally:
        torch.cuda.current_device = current_device
    if torch.backends.mps.is_available():
        # MPS conv3d is ~4x slower than per-tap conv2d; the VAE image condition is mostly 3x3x3 convs
        patch_conv3d_class(importlib.import_module("wan.modules.vae2_1").CausalConv3d)
    return wan


@dataclass
class Upstream:
    """Official components: causal-fast DiT, umT5 text encoder, Wan2.1 VAE, flow scheduler."""

    config: object
    dit: torch.nn.Module
    text_encoder: object
    vae: object
    scheduler: object


def _weight_files(dit_dir: Path):
    for index in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        if (dit_dir / index).is_file():
            weight_map = json.loads((dit_dir / index).read_text())["weight_map"]
            return [dit_dir / name for name in sorted(set(weight_map.values()))]
    for name in ("model.safetensors", "diffusion_pytorch_model.safetensors"):
        if (dit_dir / name).is_file():
            return [dit_dir / name]
    raise FileNotFoundError(f"no DiT safetensors in {dit_dir}")


def load_dit(config, ckpt_dir, *, fp32_time_weights: bool, device=None):
    """Upstream ``WanModelFast`` with BF16 weights and an FP32 time MLP.

    ``fp32_time_weights`` keeps the time MLP weights of the FP32 checkpoint; otherwise they are
    rounded to BF16 like all other weights (the upstream loader) and evaluated in FP32.
    """
    from safetensors import safe_open
    from wan.image2video import _dit_kwargs_from_config, _resolve_dit_dir
    from wan.modules.model_fast import WanModelFast

    with torch.device("meta"):
        model = WanModelFast(**_dit_kwargs_from_config(config, dict(local_attn_size=18, sink_size=6)))
    state = {}
    for path in _weight_files(Path(_resolve_dit_dir(str(ckpt_dir), config.fast_checkpoint))):
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            for name in shard.keys():
                tensor = shard.get_tensor(name)
                if name.startswith(("time_embedding.", "time_projection.")):
                    state[name] = tensor.float() if fp32_time_weights else tensor.to(torch.bfloat16).float()
                else:
                    state[name] = tensor.to(torch.bfloat16)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False).to(device or default_device())


def _load_t5(cls, config, checkpoint_path, tokenizer_path):
    """Upstream ``T5EncoderModel`` on the CPU, built without its random initialisation.

    The upstream constructor initialises the 5.7B-parameter encoder randomly before loading the
    checkpoint, which took about a minute. Building it on the meta device and assigning the
    checkpoint tensors gives the same module and weights.
    """
    from wan.modules.t5 import umt5_xxl
    from wan.modules.tokenizers import HuggingfaceTokenizer

    encoder = cls.__new__(cls)
    encoder.text_len, encoder.dtype, encoder.device = config.text_len, config.t5_dtype, torch.device("cpu")
    encoder.checkpoint_path, encoder.tokenizer_path = checkpoint_path, tokenizer_path
    model = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=config.t5_dtype, device="meta")
    state = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict({k: v.to(config.t5_dtype) for k, v in state.items()}, assign=True)
    encoder.model = model.eval().requires_grad_(False)
    encoder.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=config.text_len, clean="whitespace")
    return encoder


def load_upstream(config_name, ckpt_dir, assets_dir=None, *, fp32_time_weights, device=None):
    """Build the official components; T5/VAE files are looked up in ``ckpt_dir`` then ``assets_dir``."""
    from wan.configs import WAN_CONFIGS
    from wan.image2video import _resolve_asset_path
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    device = device or default_device()
    config = WAN_CONFIGS[config_name]
    ckpt_dir, assets_dir = str(ckpt_dir), None if assets_dir is None else str(assets_dir)
    text_encoder = _load_t5(T5EncoderModel, config,
                            _resolve_asset_path(config.t5_checkpoint, ckpt_dir, assets_dir),
                            _resolve_asset_path(config.t5_tokenizer, ckpt_dir, assets_dir))
    vae = Wan2_1_VAE(vae_pth=_resolve_asset_path(config.vae_checkpoint, ckpt_dir, assets_dir), device=device)
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=config.num_train_timesteps, shift=1,
                                            use_dynamic_shifting=False)
    dit = load_dit(config, ckpt_dir, fp32_time_weights=fp32_time_weights, device=device)
    return Upstream(config, dit, text_encoder, vae, scheduler)
