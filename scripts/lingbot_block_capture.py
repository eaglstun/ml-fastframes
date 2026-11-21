#!/usr/bin/env python3
"""Capture LingBot DiT block 0's real inputs, weights and output during a full-cache chunk (example 00,
chunk 5, first denoising pass) and time the real PyTorch block on them.

    .venv/bin/python scripts/lingbot_block_capture.py --model lingbot-1.3b \\
        --lingbot-root upstream/lingbot-world-v2 --out outputs/block0 [--size 576x320]

Writes <out>.safetensors (~430 MB at 832x464) and <out>.json. scripts/lingbot_block_mlx.py runs the
same block in MLX on them (another venv: this one has no MLX), the reference for porting the block to
another framework or device (e.g. MLX-Swift on an iPad).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])  # repository root
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from apps.lingbot import conditioning  # noqa: E402
from apps.lingbot import dit as lingbot_dit  # noqa: E402
from apps.lingbot.conditioning import Trajectory  # noqa: E402
from apps.lingbot.generate import add_model_arguments, apply_size, load_pipeline  # noqa: E402
from apps.lingbot.upstream import import_upstream  # noqa: E402
from fastframes.device import default_device, limit_mps_memory, synchronize  # noqa: E402

CAPTURE_CHUNK = 5  # the cache holds its full 18 frames from chunk 4 on


class Captured(Exception):
    """Raised once block 0 is saved, to stop the run."""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--out", type=Path, required=True, help="output path without suffix")
    args = parser.parse_args()
    limit_mps_memory(0.75)
    apply_size(args, 1)
    wan = import_upstream(args.lingbot_root)
    example = Path(wan.__file__).resolve().parents[1] / "examples" / "00"
    trajectory = Trajectory(np.load(example / "poses.npy"), np.load(example / "intrinsics.npy"),
                            13 + 16 * (CAPTURE_CHUNK + 1), device=default_device())
    pipe = load_pipeline(args, 1, 0)
    dit, device = pipe.dit, pipe.device
    original = dit._block_run

    def capture(block, value, cache_k, cache_v, text_k, text_v, cam, qkv_weight, qkv_bias, gathered, captured_k,
                captured_v, read_frames, capture_flag):
        ck, cv = cache_k.clone(), cache_v.clone()  # the pipeline's own cache stays untouched
        output = original(block, value, ck, cv, text_k, text_v, cam, qkv_weight, qkv_bias, gathered,
                          captured_k.clone(), captured_v.clone(), read_frames, capture_flag)
        end = read_frames * lingbot_dit.FRAME_TOKENS  # cache rows this pass reads
        samples = []  # the real block (eager PyTorch under BF16 autocast, the demo's path) on the same inputs
        for i in range(13):
            synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device.type, dtype=torch.bfloat16):
                original(block, value, ck, cv, text_k, text_v, cam, qkv_weight, qkv_bias, gathered,
                         captured_k, captured_v, read_frames, capture_flag)
            synchronize(device)
            if i >= 3:
                samples.append((time.perf_counter() - start) * 1000)
        attn, cross = block.self_attn, block.cross_attn
        tensors = {
            "value": value, "cache_k": cache_k[:, :end], "cache_v": cache_v[:, :end], "text_k": text_k,
            "text_v": text_v, "cam": cam, "qkv_weight": qkv_weight, "qkv_bias": qkv_bias, "rope": dit.rope,
            "time_e0": dit.time_e0, "modulation": block.modulation, "norm_q": attn.norm_q.weight,
            "norm_k": attn.norm_k.weight, "o_weight": attn.o.weight, "o_bias": attn.o.bias,
            "norm3_weight": block.norm3.weight, "norm3_bias": block.norm3.bias,
            "cross_q_weight": cross.q.weight, "cross_q_bias": cross.q.bias, "cross_norm_q": cross.norm_q.weight,
            "cross_o_weight": cross.o.weight, "cross_o_bias": cross.o.bias,
            "ffn_up_weight": block.ffn[0].weight, "ffn_up_bias": block.ffn[0].bias,
            "ffn_down_weight": block.ffn[2].weight, "ffn_down_bias": block.ffn[2].bias,
            "output": output, "cache_k_after": ck[:, :end],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().contiguous().cpu() for k, v in tensors.items()},
                  str(args.out.with_suffix(".safetensors")))
        meta = {"size": f"{conditioning.WIDTH}x{conditioning.HEIGHT}", "read_frames": read_frames, "end": end,
                "tokens": value.shape[1], "heads": attn.num_heads, "eps": block.norm1.eps,
                "attn_eps": attn.norm_q.eps, "cross_eps": cross.norm_q.eps, "norm3_eps": block.norm3.eps,
                "torch_block_ms": statistics.median(samples), "torch_samples_ms": samples,
                "dtypes": {k: str(v.dtype) for k, v in tensors.items()}}
        args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"captured block 0 at {meta['size']}, read_frames {read_frames}: PyTorch block median "
              f"{meta['torch_block_ms']:.1f} ms", flush=True)
        raise Captured

    with torch.inference_mode():
        pipe.start(Image.open(example / "image.jpg").convert("RGB"), (example / "prompt.txt").read_text().strip(),
                   seed=0, latents=trajectory.latents)
        for chunk in range(trajectory.num_chunks):
            if chunk == CAPTURE_CHUNK:
                dit._block_run = capture
            try:
                pipe.step(trajectory.rays(chunk))
            except Captured:
                return
            print(f"chunk {chunk} done", flush=True)


if __name__ == "__main__":
    main()
