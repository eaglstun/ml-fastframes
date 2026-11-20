# FastFrames: Video Generation at Interactive Latency

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://apple-aiml-research.github.io/ml-fastframes/)
[![arXiv](https://img.shields.io/badge/arXiv-2611.23226-b31b1b.svg)](https://arxiv.org/abs/2611.23226)
[![FastFrames-S](https://img.shields.io/badge/%F0%9F%A4%97_Model-FastFrames--S-yellow)](https://huggingface.co/apple/FastFrames-S)
[![FastFrames-L](https://img.shields.io/badge/%F0%9F%A4%97_Model-FastFrames--L-yellow)](https://huggingface.co/apple/FastFrames-L)

> **This fork adds Apple Silicon (MPS) support for FastFrames-S text-to-video and the LingBot
> camera-control demo.** On an M4 Max, T2V takes ~45-50 s per 81-frame 480p video with ~18 GB peak
> memory, and the LingBot demo makes each 1 s turn in ~11-22 s. See
> [Apple Silicon](#apple-silicon-mps) and [MAC.md](MAC.md). Everything else is upstream
> [apple-aiml-research/ml-fastframes](https://github.com/apple-aiml-research/ml-fastframes) and
> still targets Linux + NVIDIA.

This repository contains the official implementation of *FastFrames: Video Generation at
Interactive Latency*. We release [FastFrames-S](https://huggingface.co/apple/FastFrames-S)
(1.3B) and [FastFrames-L](https://huggingface.co/apple/FastFrames-L) (14B). Both are
autoregressive, two-step video diffusion models that generate 480p video at interactive
latency. We also release an optimized version of LingBot 1.3B that is 16x faster than the
official model and allows fluid camera-conditioned control.

- [Latency profiling](#latency-profiling)
- [Interactive demo with LingBot](#interactive-demo-with-lingbot)
- [T2V generation](#t2v-generation)
- [VBench evaluation](#vbench-evaluation)
- [Training](#training)
- [Citation](#citation)

[![FastFrames demo: LingBot camera control with and without FastFrames optimizations](docs/assets/fastframes_demo_preview.gif)](docs/assets/fastframes_demo.mp4)

## Installation

Linux x86-64 with an NVIDIA GPU and a CUDA 12.8-compatible driver. From the repository root:

```bash
uv sync
```

Download FastFrames and third-party weights using the [checkpoint guide](docs/installation.md#checkpoints).

### Apple Silicon (MPS)

FastFrames-S T2V inference runs on Apple Silicon Macs through PyTorch MPS. Don't run `uv sync`
on a Mac, because `pyproject.toml` pins the CUDA wheels. Use a separate venv instead (needs
`ffmpeg` with libx264 on `PATH`):

```bash
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python "torch==2.9.1" "torchvision==0.24.1" "transformers>=5.10" \
    safetensors einops ftfy regex pyyaml numpy huggingface_hub
HF=.venv/bin/hf
$HF download apple/FastFrames-S --include 'fastframes-s/*' --local-dir checkpoints
$HF download zhuhz22/Causal-Forcing causal-forcing++/framewise-2step.pt --local-dir checkpoints
$HF download Wan-AI/Wan2.1-T2V-1.3B --exclude 'diffusion_pytorch_model*' --local-dir checkpoints/Wan2.1-T2V-1.3B

.venv/bin/python scripts/generate.py --model ff-s --checkpoint-dir checkpoints \
    --prompt "A corgi surfing a wave at sunset" --seed 0 --output outputs/corgi.mp4
```

| M4 Max, 64 GB             |                               |
| ------------------------- | ----------------------------- |
| model load                | ~5 s                          |
| rollout, 21 latent frames | ~45-50 s (~2.2 s per latent frame) |
| peak memory               | ~18 GB                        |

This is batch generation, not interactive latency: the rollout is compute-bound, about 100×
below the B200 numbers above.

The [LingBot interactive demo](#interactive-demo-with-lingbot) also runs on a Mac (single device,
eager): `.venv/bin/python apps/lingbot/demo.py --gpus 1 --lingbot-root upstream/lingbot-world-v2`.
Once warm, each 1 s turn (16 frames) takes ~11-22 s on the M4 Max, so you explore shot by shot.
Setup, measurements and the changes are in
[MAC.md](MAC.md#lingbot-camera-control-interactive-demo).

The single-device [latency benchmarks](MAC.md#latency-benchmarks) run too: one FastFrames-S step
(4 frames) takes ~4.0 s on the M4 Max (35.4 ms on a B200), and one LingBot chunk ~27 s.

Training, multi-GPU runs, the baseline latency rows, and FastFrames-L (14B, doesn't fit in 64 GB
without offloading) are not ported. [MAC.md](MAC.md) lists every change and why, including the
Metal flash-attention kernel used for attention on MPS.

## Latency Profiling

```bash
uv run python evaluation/latency/benchmark.py ff-s
uv run python evaluation/latency/benchmark.py ff-l --gpus 8
```

Example component timings for **FastFrames-S, 1 GPU, 480p**, with a full KV cache.

| Component | Calls | Latency per call | Total latency |
|---|---:|---:|---:|
| DiT evaluation, t=1000 | 1 | 16.1 ms | 16.1 ms |
| DiT evaluation, t=833 | 1 | 16.2 ms | 16.2 ms |
| KV refinement | 1 | 2.2 ms | 2.2 ms |
| Pixel decoder | 1 | 0.9 ms | 0.9 ms |
| **Total** | | | **35.4 ms** (paper: 36 ms) |

See the [profiling guide](docs/inference.md#profiling) for details and baselines.

## Interactive Demo with LingBot

Complete the [demo setup](docs/lingbot.md#interactive-demo-13b), then launch one of:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra lingbot --extra demo python apps/lingbot/demo.py --gpus 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --extra lingbot --extra demo python apps/lingbot/demo.py --gpus 8
```

and open `http://localhost:8000`.

## T2V Generation

```bash
uv run python scripts/generate.py --model ff-s --checkpoint-dir checkpoints \
    --prompt "A corgi surfing a wave at sunset" --seed 0 --output outputs/corgi.mp4
```

Outputs 832 × 480 video, 81 frames at 16 fps. Use `--model ff-l` for FastFrames-L.

To generate the videos on our project page (20+ videos):

```bash
uv run python scripts/generate_project_page.py --model ff-s --output-dir outputs/project_page/ff-s
uv run python scripts/generate_project_page.py --model ff-l --output-dir outputs/project_page/ff-l
```

See the [generation guide](docs/inference.md#generation) for details.

## VBench Evaluation

Follow the [evaluation guide](docs/inference.md#evaluation) to reproduce Table 1's
Quality / Semantic / Total scores.

## Training

### [Causal Forcing++ -> FastFrames-S](docs/training.md#causal-forcing---fastframes-s)

FastFrames-S keeps the two-step frame-wise Causal Forcing++ generator frozen and trains the
networks around it.

### [Wan2.1-14B -> FastFrames-L](docs/training.md#wan21-14b---fastframes-l)

We first distill Wan2.1-14B into a two-step causal model following the Causal Forcing++ recipe,
then train its pixel decoder and KV refinement network as for FastFrames-S. We release all
checkpoints: 1) a causal teacher trained on latents generated by Wan2.1-14B; 2) the
consistency-distilled model; 3) the DMD-distilled model; 4) the final generator, pixel decoder
and KV refinement network.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{chou2026fastframes,
  title   = {FastFrames: Video Generation at Interactive Latency},
  author  = {Chou, Gene and Yang, Guandao and Luan, Fujun and Mescheder, Lars and
             Dong, Wei and Richter, Stephan R. and Koltun, Vladlen},
  journal = {arXiv preprint arXiv:2611.23226},
  year    = {2026}
}
```

## License and Acknowledgments

LingBot-World v2 code and weights follow their upstream [CC BY-NC-SA 4.0 license](LICENSE_LINGBOT).
All other FastFrames code is licensed under the [Apple Sample Code License](LICENSE).
Also see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for third-party code.

We thank the authors of [Wan2.1](https://github.com/Wan-Video/Wan2.1),
[Self-Forcing](https://github.com/guandeh17/Self-Forcing), and
[Causal Forcing++](https://github.com/thu-ml/Causal-Forcing) for releasing their code, on which the
FastFrames code and models build.
