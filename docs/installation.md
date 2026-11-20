# Installation and checkpoints

> **Apple Silicon Mac:** don't run `uv sync` (it installs CUDA wheels). Follow the setup in
> [MAC.md](../MAC.md) instead; it covers FastFrames-S, the LingBot demo and the latency benchmarks.

Use Linux x86-64 with an NVIDIA GPU and a CUDA 12.8-compatible driver. Run all
commands from the repository root. `uv` installs Python 3.10 and the pinned
PyTorch 2.9.1 / CUDA 12.8 environment:

```bash
uv sync
```

Complete the setup for your task below, then follow [inference](inference.md),
the [LingBot application](lingbot.md), or [training](training.md).

## Checkpoints

Download the FastFrames variant you want to use and its matching Wan snapshot. The model
cards of [FastFrames-S](https://huggingface.co/apple/FastFrames-S) and
[FastFrames-L](https://huggingface.co/apple/FastFrames-L) describe each file:

```bash
# FastFrames-S: its generator is the Causal Forcing++ checkpoint (Apache-2.0)
uv run hf download apple/FastFrames-S --include 'fastframes-s/*' --local-dir checkpoints
uv run hf download zhuhz22/Causal-Forcing causal-forcing++/framewise-2step.pt --local-dir checkpoints
uv run hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir checkpoints/Wan2.1-T2V-1.3B

# FastFrames-L
uv run hf download apple/FastFrames-L --include 'fastframes-l/*' --local-dir checkpoints
uv run hf download Wan-AI/Wan2.1-T2V-14B --local-dir checkpoints/Wan2.1-T2V-14B
```

`checkpoints/fastframes-s/` contains `kv_refiner.safetensors` and
`pixel_decoder.safetensors`; FastFrames-S runs them with the frozen two-step frame-wise
Causal Forcing++ generator `checkpoints/causal-forcing++/framewise-2step.pt` (its EMA
weights in BF16; SHA-256 in [Baselines](#baselines)). `checkpoints/fastframes-l/` contains
`dit.safetensors`, `kv_refiner.safetensors`, and `pixel_decoder.safetensors`. T2V generation reads
the text encoder and tokenizer from Wan; add `--exclude 'diffusion_pytorch_model*'`
to the Wan download to skip its base DiT for generation only. Training uses the
full snapshot. Optional training intermediates are described [below](#training-inputs).

## LingBot

LingBot-World v2 code and weights follow their upstream
[CC BY-NC-SA 4.0 license](../LICENSE_LINGBOT).

Clone the pinned upstream code and install the application dependencies:

```bash
mkdir -p upstream
git clone https://github.com/robbyant/lingbot-world-v2 upstream/lingbot-world-v2
git -C upstream/lingbot-world-v2 checkout 1895d300d8ac936401689b26389f51cbd36530eb
export LINGBOT_WORLD_ROOT=$PWD/upstream/lingbot-world-v2
uv sync --extra lingbot --extra demo
```

For the 1.3B demo, download the DiT, the FastFrames networks from
[FastFrames-S](https://huggingface.co/apple/FastFrames-S), and the shared text
encoder/tokenizer/VAE. The shared assets come from the upstream 14B repository:

```bash
uv run hf download apple/FastFrames-S --include 'lingbot-1.3b/*' --local-dir checkpoints
uv run hf download robbyant/lingbot-world-v2-1.3b-causal-fast \
    --local-dir checkpoints/lingbot-world-v2-1.3b-causal-fast/transformers
uv run hf download robbyant/lingbot-world-v2-14b-causal-fast \
    --include 'models_t5_umt5-xxl-enc-bf16.pth' --include 'Wan2.1_VAE.pth' --include 'google/umt5-xxl/*' \
    --local-dir checkpoints/lingbot-world-v2-14b-causal-fast
```

Continue with the [LingBot application guide](lingbot.md).

## Baselines

Self-Forcing, Causal Forcing++ (CF++), and CausVid use their upstream code and
weights. Download the full [Wan2.1-T2V-1.3B snapshot](#checkpoints), then:

```bash
uv sync --extra baselines
mkdir -p upstream
git clone https://github.com/guandeh17/Self-Forcing upstream/Self-Forcing
git -C upstream/Self-Forcing checkout 33593df3e81fa3ec10239271dd2c100facac6de1
git clone https://github.com/thu-ml/Causal-Forcing upstream/Causal-Forcing
git -C upstream/Causal-Forcing checkout 730d305eea6eca5f194f42a715dd58eec606c8c4
git clone https://github.com/tianweiy/CausVid upstream/CausVid
git -C upstream/CausVid checkout adb6a5ecd07666b4d0290042915c8406e6d5ce22

for repo in Self-Forcing Causal-Forcing CausVid; do
  mkdir -p upstream/$repo/wan_models
  ln -s "$PWD/checkpoints/Wan2.1-T2V-1.3B" upstream/$repo/wan_models/Wan2.1-T2V-1.3B
done

uv run hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir checkpoints/self-forcing
uv run hf download zhuhz22/Causal-Forcing causal-forcing++/framewise-2step.pt --local-dir checkpoints
uv run hf download tianweiy/CausVid autoregressive_checkpoint/model.pt \
    --revision b545eb2728fc9d1515023a270b847f7b24b3aa89 --local-dir checkpoints/causvid
```

Checkpoint hashes, verifiable with `sha256sum`:

| File under `checkpoints/` | SHA-256 |
|---|---|
| `self-forcing/checkpoints/self_forcing_dmd.pt` | `a0413986d9734e02c09504e1520f5697ba6df731bb2f0f35577485e9cc8f56a3` |
| `causal-forcing++/framewise-2step.pt` | `8f899a016dffb9539788fac82c6200e43ef6e23b6e61ec6b885bb507a760ce42` |
| `causvid/autoregressive_checkpoint/model.pt` | `8b75a848a525a6ab0b8cefa7f2a3997aefb581eb5824ad60bafb5337e81e521c` |

The wrappers use cuDNN SDPA; no flash-attn installation is needed.
See [inference](inference.md) for baseline profiling and evaluation commands.

## VBench

Scoring uses a separate environment for `transformers==4.33.2` and detectron2;
generation uses the main FastFrames environment. GNU `patch` is required.

```bash
uv run bash evaluation/vbench/setup_env.sh
export VBENCH_PYTHON=$PWD/evaluation/vbench/.venv-vbench/bin/python
export VBENCH_CACHE_DIR=$PWD/checkpoints/vbench_cache
uv run bash evaluation/vbench/download_models.sh          # ~9.4 GB
uv run bash evaluation/vbench/download_models.sh --verify
```

`setup_env.sh` builds and verifies the pinned scorer. Pass a directory argument
to choose another environment location, then update `VBENCH_PYTHON`. The model
download script verifies every metric weight by hash. Keep the two exported
variables set when running [evaluation](inference.md#evaluation).

## Training inputs

Download Self-Forcing's extended VidProM list (248,221 prompts, 145,875,068 bytes):

```bash
uv run hf download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir data/prompts
sha256sum data/prompts/vidprom_filtered_extended.txt
# 7896742f468bc8aef9e4547424d1ce0a951acdb2a82233790155401a99bf5aa5
```

The included [demos.txt](../data/prompts/demos.txt) has 109 prompts: the first 100
come from [Causal Forcing](https://github.com/thu-ml/Causal-Forcing)'s
`prompts/demos.txt` (Apache-2.0), followed by nine added for this work. Training
evaluation uses its first 32 lines. ODE generation accepts a one-prompt-per-line
file through `data.prompts`.

For adversarial decoder training, request access to the gated DINOv3 model and download it:

```bash
uv run hf download facebook/dinov3-vits16-pretrain-lvd1689m \
    --local-dir checkpoints/dinov3-vits16-pretrain-lvd1689m
```

Set `real_data.root` to a folder of H.264 `.mp4`, `.mov`, or `.m4v` clips with at
least 81 frames each. Training center-crops and resizes them to 832 × 480.

To start from an [intermediate checkpoint](training.md#intermediate-checkpoints),
download only the required subfolder of
[FastFrames-L](https://huggingface.co/apple/FastFrames-L). For example:

```bash
uv run hf download apple/FastFrames-L --include 'fastframes-l-intermediate/dmd/*' --local-dir checkpoints
```

See [training](training.md) for stage inputs, checkpoint sizes and launch commands.
Add `--extra wandb` to `uv run` for training logs, or `--extra dev` for CPU tests.
