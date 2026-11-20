# Training

Training uses NVIDIA B200 GPUs (it isn't ported to Apple Silicon). Run from the repository root after
[installation](installation.md), with a config from [`configs/`](../configs/):

```bash
uv run torchrun --nproc_per_node 8 scripts/train.py --config configs/ffs_decoder_gan.yaml \
    output_dir=outputs/my_run
```

The tables below specify the recipe's GPU count; for multiple nodes, add the usual
`torchrun` rendezvous options (`--nnodes`, `--node_rank`, `--rdzv_endpoint`). Batch
sizes are per GPU. Override existing config keys with `a.b=value`; use `+a.b=value`
to add a key. DMD counts `train.max_critic_updates`; other trainers use
`train.max_updates`. ODE generation takes `data.prompts=<file>` and has no `train`
or `wandb` section.

For Weights & Biases, add `--extra wandb` to `uv run` and set `wandb.project`.

## Inputs

Follow the [checkpoint setup](installation.md#checkpoints) and
[training input downloads](installation.md#training-inputs). FastFrames-S uses the
frozen Causal Forcing++ generator `checkpoints/causal-forcing++/framewise-2step.pt`;
the FF-S reconstruction and KV configs select its `generator_ema` weights.

## Causal Forcing++ -> FastFrames-S

Train the pixel decoder and KV refinement network around the frozen generator.
Export paths below are relative to the preceding stage's output directory.

| Stage | Config | Input | Recipe |
|---|---|---|---|
| 1. Decoder reconstruction (Eq. 4–7) | [ffs_decoder_recon.yaml](../configs/ffs_decoder_recon.yaml) | FF-S generator | 32 GPUs, 600,000 updates |
| 2. Decoder adversarial (Eq. 2–3) | [ffs_decoder_gan.yaml](../configs/ffs_decoder_gan.yaml) | stage 1 `exports/step_00600000/decoder_ema.safetensors` | 8 GPUs, 5,000 rollouts per GPU |
| 3. KV refinement | [ffs_kv_refinement.yaml](../configs/ffs_kv_refinement.yaml) | FF-S generator | 32 GPUs, 142,500 updates |

KV refinement trains from scratch in BF16 with FP32 master weights and AdamW moments.

## Wan2.1-14B -> FastFrames-L

| Stage | Config | Input | Recipe |
|---|---|---|---|
| 0. ODE trajectories | [ffl_ode_data.yaml](../configs/ffl_ode_data.yaml) | one-prompt-per-line file (`data.prompts`) | 48 Euler steps, CFG 5 |
| 1. Frame-causal teacher | [ffl_causal_teacher.yaml](../configs/ffl_causal_teacher.yaml) | stage 0 corpus | 256 GPUs; select update 1700 |
| 2. Consistency distillation | [ffl_consistency.yaml](../configs/ffl_consistency.yaml) | stage 1 `exports/step_00001700/model.safetensors` | 128 GPUs; select update 3000 |
| 3. Self-forcing DMD | [ffl_dmd.yaml](../configs/ffl_dmd.yaml) | stage 2 `exports/step_00003000/model.safetensors` | 128 GPUs; EMA at critic update 1500 |
| 4. Decoder reconstruction | [ffl_decoder_recon.yaml](../configs/ffl_decoder_recon.yaml) | stage 3 EMA; FF-S stage 1 `exports/step_00050000/decoder_ema.safetensors` | 64 GPUs, 500,000 updates |
| 5. Decoder adversarial | [ffl_decoder_gan.yaml](../configs/ffl_decoder_gan.yaml) | stage 4 `exports/step_00500000/decoder_ema.safetensors` | 8 GPUs, 5,000 rollouts per GPU |
| 6. KV refinement | [ffl_kv_refinement.yaml](../configs/ffl_kv_refinement.yaml) | stage 3 `exports/step_00001500/ema.safetensors` | 64 GPUs, 142,500 updates |

### Intermediate checkpoints

Download an intermediate to start directly from a later stage. The three folders
under `fastframes-l-intermediate/` in [apple/FastFrames-L](https://huggingface.co/apple/FastFrames-L)
total 143 GB:

| Folder | Checkpoint | dtype | Size | Config override |
|---|---|---|---|---|
| `causal_teacher/` | raw weights, update 1700 | FP32 | 57.2 GB | stage 2: `model.init` |
| `consistency/` | raw weights, update 3000 | BF16 | 28.6 GB | stage 3: `model.init` |
| `dmd/` | EMA, critic update 1500 | FP32 | 57.2 GB | stages 4 and 6: `model.generator` |

After [downloading an intermediate](installation.md#training-inputs), pass its
sharded safetensors index as the checkpoint path:

```bash
uv run torchrun --nproc_per_node 8 scripts/train.py --config configs/ffl_kv_refinement.yaml \
    model.generator=checkpoints/fastframes-l-intermediate/dmd/model.safetensors.index.json
```

Use the GPU count in the recipe table for a full training run.

## Pixel decoder

The decoder learns from frozen-generator rollouts, using DiT features from the
final denoising step and the Wan VAE's decoded frames as targets. Each rollout supplies
`train.updates_per_rollout` optimizer updates. FastFrames-L projects its 5120-wide
features to 1536 channels and initializes from a FastFrames-S reconstruction export.

The adversarial phase uses [DINOv3 and real-video inputs](installation.md#training-inputs).

## Outputs and resume

Trainers write resumable `step_<N>/` directories and `latest.json` under
`output_dir`. Restart the same command at the same GPU count to resume.
`train.keep_checkpoints` controls retained resumable checkpoints.

Inference weights go in `exports/step_<N>/`:

| Trainer | Export files |
|---|---|
| DiT | `model.safetensors`, `ema.safetensors` |
| Decoder | `decoder.safetensors`; reconstruction also writes `decoder_ema.safetensors` |
| KV refinement | `refiner.safetensors` |

Exports accumulate independently of checkpoint retention. Budget disk space for
them and retain the intermediates needed by later stages; each causal-teacher
export contains about 86 GB of weights. ODE generation writes resumable corpus
shards instead of training checkpoints.
