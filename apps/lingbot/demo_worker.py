#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Resident, single-node LingBot workers. Idle ranks wait on CPU queues, not NCCL."""
from __future__ import annotations

import gc
import hashlib
import io
import json
import multiprocessing as mp
import os
import queue
import tempfile
import time
import traceback
from pathlib import Path

from PIL import Image

ACTIONS = {"idle", "forward", "backward", "left", "right", "yaw_left", "yaw_right"}
LATENTS = 240


def _worker(rank, config, commands, replies, rendezvous):
    # Import CUDA only in spawned workers; each rank sees the same device namespace.
    import torch
    import torch.distributed as dist

    from apps.lingbot import conditioning
    from apps.lingbot.conditioning import KeyboardCamera, LiveCamera
    from apps.lingbot.pipeline import FastFramesLingBot, configure_backends
    from fastframes.device import default_device, release_mps_cache, synchronize

    pipe = None
    device = default_device()
    cuda = device.type == "cuda"

    world = config["gpus"]
    fingerprint = None
    try:
        if world > 1 and not cuda:
            raise RuntimeError("multi-GPU LingBot needs CUDA")
        if cuda:
            torch.cuda.set_device(rank)
        configure_backends()
        conditioning.set_size(*config.get("size", (conditioning.WIDTH, conditioning.HEIGHT)))
        if world > 1:
            from datetime import timedelta

            dist.init_process_group("nccl", init_method=rendezvous, world_size=world, rank=rank,
                                    timeout=timedelta(minutes=10), device_id=torch.device("cuda", rank))
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        keys = camera = None
        # MPS/CPU: reply with the frames before the chunk's KV-cache write, then write it and encode
        # the next chunk's VAE condition between turns (LINGBOT_DEMO_PREFETCH=0 disables both)
        prefetch = not cuda and os.environ.get("LINGBOT_DEMO_PREFETCH", "1") != "0"
        # MPS/CPU: send a rough still of the chunk after the first two of its four denoising passes
        # (~30% and ~60% of the turn) while it finishes (LINGBOT_DEMO_PREVIEW=0 disables)
        previews = not cuda and world == 1 and os.environ.get("LINGBOT_DEMO_PREVIEW", "1") != "0"
        while True:
            request = commands.get()  # No GPU collective is pending during user idle.
            op = request["op"]
            result = {}
            if op == "shutdown":
                break
            if op == "start":
                size = tuple(request.get("size") or (conditioning.WIDTH, conditioning.HEIGHT))
                if pipe is not None and size != (conditioning.WIDTH, conditioning.HEIGHT):
                    # Another frame size: rebuild the size-dependent buffers around the loaded weights.
                    # Free the old ones first (~6 GB of caches at 832x464) so both never coexist.
                    if cuda:
                        raise ValueError("switching the frame size is only supported off CUDA")
                    parts = pipe.parts()
                    pipe.close()
                    pipe = None
                    gc.collect()
                    release_mps_cache(device)
                    conditioning.set_size(*size)
                    pipe = FastFramesLingBot(**parts)
                    fingerprint = None
                if pipe is None:
                    conditioning.set_size(*size)
                    root = Path(config["checkpoint_dir"])
                    pipe = FastFramesLingBot.from_pretrained(
                        "lingbot-1.3b", lingbot_root=config["lingbot_root"],
                        lingbot_checkpoint=config.get("lingbot_checkpoint") or
                        root / "lingbot-world-v2-1.3b-causal-fast",
                        assets_dir=config.get("assets_dir") or root / "lingbot-world-v2-14b-causal-fast",
                        checkpoint_dir=root / "lingbot-1.3b", world=world, rank=rank,
                        compile=cuda, graphs=cuda)  # torch.compile/CUDA graphs are CUDA-only here
                    keys, camera = KeyboardCamera(), LiveCamera(device=device)
                next_fingerprint = (hashlib.sha256(request["image"]).hexdigest(), request["prompt"], size)
                if fingerprint == next_fingerprint:
                    pipe.restart(seed=request["seed"])
                else:
                    with Image.open(io.BytesIO(request["image"])) as image:
                        pipe.start(image.convert("RGB"), request["prompt"], seed=request["seed"],
                                   latents=LATENTS, noise="chunk")
                if fingerprint is None and cuda:
                    # Eager off CUDA: nothing to compile or capture, so no warm-up chunks.
                    # Visit every cache length and both cold/steady decoder shapes twice.
                    for _ in range(2):
                        keys.reset()
                        camera.reset()
                        for _ in range(8):
                            pipe.step(camera.rays(keys.chunk("idle")))
                        pipe.restart(seed=request["seed"])
                fingerprint = next_fingerprint
                keys.reset()
                camera.reset()
            elif op == "step":
                action = request["action"]
                if action not in ACTIONS:
                    raise ValueError(f"Unknown camera action: {action}")
                synchronize(device)
                start = time.perf_counter()
                def send_preview(index, frames):
                    frame = frames[frames.shape[0] // 2].cpu().numpy()
                    buffer = io.BytesIO()
                    Image.fromarray(frame).save(buffer, format="JPEG", quality=85)
                    replies.put({"rank": rank, "preview": {"jpeg": buffer.getvalue(), "pass": index + 1,
                                                           "passes": len(pipe.timesteps),
                                                           "chunk_index": pipe.chunk}})

                frames = pipe.step(camera.rays(keys.chunk(action)), defer_cache=prefetch,
                                   preview=send_preview if previews else None)
                synchronize(device)
                model_ms = (time.perf_counter() - start) * 1000
                if world > 1:
                    timing = torch.tensor(model_ms, device="cuda", dtype=torch.float64)
                    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                    model_ms = timing.item()
                if rank == 0:
                    encoded = []
                    for frame in frames.cpu().numpy():
                        buffer = io.BytesIO()
                        Image.fromarray(frame).save(buffer, format="JPEG", quality=90)
                        encoded.append(buffer.getvalue())
                    result = {"frames": encoded, "chunk_index": pipe.chunk - 1, "model_ms": model_ms,
                              "terminal": pipe.chunk >= pipe.num_chunks, "width": conditioning.WIDTH,
                              "height": conditioning.HEIGHT}
            elif op == "stop":
                # No generation is in flight: the parent serializes all requests.
                # Keep weights, compiled graphs and the last conditioning for restart.
                pass
            else:
                raise ValueError(f"Unknown worker operation: {op}")
            synchronize(device)
            if cuda:
                memory = {"rank": rank, "gpu": torch.cuda.get_device_name(),
                          "allocated_bytes": torch.cuda.memory_allocated(),
                          "reserved_bytes": torch.cuda.memory_reserved(),
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            else:  # MPS has no peak counters: report the current values
                allocated = torch.mps.current_allocated_memory() if device.type == "mps" else 0
                reserved = torch.mps.driver_allocated_memory() if device.type == "mps" else 0
                memory = {"rank": rank, "gpu": f"Apple Silicon ({device.type})", "allocated_bytes": allocated,
                          "reserved_bytes": reserved, "peak_allocated_bytes": allocated,
                          "peak_reserved_bytes": reserved}
            replies.put({"rank": rank, "result": result, "memory": memory})
            if prefetch and op in ("start", "step"):
                # Neither the cache write (KV refinement, ~0.6 s on MPS) nor the next chunk's VAE
                # condition (~4 s) depends on the next action: run them while the player decides.
                pipe.finish_chunk()
                pipe.prefetch_condition()
            if cuda:
                torch.cuda.reset_peak_memory_stats()
    except BaseException:
        replies.put({"rank": rank, "error": traceback.format_exc()})
    finally:
        if pipe is not None:
            pipe.close()
        if dist.is_initialized():
            dist.destroy_process_group()


class WorkerPool:
    """Call from one executor thread. All commands are broadcast through CPU queues."""

    def __init__(self, config):
        self.config = config
        self.processes = []
        self.commands = []
        self.replies = None
        self.directory = None
        self.ready = False
        self.records = []

    def _launch(self):
        ctx = mp.get_context("spawn")
        self.directory = tempfile.TemporaryDirectory(prefix="lingbot-workers-")
        rendezvous = Path(self.directory.name, "rendezvous").as_uri()
        self.replies = ctx.Queue()
        for rank in range(self.config["gpus"]):
            commands = ctx.Queue()
            process = ctx.Process(target=_worker, args=(rank, self.config, commands, self.replies, rendezvous))
            process.start()
            self.commands.append(commands)
            self.processes.append(process)

    def call(self, op, *, on_preview=None, **payload):
        """Run ``op`` on every rank; ``on_preview(preview)`` (this thread) gets rank 0's step previews."""
        if not self.processes:
            if op == "stop":
                return {}
            self._launch()
        started = time.perf_counter()
        for commands in self.commands:
            commands.put({"op": op, **payload})
        responses = {}
        deadline = time.monotonic() + (1800 if op == "start" else 180)
        try:
            while len(responses) < len(self.processes):
                try:
                    response = self.replies.get(timeout=1)
                except queue.Empty:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"LingBot {op} timed out")
                    if any(process.exitcode is not None for process in self.processes):
                        raise RuntimeError("A LingBot worker exited; check the server log")
                    continue
                if "error" in response:
                    raise RuntimeError(response["error"])
                if "preview" in response:
                    if on_preview is not None:
                        on_preview(response["preview"])
                    continue
                responses[response["rank"]] = response
        except BaseException:
            self.close(force=True)
            raise
        elapsed = (time.perf_counter() - started) * 1000
        if self.config.get("profile_vram"):
            self.records.append({"operation": op, "wall_ms": elapsed,
                                 "gpus": [responses[rank]["memory"] for rank in sorted(responses)]})
        result = responses[0]["result"]
        result["worker_ms"] = elapsed
        self.ready = True
        return result

    def save_memory_report(self, path):
        report = {"model": "lingbot-1.3b", "gpus": self.config["gpus"], "latents": LATENTS,
                  "records": self.records}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(report, indent=2) + "\n")

    def close(self, force=False):
        if not force:
            for commands in self.commands:
                commands.put({"op": "shutdown"})
            for process in self.processes:
                process.join(timeout=5)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for commands in self.commands:
            commands.cancel_join_thread()
            commands.close()
        if self.replies is not None:
            self.replies.close()
        self.processes.clear()
        self.commands.clear()
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None
        self.ready = False
