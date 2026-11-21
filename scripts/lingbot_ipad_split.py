#!/usr/bin/env python3
"""One FastFrames-LingBot DiT denoising pass split between this Mac (PyTorch) and an iPad (MLX-Swift).

The chunk's tokens are split: the Mac keeps [0, n), the iPad runs [n, tokens) through all 30 blocks
(SplitStage.swift in the exo-ipad-spike app, `SPIKE_MODE=split`). In each block both compute their
tokens' keys and values, swap them, write both into their own copy of the KV cache and attend over
all of it; the rest of the block is per token. Runs the real pipeline to a full cache (example 00,
chunk 5), then for the first pass of that chunk:

1. the Mac alone, through the pipeline's own block (the reference, and its time),
2. this script's split block with every token on the Mac (must match 1: checks the reimplementation),
3. the split with the iPad, compared with 1 and timed.

    # on the iPad: SPIKE_MODE=split (exo-ipad-spike: mac/run_split_stage.sh)
    .venv/bin/python scripts/lingbot_ipad_split.py --model lingbot-1.3b \\
        --lingbot-root upstream/lingbot-world-v2 [--ipad 169.254.154.199] [--ipad-share 0.48]
"""
import argparse
import json
import queue
import socket
import struct
import sys
import threading
import time
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])  # repository root
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from apps.lingbot import dit as lingbot_dit  # noqa: E402
from apps.lingbot.conditioning import Trajectory  # noqa: E402
from apps.lingbot.dit import attention, layer_norm, rms_norm, rope_real  # noqa: E402
from apps.lingbot.generate import add_model_arguments, apply_size, load_pipeline  # noqa: E402
from apps.lingbot.upstream import import_upstream  # noqa: E402
from fastframes.device import default_device, limit_mps_memory, synchronize  # noqa: E402

CHUNK = 5  # the cache holds its full 18 frames from chunk 4 on
DTYPES = {torch.bfloat16: "bfloat16", torch.float32: "float32", torch.float16: "float16"}
TORCH_DTYPES = {name: dtype for dtype, name in DTYPES.items()}


class Done(Exception):
    """Raised once the experiment ran, to stop the pipeline."""


# ------------------------------------------------------------------ frames (SplitStage.swift's format)

def tensor_bytes(tensor):
    tensor = tensor.detach().contiguous().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    return tensor.numpy().tobytes()


def encode(op, *, layer=0, meta=None, arrays=()):
    specs = [{"name": name, "shape": list(t.shape), "dtype": DTYPES[t.dtype]} for name, t in arrays]
    header = json.dumps({"op": op, "layer": layer, "meta": meta or {}, "arrays": specs}).encode()
    return b"".join([struct.pack("<I", len(header)), header] + [tensor_bytes(t) for _, t in arrays])


def read_exactly(sock, count):
    buffer = bytearray(count)
    view, received = memoryview(buffer), 0
    while received < count:
        n = sock.recv_into(view[received:], count - received)
        if n == 0:
            raise ConnectionError("the iPad closed the connection")
        received += n
    return buffer


def read(sock, device):
    (length,) = struct.unpack("<I", read_exactly(sock, 4))
    header = json.loads(read_exactly(sock, length))
    arrays = {}
    for spec in header["arrays"]:
        dtype = TORCH_DTYPES[spec["dtype"]]
        count = int(np.prod(spec["shape"]))
        raw = read_exactly(sock, count * (4 if dtype == torch.float32 else 2))
        arrays[spec["name"]] = torch.frombuffer(raw, dtype=dtype).view(spec["shape"]).to(device)
    return header, arrays


class Link:
    """The socket to the iPad; frames are written by a sender thread, so both sides can send at once."""

    def __init__(self, host, port, device):
        self.sock = socket.create_connection((host, port), timeout=600)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.device, self.outbox = device, queue.Queue()
        threading.Thread(target=self._send_loop, daemon=True).start()

    def _send_loop(self):
        while (data := self.outbox.get()) is not None:
            self.sock.sendall(data)

    def send(self, data):
        self.outbox.put(data)

    def read(self):
        return read(self.sock, self.device)

    def close(self):
        self.send(encode("bye"))
        self.outbox.put(None)


# ------------------------------------------------------------------ the block, split by tokens

def pre(block, value, qkv_weight, qkv_bias, rope, time_e0, heads):
    """LingBotDiT._block up to the keys: this share's modulation, query, (RoPE'd) keys and values."""
    tokens, dim = value.shape[1], value.shape[2]
    e = (block.modulation.unsqueeze(0) + time_e0).unbind(2)
    hidden = layer_norm(value, block.norm1).float() * (1 + e[1]) + e[0]
    a = block.self_attn
    query, key, val = F.linear(hidden, qkv_weight, qkv_bias).chunk(3, -1)
    query = rope_real(rms_norm(query, a.norm_q).view(1, tokens, heads, dim // heads), rope)
    key = rope_real(rms_norm(key, a.norm_k).view(1, tokens, heads, dim // heads), rope)
    return e, query, key, val.view(1, tokens, heads, dim // heads)


def post(block, value, e, query, cache_k, cache_v, end, text_k, text_v, cam):
    """LingBotDiT._block from attention on, for this share's tokens (cache rows already written)."""
    tokens, dim = value.shape[1], value.shape[2]
    a = block.self_attn
    attended = attention(query, cache_k[:, :end], cache_v[:, :end], 1)
    value = value + F.linear(attended.reshape(1, tokens, dim), a.o.weight, a.o.bias) * e[2]
    value = (1.0 + cam[0]) * value + cam[1]
    cross = block.cross_attn
    q = rms_norm(cross.q(layer_norm(value, block.norm3)), cross.norm_q).view(1, tokens, a.num_heads, dim // a.num_heads)
    value = value + cross.o(attention(q, text_k, text_v).reshape(1, tokens, dim))
    hidden = layer_norm(value, block.norm2).float() * (1 + e[4]) + e[3]
    return value + block.ffn(hidden) * e[5]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--ipad", default="169.254.154.199", help="the iPad's address on the USB link")
    parser.add_argument("--port", type=int, default=5570)
    parser.add_argument("--ipad-share", type=float, default=0.48, help="fraction of the chunk's tokens on the iPad")
    parser.add_argument("--passes", type=int, default=4)
    args = parser.parse_args()
    limit_mps_memory(0.75)
    apply_size(args, 1)
    wan = import_upstream(args.lingbot_root)
    example = Path(wan.__file__).resolve().parents[1] / "examples" / "00"
    trajectory = Trajectory(np.load(example / "poses.npy"), np.load(example / "intrinsics.npy"),
                            13 + 16 * (CHUNK + 1), device=default_device())
    pipe = load_pipeline(args, 1, 0)
    dit, device = pipe.dit, pipe.device
    block_run = dit._block_run

    def experiment(read_frames, capture):
        tokens, heads = lingbot_dit.TOKENS, dit.heads
        end = read_frames * lingbot_dit.FRAME_TOKENS
        rows = end - tokens  # the cache row of this chunk's first token
        n = tokens - round(tokens * args.ipad_share)  # the Mac's tokens: [0, n)
        value0 = dit._patch_run()
        blocks = dit.model.blocks

        def mac_alone():
            value = value0
            for i, block in enumerate(blocks):
                value = block_run(block, value, dit.cache_k[i], dit.cache_v[i], dit.text_k[i], dit.text_v[i],
                                  dit.camera_params[i], *dit.qkv[i], dit.gather[i], dit.result.key[i],
                                  dit.result.value[i], read_frames, False)
            return value

        def mac_share(share, exchange=None):
            """This script's split pass for tokens [0, share); `exchange(i, k, v)` returns the iPad's rows."""
            value = value0[:, :share]
            for i, block in enumerate(blocks):
                e, query, key, val = pre(block, value, *dit.qkv[i], dit.rope[:share], dit.time_e0, heads)
                if exchange is not None:
                    exchange(i, key, val)
                dit.cache_k[i][:, rows:rows + share] = key
                dit.cache_v[i][:, rows:rows + share] = val
                if exchange is not None:
                    their_k, their_v = exchange.receive(i)
                    dit.cache_k[i][:, rows + share:end] = their_k
                    dit.cache_v[i][:, rows + share:end] = their_v
                value = post(block, value, e, query, dit.cache_k[i], dit.cache_v[i], end, dit.text_k[i],
                             dit.text_v[i], dit.camera_params[i][:, :, :share])
            return value

        def timed(fn, repeats=2):
            samples, out = [], None
            for _ in range(repeats):
                synchronize(device)
                start = time.perf_counter()
                out = fn()
                synchronize(device)
                samples.append(time.perf_counter() - start)
            return out, min(samples)

        def compare(label, ours, reference):
            a, b = ours.float(), reference.float()
            rel = ((a - b).norm() / b.norm()).item()
            print(f"  {label:44s} relative L2 {rel:.2e}  max abs {(a - b).abs().max().item():.3g}", flush=True)

        print(f"pass of chunk {CHUNK} at {tokens} tokens x {end} cache rows, {len(blocks)} blocks", flush=True)
        reference, alone = timed(mac_alone)
        print(f"Mac alone (the pipeline's block): {alone:.2f} s per pass", flush=True)
        whole, ours = timed(lambda: mac_share(tokens))
        compare("this script's block, all tokens on the Mac", whole, reference)
        print(f"  ({ours:.2f} s per pass)", flush=True)

        link = Link(args.ipad, args.port, device)
        t0 = time.perf_counter()
        meta = {"tokens": tokens, "start": n, "end": end, "heads": heads, "layers": len(blocks),
                "eps": blocks[0].norm1.eps, "attn_eps": blocks[0].self_attn.norm_q.eps,
                "cross_eps": blocks[0].cross_attn.norm_q.eps, "norm3_eps": blocks[0].norm3.eps}
        link.send(encode("setup", meta=meta, arrays=[("rope", dit.rope[n:]), ("value", value0[:, n:])]))
        sent = 0
        for i, block in enumerate(blocks):
            attn, cross = block.self_attn, block.cross_attn
            bf16 = torch.bfloat16
            arrays = [
                ("qkv_weight", dit.qkv[i][0]), ("qkv_bias", dit.qkv[i][1]), ("modulation", block.modulation),
                ("norm_q", attn.norm_q.weight), ("norm_k", attn.norm_k.weight),
                ("o_weight", attn.o.weight.to(bf16)), ("o_bias", attn.o.bias.to(bf16)),
                ("norm3_weight", block.norm3.weight), ("norm3_bias", block.norm3.bias),
                ("cross_q_weight", cross.q.weight.to(bf16)), ("cross_q_bias", cross.q.bias.to(bf16)),
                ("cross_norm_q", cross.norm_q.weight),
                ("cross_o_weight", cross.o.weight.to(bf16)), ("cross_o_bias", cross.o.bias.to(bf16)),
                ("ffn_up_weight", block.ffn[0].weight.to(bf16)), ("ffn_up_bias", block.ffn[0].bias.to(bf16)),
                ("ffn_down_weight", block.ffn[2].weight.to(bf16)), ("ffn_down_bias", block.ffn[2].bias.to(bf16)),
                ("text_k", dit.text_k[i]), ("text_v", dit.text_v[i]), ("cam", dit.camera_params[i][:, :, n:]),
                ("cache_k", dit.cache_k[i][:, :end]), ("cache_v", dit.cache_v[i][:, :end]),
            ]
            frame = encode("layer", layer=i, arrays=arrays)
            sent += len(frame)
            link.send(frame)
        header, _ = link.read()
        assert header["op"] == "ready", header
        print(f"iPad loaded {sent / 1e9:.1f} GB in {time.perf_counter() - t0:.1f} s "
              f"(app footprint {header['meta']['footprint'] / 1e9:.1f} GB); "
              f"iPad runs tokens {n}..{tokens} ({tokens - n} tokens, {(tokens - n) / tokens:.0%})", flush=True)

        class Exchange:
            def __call__(self, i, key, val):
                link.send(encode("kv", layer=i, arrays=[("k", key), ("v", val)]))

            def receive(self, i):
                header, arrays = link.read()
                assert header["op"] == "kv" and header["layer"] == i, header
                return arrays["k"], arrays["v"]

        pairs = []
        for p in range(args.passes):
            # Interleaved with a Mac-alone pass each time: background GPU work moves timings by 10-30%
            _, alone = timed(mac_alone, repeats=1)
            synchronize(device)
            start = time.perf_counter()
            link.send(encode("pass", arrays=[("time_e0", dit.time_e0), ("value", value0[:, n:])]))
            mac_part = mac_share(n, Exchange())
            header, arrays = link.read()
            synchronize(device)
            elapsed = time.perf_counter() - start
            assert header["op"] == "result", header
            pairs.append((alone, elapsed))
            print(f"split pass {p + 1}: {elapsed:.2f} s (Mac alone just before {alone:.2f} s, {alone / elapsed:.2f}x); "
                  f"iPad {header['meta']['ms'] / 1000:.2f} s, of it {header['meta']['wait_ms'] / 1000:.2f} s "
                  f"waiting for the Mac's K/V", flush=True)
            if p == 0:
                compare("split pass vs the Mac alone", torch.cat([mac_part, arrays["value"]], 1), reference)
                compare("  the Mac's tokens", mac_part, reference[:, :n])
                compare("  the iPad's tokens", arrays["value"], reference[:, n:])
        steady = pairs[1:] or pairs  # the first split pass includes the iPad's graph compiles
        speedups = sorted(alone / split for alone, split in steady)
        middle = len(steady) // 2
        alone_median = sorted(a for a, _ in steady)[middle]
        split_median = sorted(s for _, s in steady)[middle]
        print(f"median of {len(steady)} interleaved pairs: Mac alone {alone_median:.2f} s, split {split_median:.2f} s, "
              f"speedup {speedups[middle]:.2f}x", flush=True)
        link.close()
        raise Done

    with torch.inference_mode():
        pipe.start(Image.open(example / "image.jpg").convert("RGB"), (example / "prompt.txt").read_text().strip(),
                   seed=0, latents=trajectory.latents)
        for chunk in range(trajectory.num_chunks):
            if chunk == CHUNK:
                dit._run = experiment
            try:
                pipe.step(trajectory.rays(chunk))
            except Done:
                return
            print(f"chunk {chunk} done", flush=True)


if __name__ == "__main__":
    main()
