#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Share LingBot's DiT with an iPad (or another Apple device) by tokens, over a socket.

The iPad runs SplitStage.swift from the exo-touch app (MLX-Swift, github.com/eaglstun/exo-touch,
`SPIKE_MODE=split`). Each
denoising pass's chunk tokens are split: this Mac keeps [0, n), the iPad runs [n, tokens) through
all 30 blocks. In every block both sides compute their tokens' keys and values, swap the canonical
(pre-RoPE) ones, rotate the other side's with the chunk's RoPE table, write both into their own KV
cache and attend over all of it; the rest of a block is per token. Everything else stays here: the
patch embedding, the head, the KV refiner (it reads every token's canonical K/V, which this side
then holds), the decoder and the VAE.

The iPad mirrors the cache: it computes its own camera rows from its slice of the camera tokens,
evicts like ``FastFramesLingBot._evict``, zeroes on restart, and receives the refined rows of each
chunk (``sync_rows``; the demo does it between turns) and the text K/V of each prompt.

The exchange hides behind attention: once a block's own K/V are on their way, both sides attend over
the cache's older rows while the other side's are in flight, then over this chunk's rows, and merge the
two by their log-sum-exp (``overlap``; the first chunk, with no older rows, waits as before).

Frames both ways: little-endian u32 header length, a JSON header
``{"op", "layer", "meta", "arrays": [{"name", "shape", "dtype"}]}``, then the arrays' bytes in order.

    pipe = FastFramesLingBot.from_pretrained(...)
    IPadAssist.attach(pipe, "169.254.154.199", share=0.55)   # before start()
"""
from __future__ import annotations

import json
import queue
import socket
import struct
import threading

import numpy as np
import torch
import torch.nn.functional as F

from fastframes.wan.mps_flash_attention import flash_attention_lse, merge_attention

from . import dit as lingbot_dit
from .conditioning import CHUNK
from .dit import attention, layer_norm, rms_norm, rope_real

__all__ = ["IPadAssist", "encode", "DEFAULT_PORT"]

DEFAULT_PORT = 5570
DTYPES = {torch.bfloat16: "bfloat16", torch.float32: "float32", torch.float16: "float16"}
TORCH_DTYPES = {name: dtype for dtype, name in DTYPES.items()}


def _bytes(tensor):
    tensor = tensor.detach().contiguous().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    return tensor.numpy().tobytes()


def encode(op, *, layer=0, meta=None, arrays=()):
    """One frame: ``arrays`` is a sequence of (name, tensor)."""
    specs = [{"name": name, "shape": list(t.shape), "dtype": DTYPES[t.dtype]} for name, t in arrays]
    header = json.dumps({"op": op, "layer": layer, "meta": meta or {}, "arrays": specs}).encode()
    return b"".join([struct.pack("<I", len(header)), header] + [_bytes(t) for _, t in arrays])


class Link:
    """The socket; frames are written by a sender thread, so both sides can send at once."""

    def __init__(self, host, port, device):
        self.sock = socket.create_connection((host, port), timeout=5)  # an absent iPad fails fast
        self.sock.settimeout(600)  # a pass, or the weights at setup, can take a while
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.device, self.outbox = device, queue.Queue()
        self.timing = {"socket": 0.0, "decode": 0.0, "encode": 0.0}  # seconds, for profiling
        self.sender = threading.Thread(target=self._send_loop, daemon=True)
        self.sender.start()

    def _send_loop(self):
        while (data := self.outbox.get()) is not None:
            self.sock.sendall(data)

    def send(self, data):
        self.outbox.put(data)

    def _exactly(self, count):
        buffer = bytearray(count)
        view, received = memoryview(buffer), 0
        while received < count:
            n = self.sock.recv_into(view[received:], count - received)
            if n == 0:
                raise ConnectionError("the iPad closed the connection")
            received += n
        return buffer

    def read(self, op=None, layer=None):
        import time

        start = time.perf_counter()
        (length,) = struct.unpack("<I", self._exactly(4))
        header = json.loads(self._exactly(length))
        if header["op"] == "error":
            raise RuntimeError(f"iPad: {header['meta'].get('message')}")
        if (op is not None and header["op"] != op) or (layer is not None and header["layer"] != layer):
            raise RuntimeError(f"expected {op} {layer} from the iPad, got {header['op']} {header['layer']}")
        raws = []
        for spec in header["arrays"]:
            dtype = TORCH_DTYPES[spec["dtype"]]
            count = int(np.prod(spec["shape"])) * (4 if dtype == torch.float32 else 2)
            raws.append((spec, dtype, self._exactly(count)))
        received = time.perf_counter()
        arrays = {spec["name"]: torch.frombuffer(raw, dtype=dtype).view(spec["shape"]).to(self.device)
                  for spec, dtype, raw in raws}
        self.timing["socket"] += received - start
        self.timing["decode"] += time.perf_counter() - received
        return header, arrays

    def close(self):
        try:
            self.send(encode("bye"))
            self.outbox.put(None)
            self.sender.join(timeout=10)
        finally:
            self.sock.close()


def block_pre(block, value, qkv_weight, qkv_bias, rope, time_e0, heads):
    """``LingBotDiT._block`` up to the keys for one share of tokens: modulation, RoPE'd query,
    canonical (pre-RoPE) key, value."""
    tokens, dim = value.shape[1], value.shape[2]
    e = (block.modulation.unsqueeze(0) + time_e0).unbind(2)
    hidden = layer_norm(value, block.norm1).float() * (1 + e[1]) + e[0]
    a = block.self_attn
    query, key, val = F.linear(hidden, qkv_weight, qkv_bias).chunk(3, -1)
    query = rope_real(rms_norm(query, a.norm_q).view(1, tokens, heads, dim // heads), rope)
    key = rms_norm(key, a.norm_k).view(1, tokens, heads, dim // heads)
    return e, query, key, val.view(1, tokens, heads, dim // heads)


def block_post(block, value, e, attended, text_k, text_v, cam):
    """``LingBotDiT._block`` after self-attention, for one share's tokens."""
    tokens, dim = value.shape[1], value.shape[2]
    a = block.self_attn
    value = value + F.linear(attended.reshape(1, tokens, dim), a.o.weight, a.o.bias) * e[2]
    value = (1.0 + cam[0]) * value + cam[1]
    cross = block.cross_attn
    q = rms_norm(cross.q(layer_norm(value, block.norm3)), cross.norm_q).view(1, tokens, a.num_heads, dim // a.num_heads)
    value = value + cross.o(attention(q, text_k, text_v).reshape(1, tokens, dim))
    hidden = layer_norm(value, block.norm2).float() * (1 + e[4]) + e[3]
    return value + block.ffn(hidden) * e[5]


class IPadAssist:
    """Runs a share of every DiT pass of ``pipe`` on a connected iPad (one device pipelines only)."""

    # Shares move in steps of this many tokens: each new share compiles new graphs on the iPad
    STEP = 128

    def __init__(self, pipe, host, port=DEFAULT_PORT, share=0.55, adaptive=True, overlap=True):
        dit = pipe.dit
        if dit.world != 1:
            raise ValueError("iPad assist runs with a single-device pipeline")
        self.pipe, self.dit = pipe, dit
        self.tokens = lingbot_dit.TOKENS
        self.n = self._quantize(self.tokens - round(self.tokens * share))  # this Mac's tokens: [0, n)
        self.adaptive = adaptive
        # Attend over the older cache rows while the K/V are in flight (needs the Metal flash kernel)
        self.overlap = overlap and pipe.device.type == "mps"
        self.link = Link(host, port, pipe.device)
        self.stats = {"passes": 0, "wait": 0.0, "shares": [], "thermal": None}
        self._rates = []  # this chunk's passes: (Mac tokens/s, iPad tokens/s)
        self._setup()

    def _quantize(self, n):
        return int(min(max(round(n / self.STEP) * self.STEP, self.tokens // 5), self.tokens - self.tokens // 5))

    def rebalance(self):
        """Move the split towards where both sides finish a pass together, from the last chunk's
        compute rates: the iPad slows as it heats up, this Mac when other work uses its GPU."""
        if not self.adaptive or not self._rates:
            return
        mac = sum(r[0] for r in self._rates) / len(self._rates)
        ipad = sum(r[1] for r in self._rates) / len(self._rates)
        target = self.tokens * mac / (mac + ipad)
        self.n = self._quantize(0.5 * self.n + 0.5 * target)  # halfway: one noisy chunk can't swing it
        self._rates = []

    @classmethod
    def attach(cls, pipe, host, port=DEFAULT_PORT, share=0.55, adaptive=True, overlap=True):
        """Connect, send the weights, and route ``pipe``'s DiT passes and cache events through the iPad.
        ``share``: the iPad's starting fraction of the tokens; ``adaptive`` rebalances it every chunk;
        ``overlap`` hides the K/V exchange behind attention over the older cache rows."""
        assist = cls(pipe, host, port, share, adaptive, overlap)
        dit = pipe.dit
        assist._originals = {name: getattr(dit, name) for name in ("_run", "begin_chunk", "set_context")}
        dit._run = assist.run
        begin_chunk = dit.begin_chunk

        def begin(y, camera, rope):
            begin_chunk(y, camera, rope)
            assist.begin_chunk()
        dit.begin_chunk = begin
        set_context = dit.set_context

        def context(value):
            set_context(value)
            assist.send_context()
        dit.set_context = context
        pipe.assist = assist
        if pipe.prompt is not None:  # a session already set the prompt: the iPad needs its text K/V
            assist.send_context()
        return assist

    def detach(self):
        """Stop sharing: ``pipe`` runs its DiT alone again (the next ``start``/``restart`` resets its cache)."""
        for name, method in self._originals.items():
            setattr(self.dit, name, method)
        self.pipe.assist = None
        self.close()

    # ------------------------------------------------------------------ session events
    def _setup(self):
        dit, m = self.dit, self.dit.model
        bf16 = torch.bfloat16
        blocks = m.blocks
        meta = {"tokens": self.tokens, "start": self.n, "cache_rows": dit.cache_k.shape[2], "heads": dit.heads,
                "layers": len(blocks), "eps": blocks[0].norm1.eps, "attn_eps": blocks[0].self_attn.norm_q.eps,
                "cross_eps": blocks[0].cross_attn.norm_q.eps, "norm3_eps": blocks[0].norm3.eps}
        camera = [("cam_patch_weight", m.patch_embedding_wancamctrl.weight.to(bf16)),
                  ("cam_patch_bias", m.patch_embedding_wancamctrl.bias.to(bf16)),
                  ("c2ws1_weight", m.c2ws_hidden_states_layer1.weight.to(bf16)),
                  ("c2ws1_bias", m.c2ws_hidden_states_layer1.bias.to(bf16)),
                  ("c2ws2_weight", m.c2ws_hidden_states_layer2.weight.to(bf16)),
                  ("c2ws2_bias", m.c2ws_hidden_states_layer2.bias.to(bf16))]
        self.link.send(encode("setup", meta=meta, arrays=camera))
        for i, block in enumerate(blocks):
            attn, cross = block.self_attn, block.cross_attn
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
                ("inj1_weight", block.cam_injector_layer1.weight.to(bf16)),
                ("inj1_bias", block.cam_injector_layer1.bias.to(bf16)),
                ("inj2_weight", block.cam_injector_layer2.weight.to(bf16)),
                ("inj2_bias", block.cam_injector_layer2.bias.to(bf16)),
                ("cam_weight", dit.cam[i][0].to(bf16)), ("cam_bias", dit.cam[i][1].to(bf16)),
            ]
            self.link.send(encode("layer", layer=i, arrays=arrays))
        header, _ = self.link.read("ready")
        self.footprint = header["meta"].get("footprint", 0)

    def send_context(self):
        """The prompt's text K/V of every block (after ``LingBotDiT.set_context``)."""
        for i in range(self.dit.layers):
            self.link.send(encode("context", layer=i, arrays=[("text_k", self.dit.text_k[i]),
                                                              ("text_v", self.dit.text_v[i])]))

    def reset(self):
        """The caches were zeroed (``FastFramesLingBot.restart``)."""
        self.link.send(encode("reset"))

    def evict(self, sink, shift, rolled):
        """Mirror ``FastFramesLingBot._evict``'s row copies."""
        self.link.send(encode("evict", meta={"sink": sink, "shift": shift, "rolled": rolled}))

    def begin_chunk(self):
        """After ``LingBotDiT.begin_chunk``: the split for this chunk, its RoPE table and the iPad's
        camera tokens."""
        self.rebalance()
        self.stats["shares"].append(1 - self.n / self.tokens)
        dit = self.dit
        grid_h, grid_w = lingbot_dit.GRID_H, lingbot_dit.GRID_W
        camera = dit.camera.reshape(1, 384, CHUNK, grid_h, 2, grid_w, 2).permute(0, 2, 3, 5, 1, 4, 6)
        camera = camera.reshape(1, self.tokens, 384 * 4)[:, self.n:]
        self.link.send(encode("chunk", meta={"start": self.n}, arrays=[("rope", dit.rope), ("camera", camera)]))

    def sync_rows(self, read_frames):
        """Send this chunk's committed cache rows (the KV refiner's) to the iPad."""
        end = read_frames * lingbot_dit.FRAME_TOKENS
        rows = slice(end - self.tokens, end)
        for i in range(self.dit.layers):
            self.link.send(encode("rows", layer=i, meta={"end": end},
                                  arrays=[("k", self.dit.cache_k[i][:, rows]), ("v", self.dit.cache_v[i][:, rows])]))

    def close(self):
        self.link.close()

    # ------------------------------------------------------------------ a pass
    def run(self, read_frames, capture):
        """Replaces ``LingBotDiT._run``: the blocks on [0, n) here, [n, tokens) on the iPad."""
        import time

        dit, n, tokens = self.dit, self.n, self.tokens
        end = read_frames * lingbot_dit.FRAME_TOKENS
        rows = end - tokens
        began = time.perf_counter()
        value0 = dit._patch_run()
        overlap = self.overlap and rows > 0  # the first chunk has no older rows to attend over
        self.link.send(encode("pass", meta={"end": end, "overlap": overlap},
                              arrays=[("time_e0", dit.time_e0), ("value", value0[:, n:])]))
        value, waited = value0[:, :n], 0.0
        for i, block in enumerate(dit.model.blocks):
            cache_k, cache_v = dit.cache_k[i], dit.cache_v[i]
            e, query, key, val = block_pre(block, value, *dit.qkv[i], dit.rope[:n], dit.time_e0, dit.heads)
            encoding = time.perf_counter()
            if overlap:
                # Copy the K/V out, then queue attention over the older rows behind the copy; the events
                # commit both to the GPU, and only the copy is waited for before sending
                kv = [("k", key.to("cpu", non_blocking=True)), ("v", val.to("cpu", non_blocking=True))]
                copied = torch.mps.Event()
                copied.record()
                older = flash_attention_lse(query, cache_k[:, :rows], cache_v[:, :rows])
                torch.mps.Event().record()
                copied.synchronize()
            else:
                kv = [("k", key), ("v", val)]
            self.link.send(encode("kv", layer=i, arrays=kv))
            self.link.timing["encode"] += time.perf_counter() - encoding
            cache_k[:, rows:rows + n] = rope_real(key, dit.rope[:n])
            cache_v[:, rows:rows + n] = val
            start = time.perf_counter()
            _, theirs = self.link.read("kv", i)
            waited += time.perf_counter() - start
            cache_k[:, rows + n:end] = rope_real(theirs["k"], dit.rope[n:])
            cache_v[:, rows + n:end] = theirs["v"]
            if overlap:
                attended = merge_attention(*older, *flash_attention_lse(query, cache_k[:, rows:end],
                                                                         cache_v[:, rows:end]), query.dtype)
            else:
                attended = attention(query, cache_k[:, :end], cache_v[:, :end], 1)
            if capture:  # canonical K/V of every token, for the KV refiner
                dit.result.key[i][:, :n] = key
                dit.result.key[i][:, n:] = theirs["k"]
                dit.result.value[i][:, :n] = val
                dit.result.value[i][:, n:] = theirs["v"]
            value = block_post(block, value, e, attended, dit.text_k[i], dit.text_v[i],
                               dit.camera_params[i][:, :, :n])
        start = time.perf_counter()
        header, result = self.link.read("result")
        waited += time.perf_counter() - start
        mac_seconds = time.perf_counter() - began - waited
        ipad_seconds = (header["meta"]["ms"] - header["meta"]["wait_ms"]) / 1000
        self._rates.append((n / max(mac_seconds, 1e-3), (tokens - n) / max(ipad_seconds, 1e-3)))
        self.stats["passes"] += 1
        self.stats["wait"] += waited
        self.stats["thermal"] = header["meta"].get("thermal")
        dit._post_run(torch.cat([value, result["value"].to(value.dtype)], 1), capture)
