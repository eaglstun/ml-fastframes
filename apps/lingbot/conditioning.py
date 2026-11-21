#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Conditioning of FastFrames-LingBot: prompt, first image and camera.

LingBot generates ``464 x 832`` frames: the upstream ``480*832`` size setting derives
the latent height from ``sqrt(480 * 832 * 480 / 832) = 479.99999...``, which rounds down
to 58 latent rows (464 pixels). Camera intrinsics are given for ``480 x 832`` frames,
as in the upstream examples, and are rescaled like upstream.

The camera enters the DiT as Pluecker rays of four latent frames at a time. Two sources
are supported:

* ``Trajectory`` -- a complete camera path at the RGB frame rate (``poses.npy`` and
  ``intrinsics.npy`` of an upstream example), resampled and normalized over the whole path
  exactly like the upstream generator;
* ``LiveCamera`` -- absolute camera-to-world poses supplied chunk by chunk (interactive
  control); each chunk is expressed relative to the last pose of the previous chunk and
  normalized on its own, so the largest translation of a chunk has unit length.
  ``KeyboardCamera`` turns keypress actions into such poses.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["HEIGHT", "WIDTH", "LATENT_HEIGHT", "LATENT_WIDTH", "CHUNK", "DEFAULT_INTRINSICS", "DEFAULT_SIZE",
           "parse_size", "set_size",
           "text_context", "image_condition", "LazyImageCondition", "latent_count", "Trajectory", "LiveCamera",
           "KeyboardCamera", "ACTIONS"]

HEIGHT, WIDTH = 464, 832
LATENT_HEIGHT, LATENT_WIDTH = 58, 104
CHUNK = 4  # latent frames per generation step
DEFAULT_SIZE = (WIDTH, HEIGHT)  # the size LingBot and the FastFrames networks were trained at


def parse_size(text):
    """``"832x464"`` (width x height) -> ``(832, 464)``."""
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"size must look like 576x320 (width x height), got {text!r}") from None
    return width, height


def set_size(width, height):
    """Generate ``width x height`` frames instead of the trained ``832 x 464``; call before building a pipeline.

    Only 832 x 464 is what LingBot and the FastFrames networks were trained at. Smaller sizes run
    much faster (on an M4 Max, a full-cache chunk takes ~7 s at 576 x 320 and ~3.6 s at 416 x 240,
    against ~31 s) and are increasingly out of distribution: softer frames, and at 416 x 240 the
    scene melts and stops following the camera within a few chunks. Both sides must be multiples
    of 16 (8x VAE downsampling, then 2x2 DiT patches).

    The size constants are module globals copied by ``from .conditioning import ...`` into the
    other LingBot modules, so this updates them (and the token counts derived from them) in every
    loaded ``apps.lingbot`` module; modules imported later read the new values here.
    """
    import sys

    width, height = int(width), int(height)
    if width % 16 or height % 16 or not (128 <= width <= 1280 and 128 <= height <= 1280):
        raise ValueError(f"LingBot sizes are multiples of 16 between 128 and 1280, got {width} x {height}")
    grid_h, grid_w = height // 16, width // 16
    values = {"HEIGHT": height, "WIDTH": width, "LATENT_HEIGHT": height // 8, "LATENT_WIDTH": width // 8,
              "GRID_H": grid_h, "GRID_W": grid_w, "FRAME_TOKENS": grid_h * grid_w,
              "TOKENS": CHUNK * grid_h * grid_w}
    globals().update({k: values[k] for k in ("HEIGHT", "WIDTH", "LATENT_HEIGHT", "LATENT_WIDTH")})
    for name, module in list(sys.modules.items()):
        if module is not None and name.startswith("apps.lingbot.") and name != __name__:
            for key, value in values.items():
                if key in vars(module):
                    setattr(module, key, value)
# fx, fy, cx, cy for 480 x 832 frames (the intrinsics of upstream example 00).
DEFAULT_INTRINSICS = (502.9116, 503.10812, 415.77786, 239.77779)


def text_context(text_encoder, prompt, device):
    """umT5 embedding of ``prompt`` padded to 512 tokens, ``[512, 4096]`` BF16."""
    text_encoder.model.to(device)
    raw = text_encoder([prompt], device)[0]
    text_encoder.model.cpu()
    return torch.cat([raw, raw.new_zeros(512 - raw.shape[0], raw.shape[1])])


def image_condition(vae, image, latents, device):
    """First-image conditioning ``[20, latents, 58, 104]``: 4 mask channels + 16 VAE channels.

    As upstream, the image is resized (bicubic, on the CPU) to ``464 x 832`` and encoded
    followed by ``4 * (latents - 1)`` black frames.
    """
    from torchvision.transforms import functional as TF

    if abs(image.width * HEIGHT / (image.height * WIDTH) - 1) > 0.05:
        warnings.warn(f"the {image.width} x {image.height} image is stretched to {WIDTH} x {HEIGHT}")
    pixels = TF.to_tensor(image.convert("RGB")).sub_(0.5).div_(0.5)
    first = F.interpolate(pixels[None], size=(HEIGHT, WIDTH), mode="bicubic").transpose(0, 1)
    video = torch.cat([first, torch.zeros(3, 4 * (latents - 1), HEIGHT, WIDTH)], dim=1).to(device)
    encoded = vae.encode([video])[0]
    mask = torch.zeros(4, latents, LATENT_HEIGHT, LATENT_WIDTH, device=device, dtype=encoded.dtype)
    mask[:, 0] = 1
    return torch.cat([mask, encoded])


class LazyImageCondition:
    """``image_condition`` encoded chunk by chunk, on first use.

    Upstream encodes the first image plus ``4 * (latents - 1)`` black frames for the whole horizon
    up front. The Wan VAE encoder is causal: it consumes 1 frame, then 4 frames per latent, with a
    rolling feature cache, and its output projection is per frame. So encoding latent ``l`` only
    when chunk ``l // 4`` needs it, while keeping the cache alive, performs the same computation:
    same results, without the multi-minute wait and peak memory of a long horizon.
    ``chunk(c)`` must be called with ``c = 0, 1, 2, ...`` (each at most once beyond the cache).

    After the first image the encoder only sees black frames, and its causal state converges: at
    832 x 464 the latents are bit-identical from latent 30 on. Once a new latent and the whole
    feature cache are both bit-identical to the previous step's, the encoder is at a fixed point
    (same state, same zero input), so every later latent equals the last one and is reused without
    encoding: exactly the same values, ~1 s of VAE work per latent saved (~4 s per chunk on MPS).
    """

    def __init__(self, vae, image, latents, device):
        from torchvision.transforms import functional as TF

        if abs(image.width * HEIGHT / (image.height * WIDTH) - 1) > 0.05:
            warnings.warn(f"the {image.width} x {image.height} image is stretched to {WIDTH} x {HEIGHT}")
        pixels = TF.to_tensor(image.convert("RGB")).sub_(0.5).div_(0.5)
        self.first = F.interpolate(pixels[None], size=(HEIGHT, WIDTH), mode="bicubic").transpose(0, 1)[None]
        self.vae, self.latents, self.device = vae, latents, torch.device(device)
        self.encoded = []  # latent frames [16, 1, 58, 104] in order
        self.steady = False  # the encoder reached its fixed point: later latents repeat the last one
        self._previous_cache = None  # feature cache after the previous latent, kept once outputs repeat
        vae.model.clear_cache()

    @staticmethod
    def _same_cache(a, b):
        return len(a) == len(b) and all(
            torch.equal(x, y) if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) else
            (not isinstance(x, torch.Tensor) and not isinstance(y, torch.Tensor) and x == y)
            for x, y in zip(a, b))

    def _encode_next(self):
        model, index = self.vae.model, len(self.encoded)
        if self.steady:
            self.encoded.append(self.encoded[-1])
            return
        frames = self.first if index == 0 else torch.zeros(1, 3, 4, HEIGHT, WIDTH)
        with torch.amp.autocast("cuda", dtype=self.vae.dtype, enabled=self.device.type == "cuda"):
            model._enc_conv_idx = [0]
            out = model.encoder(frames.to(self.device), feat_cache=model._enc_feat_map,
                                feat_idx=model._enc_conv_idx)
            mu = model.conv1(out).chunk(2, dim=1)[0]
            scale = self.vae.scale
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, model.z_dim, 1, 1, 1)) * scale[1].view(1, model.z_dim, 1, 1, 1)
            else:
                mu = (mu - scale[0]) * scale[1]
        latent = mu.float().squeeze(0)
        cache = model._enc_feat_map
        if index >= 2 and torch.equal(latent, self.encoded[-1]):
            # Outputs repeat: check whether the whole causal state repeats too (fixed point)
            if self._previous_cache is not None and self._same_cache(cache, self._previous_cache):
                self.steady, self._previous_cache = True, None
            else:
                self._previous_cache = [x.clone() if isinstance(x, torch.Tensor) else x for x in cache]
        else:
            self._previous_cache = None
        self.encoded.append(latent)

    def chunk(self, c):
        """``[20, 4, 58, 104]`` condition of latent frames ``[4c, 4c + 4)``."""
        while len(self.encoded) < CHUNK * (c + 1):
            self._encode_next()
        if len(self.encoded) == self.latents:
            self.vae.model.clear_cache()  # horizon fully encoded: drop the feature cache
        encoded = torch.cat(self.encoded[CHUNK * c:CHUNK * (c + 1)], dim=1)
        mask = torch.zeros(4, CHUNK, LATENT_HEIGHT, LATENT_WIDTH, device=encoded.device, dtype=encoded.dtype)
        if c == 0:
            mask[:, 0] = 1
        return torch.cat([mask, encoded])


def latent_count(frames):
    """Latent frames generated for ``frames`` requested RGB frames (upstream rounding)."""
    latents = (frames - 1) // 4 + 1
    return latents - latents % CHUNK


def _lens(intrinsics, device):
    from wan.utils.cam_utils import get_Ks_transformed

    ks = torch.as_tensor(np.asarray(intrinsics, dtype=np.float32)).reshape(-1, 4)
    ks = get_Ks_transformed(ks, height_org=480, width_org=832, height_resize=HEIGHT,
                            width_resize=WIDTH, height_final=HEIGHT, width_final=WIDTH)
    return ks[:1].repeat(CHUNK, 1).to(device)


def _rays(relative, lens):
    """Pluecker rays of four relative poses -> ``[1, 384, 4, 58, 104]`` BF16 DiT input."""
    from wan.utils.cam_utils import get_plucker_embeddings

    rays = get_plucker_embeddings(relative, lens, HEIGHT, WIDTH)
    rays = rays.reshape(CHUNK, LATENT_HEIGHT, 8, LATENT_WIDTH, 8, 6).permute(5, 2, 4, 0, 1, 3)
    return rays.reshape(1, 6 * 64, CHUNK, LATENT_HEIGHT, LATENT_WIDTH).to(torch.bfloat16)


class Trajectory:
    """A camera path at the RGB frame rate, conditioned like the upstream generator.

    ``poses``: ``[N, 4, 4]`` OpenCV camera-to-world matrices; ``intrinsics``: ``[N, 4]`` or
    ``[4]`` (only the first row is used, as upstream). The first ``frames`` poses are
    resampled to one pose per latent frame and turned into normalized frame-to-frame motion.
    """

    def __init__(self, poses, intrinsics, frames, device="cuda"):
        from wan.utils.cam_utils import compute_relative_poses, interpolate_camera_poses

        poses = np.asarray(poses)
        usable = (len(poses) - 1) // 4 * 4 + 1
        frames = min((frames - 1) // 4 * 4 + 1, usable)
        poses = poses[:frames]
        self.latents = latent_count(frames)
        if self.latents < CHUNK:
            raise ValueError(f"a trajectory of {frames} frames is shorter than one chunk")
        world = interpolate_camera_poses(
            src_indices=np.linspace(0, frames - 1, frames), src_rot_mat=poses[:, :3, :3],
            src_trans_vec=poses[:, :3, 3], tgt_indices=np.linspace(0, frames - 1, self.latents))
        self.relative = compute_relative_poses(world, framewise=True).to(device)
        self.lens = _lens(intrinsics, device)

    @property
    def num_chunks(self):
        return self.latents // CHUNK

    def rays(self, chunk):
        return _rays(self.relative[CHUNK * chunk:CHUNK * (chunk + 1)], self.lens)


class LiveCamera:
    """Absolute camera-to-world poses given four latent frames at a time.

    As upstream normalizes a whole path, the frame-to-frame translations of each chunk are
    divided by their maximum: the direction of motion is kept, its speed is not.
    """

    def __init__(self, intrinsics=DEFAULT_INTRINSICS, device="cuda"):
        self.lens = _lens(intrinsics, device)
        self.previous = torch.eye(4, device=device)

    def reset(self):
        self.previous = torch.eye(4, device=self.previous.device)

    def rays(self, c2ws):
        """``c2ws``: ``[4, 4, 4]`` poses of the next four latent frames (start pose: identity)."""
        from wan.utils.cam_utils import compute_relative_poses

        c2ws = torch.as_tensor(c2ws).to(device=self.previous.device, dtype=torch.float32)
        if tuple(c2ws.shape) != (CHUNK, 4, 4):
            raise ValueError(f"expected {CHUNK} 4x4 poses, got {tuple(c2ws.shape)}")
        relative = compute_relative_poses(torch.cat((self.previous[None], c2ws)), framewise=True)[1:]
        self.previous = c2ws[-1].clone()
        return _rays(relative, self.lens)


def _rotation(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# action -> (translation in camera coordinates, rotation); OpenCV axes: +z forward, +y down.
# LiveCamera normalizes translations, so only the direction of _STEP matters.
_STEP, _TURN = 0.08, math.radians(3.0)
ACTIONS = {
    "idle": None,
    "forward": ((0, 0, _STEP), None), "backward": ((0, 0, -_STEP), None),
    "left": ((-_STEP, 0, 0), None), "right": ((_STEP, 0, 0), None),
    "up": ((0, -_STEP, 0), None), "down": ((0, _STEP, 0), None),
    "yaw_left": (None, ("y", -_TURN)), "yaw_right": (None, ("y", _TURN)),
    "pitch_up": (None, ("x", _TURN)), "pitch_down": (None, ("x", -_TURN)),
}


class KeyboardCamera:
    """Keypress control: every action moves or turns (3 degrees) the camera once per latent frame.

    ``chunk(action)`` returns the ``[4, 4, 4]`` camera-to-world poses of the next chunk for
    ``LiveCamera.rays``. The first chunk starts at the initial pose.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.c2w = np.eye(4)
        self.first = True

    def _apply(self, action):
        translation, rotation = ACTIONS[action]
        if rotation is not None:
            self.c2w[:3, :3] = self.c2w[:3, :3] @ _rotation(*rotation)
        else:
            self.c2w[:3, 3] += self.c2w[:3, :3] @ np.asarray(translation, dtype=np.float64)

    def chunk(self, action="idle"):
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}; choose from {sorted(ACTIONS)}")
        poses = [self.c2w.copy()] if self.first or action == "idle" else []
        while len(poses) < CHUNK:
            if action != "idle":
                self._apply(action)
            poses.append(self.c2w.copy())
        self.first = False
        return torch.from_numpy(np.stack(poses).astype(np.float32))
