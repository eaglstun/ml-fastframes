#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Save a demo session as a video when it ends: no torch, runs in the server process.

The server already receives every chunk as JPEG frames (the ones the browser plays), so a
session's video is those frames piped through ffmpeg into a faststart H.264 MP4 at the model's
16 fps, with the BT.601 limited-range tagging of ``fastframes.video``. Next to it go the start
image (PNG) and a JSON file with the prompt, seed, size and camera actions.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

__all__ = ["FPS", "VIDEO_NAME", "save_recording"]

FPS = 16  # LingBot generates 16 fps video (one chunk = 16 frames = 1 s)
VIDEO_NAME = re.compile(r"^\d{8}-\d{6}-seed\d+-\d+x\d+(-\d+)?\.mp4$")  # what save_recording writes


def _unique_stem(directory: Path, stem: str) -> str:
    candidate, n = stem, 1
    while (directory / f"{candidate}.mp4").exists():
        n += 1
        candidate = f"{stem}-{n}"
    return candidate


def save_recording(directory, frames, *, prompt, seed, size, actions, image=None, started=None):
    """Write ``frames`` (JPEG bytes, in order) to ``directory`` as an MP4 plus sidecars; returns the MP4 path.

    ``size`` is ``(width, height)``; ``actions`` the camera action of every chunk; ``image`` the
    prepared start image (PNG bytes).
    """
    if not frames:
        raise ValueError("no frames to save")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    width, height = size
    started = started if started is not None else time.time()
    stem = _unique_stem(directory, f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(started))}-seed{seed}-"
                                   f"{width}x{height}")
    path = directory / f"{stem}.mp4"
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(FPS),
           "-i", "-", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
           "-vf", "scale=out_color_matrix=bt601:out_range=tv",
           "-colorspace", "smpte170m", "-color_primaries", "smpte170m", "-color_trc", "smpte170m",
           "-color_range", "tv", "-movflags", "+faststart", str(path)]
    subprocess.run(cmd, input=b"".join(frames), check=True, capture_output=True)
    if image is not None:
        (directory / f"{stem}.png").write_bytes(image)
    meta = {"video": path.name, "prompt": prompt, "seed": seed, "width": width, "height": height,
            "fps": FPS, "frames": len(frames), "seconds": round(len(frames) / FPS, 3),
            "chunks": len(actions), "actions": list(actions),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
            "start_image": f"{stem}.png" if image is not None else None}
    (directory / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path
