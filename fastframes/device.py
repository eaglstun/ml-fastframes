"""Device helpers for running inference off CUDA (Apple Silicon MPS / CPU).

MPS has no float64 / complex128, so the few places that compute in double precision
do it on the CPU (``f64_device``) and move the (small) results back.
"""
import os

import torch

__all__ = ["default_device", "f64_device", "on_mps"]


def default_device():
    """``$FASTFRAMES_DEVICE``, else cuda > mps > cpu."""
    if os.environ.get("FASTFRAMES_DEVICE"):
        return torch.device(os.environ["FASTFRAMES_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def on_mps(device):
    return torch.device(device).type == "mps"


def f64_device(device):
    """Where float64 math for tensors on ``device`` can run."""
    return torch.device("cpu") if on_mps(device) else torch.device(device)
