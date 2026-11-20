"""Device helpers for running inference off CUDA (Apple Silicon MPS / CPU).

MPS has no float64 / complex128, so the few places that compute in double precision
do it on the CPU (``f64_device``) and move the (small) results back.
"""
import os

import torch

__all__ = ["default_device", "device_name", "f64_device", "limit_mps_memory", "on_mps", "release_mps_cache",
           "synchronize"]


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


def synchronize(device):
    """``torch.cuda.synchronize`` / ``torch.mps.synchronize`` for ``device`` (no-op on CPU)."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def device_name(device):
    """``torch.cuda.get_device_name`` / the Apple chip (e.g. ``Apple M4 Max``) / ``cpu``."""
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        import subprocess
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or "mps"
        except Exception:  # diagnostic only
            return "mps"
    return "cpu"


def limit_mps_memory(default=0.6):
    """Cap MPS allocations so an oversized run fails with an out-of-memory error.

    By default MPS may allocate up to 1.7x the recommended working set (~75% of RAM) and then
    swaps; on a machine that is also running other apps that stalls the whole system until the
    watchdog panics and reboots it. The cap is ``$FASTFRAMES_MPS_MEMORY_FRACTION`` (else ``default``) of
    the recommended working set; ``PYTORCH_MPS_HIGH_WATERMARK_RATIO``, if set, takes precedence.
    """
    if not torch.backends.mps.is_available() or "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in os.environ:
        return
    torch.mps.set_per_process_memory_fraction(float(os.environ.get("FASTFRAMES_MPS_MEMORY_FRACTION", default)))


def release_mps_cache(device):
    """On MPS, wait for the GPU and return cached allocator blocks (no-op elsewhere)."""
    if torch.device(device).type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
