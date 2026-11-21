"""3D convolution as a sum of 2D convolutions, for the Wan VAE on Apple Silicon (MPS).

MPS ``conv3d`` runs at ~3.3 TFLOPS on the VAE's 3x3x3 layers whatever the dtype, while MPS
``conv2d`` reaches 13-16 TFLOPS in FP32. A ``kt x kh x kw`` convolution with temporal stride
``st`` is exactly the sum over the ``kt`` temporal taps of a 2D convolution of the matching input
frames (frames batched): ~4.2-4.8x faster on the VAE shapes, FP32, relative L2 ~1e-6 against
``conv3d`` (summation order only).

The per-tap weights are made contiguous once and cached on the module: MPS ``conv2d`` silently
computes wrong results from a non-contiguous weight view such as ``weight[:, :, k]``
(PyTorch 2.9.1).
"""
import os

import torch.nn.functional as F

__all__ = ["conv3d_as_conv2d", "patch_conv3d_class", "use_conv2d"]

# "conv2d" (default) or "native" (PyTorch's MPS conv3d), for A/B checks
_MODE = os.environ.get("FASTFRAMES_MPS_CONV3D", "conv2d").lower()
if _MODE not in ("conv2d", "native"):
    raise ValueError(f"FASTFRAMES_MPS_CONV3D must be 'conv2d' or 'native', got {_MODE!r}")


def use_conv2d(conv, x):
    """Whether ``conv`` (an ``nn.Conv3d`` with its padding already applied) should take this path."""
    return (_MODE == "conv2d" and x.device.type == "mps" and conv.groups == 1 and tuple(conv.dilation) == (1, 1, 1)
            and tuple(conv.padding) == (0, 0, 0) and x.dim() == 5)


def _taps(conv):
    weight = conv.weight
    # Rebuild when the weight is replaced or modified in place. Inference tensors (a model loaded
    # under torch.inference_mode) have no version counter and can't be modified in place anyway.
    version = None if weight.is_inference() else weight._version
    cached = getattr(conv, "_mps_conv2d_taps", None)
    if cached is None or cached[0] is not weight or cached[1] != version:
        taps = [weight[:, :, k].contiguous() for k in range(weight.shape[2])]
        conv._mps_conv2d_taps = cached = (weight, version, taps)
    return cached[2]


def conv3d_as_conv2d(conv, x):
    """``conv(x)`` for an unpadded ``nn.Conv3d`` (groups 1, dilation 1) via per-tap ``conv2d``."""
    batch, channels, frames, height, width = x.shape
    taps = _taps(conv)
    st, sh, sw = conv.stride
    out_frames = (frames - len(taps)) // st + 1
    xt = x.transpose(1, 2).contiguous()  # [B, T, C, H, W]: each frame is a contiguous [C, H, W] image
    out = None
    for k, weight in enumerate(taps):
        source = xt[:, k:k + st * (out_frames - 1) + 1:st].reshape(batch * out_frames, channels, height, width)
        y = F.conv2d(source, weight, conv.bias if k == 0 else None, stride=(sh, sw))
        out = y if out is None else out.add_(y)
    _, out_channels, out_h, out_w = out.shape
    return out.view(batch, out_frames, out_channels, out_h, out_w).transpose(1, 2).contiguous()


def patch_conv3d_class(cls):
    """Route ``cls`` (an ``nn.Conv3d`` subclass, e.g. the Wan VAE's ``CausalConv3d``) through
    ``conv3d_as_conv2d`` on MPS. Only ``_conv_forward`` is replaced, so the class's own
    ``forward`` (causal padding, feature caches) runs unchanged; other devices are untouched."""
    if getattr(cls, "_mps_conv2d_patched", False):
        return cls
    original = cls._conv_forward

    def _conv_forward(self, input, weight, bias):
        if weight is self.weight and bias is self.bias and use_conv2d(self, input):
            return conv3d_as_conv2d(self, input)
        return original(self, input, weight, bias)

    cls._conv_forward = _conv_forward
    cls._mps_conv2d_patched = True
    return cls
