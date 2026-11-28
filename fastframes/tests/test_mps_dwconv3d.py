"""Verify MPS causal 3x3x3 depthwise convolution accuracy, streaming parity, and fallbacks."""
import os
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from apps.lingbot.decoder import StreamingDecoder
from fastframes.device import limit_mps_memory
from fastframes.models.mps_dwconv3d import try_causal_dwconv3d
from fastframes.models.pixel_decoder import PixelDecoder, _CausalConv3d


def test_cpu_never_loads_shader():
    conv = _CausalConv3d(8, 8, kernel_size=3, padding=1, groups=8)
    x = torch.randn(1, 8, 4, 6, 6)
    with patch("fastframes.models.mps_dwconv3d._lib", side_effect=AssertionError("unexpected shader load")):
        got = try_causal_dwconv3d(x, conv.weight, conv.bias, conv.stride, conv.dilation, conv.groups, conv._pad)
        assert got is None
        out = conv(x)
        assert out.shape == x.shape


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
class TestMPSDecoderConv:
    @classmethod
    def setup_class(cls):
        limit_mps_memory(0.45)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("with_bias", [True, False])
    def test_matches_cpu_fp64_reference(self, dtype, with_bias):
        torch.manual_seed(0)
        conv_cpu = _CausalConv3d(32, 32, kernel_size=3, padding=1, groups=32, bias=with_bias).double()
        x_cpu = torch.randn(2, 32, 5, 10, 12, dtype=torch.float64)
        ref = F.conv3d(F.pad(x_cpu, (1, 1, 1, 1, 2, 0)), conv_cpu.weight, conv_cpu.bias, groups=32)

        conv_mps = _CausalConv3d(32, 32, kernel_size=3, padding=1, groups=32, bias=with_bias).to("mps", dtype=dtype)
        with torch.no_grad():
            conv_mps.weight.copy_(conv_cpu.weight.to(dtype))
            if with_bias:
                conv_mps.bias.copy_(conv_cpu.bias.to(dtype))
        x_mps = x_cpu.to("mps", dtype=dtype)

        with patch.dict(os.environ, FASTFRAMES_MPS_DECODER_CONV="metal"), torch.no_grad():
            got = try_causal_dwconv3d(x_mps, conv_mps.weight, conv_mps.bias,
                                      conv_mps.stride, conv_mps.dilation, conv_mps.groups, conv_mps._pad)
        assert got is not None and got.dtype == dtype
        rel_err = ((got.cpu().double() - ref).norm() / ref.norm()).item()
        if dtype == torch.float32:
            assert rel_err < 1e-6
        else:
            # Metal accumulates in FP32, so BF16 output is closer to FP64 than native MPS BF16 conv3d.
            with patch.dict(os.environ, FASTFRAMES_MPS_DECODER_CONV="native"), torch.no_grad():
                native = conv_mps(x_mps)
            native_err = ((native.cpu().double() - ref).norm() / ref.norm()).item()
            assert rel_err < 5e-3 and rel_err <= native_err

    def test_streaming_decoder_bit_identical_to_clip_decoder(self):
        torch.manual_seed(1)
        dec = PixelDecoder(dim=64, n_low_blocks=2, n_high_blocks=1).to("mps", dtype=torch.bfloat16).eval()
        # Give fc_out and pw2 non-zero weights so every block contributes to the output!
        with torch.no_grad():
            for p in dec.parameters():
                p.normal_(std=0.05)
        stream = StreamingDecoder(dec, rows=6, width=8)
        stream.reset()

        chunks = 3
        features = torch.randn(1, chunks * 4 * 6 * 8, 64, device="mps", dtype=torch.bfloat16)
        modulation = torch.randn(1, chunks * 4, 1, 64, device="mps", dtype=torch.bfloat16)
        grid = torch.tensor([[chunks * 4, 6, 8]], device="mps", dtype=torch.long)

        with patch.dict(os.environ, FASTFRAMES_MPS_DECODER_CONV="metal"), torch.no_grad():
            clip_out = dec(features, modulation, grid)
            parts = []
            tokens_per_chunk = 4 * 6 * 8
            for c in range(chunks):
                parts.append(stream(features[:, c * tokens_per_chunk:(c + 1) * tokens_per_chunk],
                                    modulation[:, c * 4:(c + 1) * 4]))
            stream_out = torch.cat(parts, dim=1)
        assert torch.equal(clip_out.cpu().view(torch.int16), stream_out.cpu().view(torch.int16))

    def test_autocast_fp32_weights_emit_bf16(self):
        torch.manual_seed(2)
        conv = _CausalConv3d(16, 16, kernel_size=3, padding=1, groups=16).to("mps", dtype=torch.float32)
        x = torch.randn(1, 16, 4, 8, 8, device="mps", dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast("mps", dtype=torch.bfloat16):
            out = conv(x)
        assert out.dtype == torch.bfloat16 and out.shape == x.shape

    def test_unsupported_shapes_and_autograd_fallback(self):
        conv_dense = _CausalConv3d(8, 8, kernel_size=3, padding=1, groups=1).to("mps", dtype=torch.bfloat16)
        x = torch.randn(1, 8, 4, 6, 6, device="mps", dtype=torch.bfloat16)
        assert try_causal_dwconv3d(x, conv_dense.weight, conv_dense.bias,
                                   conv_dense.stride, conv_dense.dilation, conv_dense.groups, conv_dense._pad) is None
        conv_dw = _CausalConv3d(8, 8, kernel_size=3, padding=1, groups=8).to("mps", dtype=torch.bfloat16)
        xg = x.detach().clone().requires_grad_(True)
        assert try_causal_dwconv3d(xg, conv_dw.weight, conv_dw.bias,
                                   conv_dw.stride, conv_dw.dilation, conv_dw.groups, conv_dw._pad) is None
        loss = conv_dw(xg).float().sum()
        assert torch.autograd.grad(loss, xg)[0] is not None
