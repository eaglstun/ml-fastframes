"""Verify Metal int8 KV cache quantize/dequantize bit-parity and fallbacks."""
import os
from unittest.mock import patch

import pytest
import torch

from fastframes.device import limit_mps_memory
from fastframes.wan.causal_model import QuantizedKV, _attended, allocate_kv_cache, kv_read, kv_write
from fastframes.wan.mps_kv8 import try_dequantize_kv, try_quantize_kv


def test_cpu_never_loads_shader():
    cache = QuantizedKV((1, 8, 2, 128), "cpu")
    x = torch.randn(1, 4, 2, 128, dtype=torch.bfloat16)
    with patch("fastframes.wan.mps_kv8._lib", side_effect=AssertionError("unexpected shader load")):
        assert not try_quantize_kv(cache.data[:, :4], cache.scale[:, :4], x)
        assert try_dequantize_kv(cache.data[:, :4], cache.scale[:, :4]) is None
        kv_write(cache, 0, 4, x)
        out = kv_read(cache, 0, 4)
        assert out.shape == (1, 4, 2, 128) and out.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
class TestMPSQuantizedKV:
    @classmethod
    def setup_class(cls):
        limit_mps_memory(0.45)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("batch,heads", [(1, 12), (1, 40), (2, 4)])
    def test_store_and_load_bit_identical(self, dtype, batch, heads):
        torch.manual_seed(0)
        shape = (batch, 24, heads, 128)
        eager = QuantizedKV(shape, "mps")
        metal = QuantizedKV(shape, "mps")
        x0 = torch.randn(batch, 8, heads, 128, device="mps", dtype=dtype) * 5.0
        x1 = torch.randn(batch, 10, heads, 128, device="mps", dtype=dtype) * 0.03
        with patch.dict(os.environ, FASTFRAMES_MPS_KV8="eager"):
            eager.store(0, 8, x0)
            eager.store(8, 18, x1)
            expected_load = eager.load(0, 18)
            out_eager = torch.empty((batch, 24, heads, 128), device="mps", dtype=torch.bfloat16)
            eager.load(2, 16, out=out_eager[:, 2:16])
        with patch.dict(os.environ, FASTFRAMES_MPS_KV8="metal"):
            assert try_quantize_kv(metal.data[:, 0:8], metal.scale[:, 0:8], x0)
            metal.store(8, 18, x1)
            actual_load = metal.load(0, 18)
            out_metal = torch.empty((batch, 24, heads, 128), device="mps", dtype=torch.bfloat16)
            metal.load(2, 16, out=out_metal[:, 2:16])

        assert torch.equal(metal.data.cpu(), eager.data.cpu())
        assert torch.equal(metal.scale.cpu(), eager.scale.cpu())
        assert torch.equal(actual_load.cpu().view(torch.int16), expected_load.cpu().view(torch.int16))
        assert torch.equal(out_metal[:, 2:16].cpu().view(torch.int16), out_eager[:, 2:16].cpu().view(torch.int16))

    def test_edge_values_and_tie_rounding(self):
        # All-zero rows (clamped at 1e-12), signed zeros, saturation, and exact half-way rounding ties.
        row0 = torch.zeros(128, dtype=torch.float32)
        row1 = torch.tensor([-0.0, 0.0, 1e-15, -1e-15] * 32, dtype=torch.float32)
        # With max = 127.0, scale = 1.0, so 0.5, 1.5, 2.5, -0.5, -1.5, -2.5 test round-to-even!
        ties = torch.tensor([127.0, -127.0, 0.5, 1.5, 2.5, 3.5, -0.5, -1.5] + [0.25] * 120, dtype=torch.float32)
        x = torch.stack([row0, row1, ties]).view(1, 3, 1, 128).to("mps")
        for inp in (x, x.to(torch.bfloat16)):
            eager = QuantizedKV((1, 3, 1, 128), "mps")
            metal = QuantizedKV((1, 3, 1, 128), "mps")
            with patch.dict(os.environ, FASTFRAMES_MPS_KV8="eager"):
                eager.store(0, 3, inp)
                ref = eager.load(0, 3)
            with patch.dict(os.environ, FASTFRAMES_MPS_KV8="metal"):
                metal.store(0, 3, inp)
                got = metal.load(0, 3)
            assert torch.equal(metal.data.cpu(), eager.data.cpu())
            assert torch.equal(metal.scale.cpu(), eager.scale.cpu())
            assert torch.equal(got.cpu().view(torch.int16), ref.cpu().view(torch.int16))

    def test_shared_scratch_in_allocate_kv_cache(self):
        class DummyModel:
            num_heads = 4
            dim = 512
            num_layers = 3

        caches = allocate_kv_cache(DummyModel(), num_frames=4, frame_tokens=8, device="mps", quantize=True)
        assert caches[0]["k"].scratch is caches[1]["k"].scratch
        assert caches[0]["v"].scratch is caches[2]["v"].scratch
        assert caches[0]["k"].scratch is not caches[0]["v"].scratch

        k0 = torch.randn(1, 8, 4, 128, device="mps", dtype=torch.bfloat16)
        k1 = torch.randn(1, 8, 4, 128, device="mps", dtype=torch.bfloat16)
        caches[0]["k"].store(0, 8, k0)
        caches[0]["k"].store(8, 16, k1)
        with patch.dict(os.environ, FASTFRAMES_MPS_KV8="eager"):
            expected = _attended(caches[0]["k"], 8, 16, k1).clone()
        with patch.dict(os.environ, FASTFRAMES_MPS_KV8="metal"):
            actual = _attended(caches[0]["k"], 8, 16, k1)
            assert actual.data_ptr() == caches[0]["k"].scratch.data_ptr()
            assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))

    def test_unsupported_head_dim_and_strided_views_fallback(self):
        # head_dim != 128 falls back to eager cleanly
        cache64 = QuantizedKV((1, 8, 2, 64), "mps")
        x64 = torch.randn(1, 4, 2, 64, device="mps", dtype=torch.bfloat16)
        assert not try_quantize_kv(cache64.data[:, :4], cache64.scale[:, :4], x64)
        cache64.store(0, 4, x64)
        assert try_dequantize_kv(cache64.data[:, :4], cache64.scale[:, :4]) is None
        out64 = cache64.load(0, 4)
        assert out64.shape == (1, 4, 2, 64) and out64.dtype == torch.bfloat16
