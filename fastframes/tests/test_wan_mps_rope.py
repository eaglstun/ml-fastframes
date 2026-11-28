"""Verify FastFrames causal_rope_apply and rope_real Metal path against eager MPS."""
import os
from unittest.mock import patch

import pytest
import torch

from fastframes.device import limit_mps_memory
from fastframes.wan.causal_model import CausalWanModel, allocate_kv_cache, rope_real
from fastframes.wan.model import WanConfig, causal_rope_apply, rope_params
from fastframes.wan.mps_rope import try_rope


def _freqs(head_dim=128):
    return torch.cat([
        rope_params(1024, head_dim - 4 * (head_dim // 6)),
        rope_params(1024, 2 * (head_dim // 6)),
        rope_params(1024, 2 * (head_dim // 6)),
    ], dim=1)


def test_cpu_never_loads_shader():
    value = torch.ones(1, 1, 2, 128, dtype=torch.bfloat16)
    table = torch.zeros(1, 1, 64, 2)
    table[..., 0] = 1.0
    freqs = _freqs(128)
    grid = torch.tensor([[1, 1, 1]], dtype=torch.long)
    with patch("fastframes.wan.mps_rope._kernel", side_effect=AssertionError("unexpected shader load")):
        assert try_rope(value, table) is None
        assert torch.equal(rope_real(value, table), value)
        assert torch.equal(causal_rope_apply(value, grid, freqs, start_frame=0), value)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
class TestFastFramesMPSRope:
    @classmethod
    def setup_class(cls):
        limit_mps_memory(0.45)

    @pytest.mark.parametrize("heads", [12, 40])
    @pytest.mark.parametrize("batch", [1, 2])
    def test_causal_rope_apply_bit_identical(self, heads, batch):
        torch.manual_seed(0)
        freqs = _freqs(128)
        grid = torch.tensor([[1, 30, 52]] * batch, dtype=torch.long)
        x = torch.randn(batch, 1560, heads, 128, device="mps", dtype=torch.bfloat16)
        with patch.dict(os.environ, FASTFRAMES_MPS_ROPE="eager"):
            expected = causal_rope_apply(x, grid, freqs, start_frame=7)
        with patch.dict(os.environ, FASTFRAMES_MPS_ROPE="metal"):
            actual = causal_rope_apply(x, grid, freqs, start_frame=7)
        assert actual.dtype == torch.bfloat16
        assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))

    def test_padded_or_mismatched_grids_fallback_cleanly(self):
        torch.manual_seed(1)
        freqs = _freqs(128)
        x_padded = torch.randn(1, 10, 4, 128, device="mps", dtype=torch.bfloat16)
        grid_short = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with patch.dict(os.environ, FASTFRAMES_MPS_ROPE="eager"):
            expected_pad = causal_rope_apply(x_padded, grid_short, freqs, start_frame=2)
        with patch.dict(os.environ, FASTFRAMES_MPS_ROPE="metal"):
            actual_pad = causal_rope_apply(x_padded, grid_short, freqs, start_frame=2)
        assert torch.equal(actual_pad.cpu().view(torch.int16), expected_pad.cpu().view(torch.int16))

        x_mixed = torch.randn(2, 6, 4, 128, device="mps", dtype=torch.bfloat16)
        grid_mixed = torch.tensor([[1, 2, 3], [1, 3, 2]], dtype=torch.long)
        with patch.dict(os.environ, FASTFRAMES_MPS_ROPE="eager"):
            expected_mix = causal_rope_apply(x_mixed, grid_mixed, freqs, start_frame=1)
        with patch.dict(os.environ, FASTFRAMES_MPS_ROPE="metal"):
            actual_mix = causal_rope_apply(x_mixed, grid_mixed, freqs, start_frame=1)
        assert torch.equal(actual_mix.cpu().view(torch.int16), expected_mix.cpu().view(torch.int16))

    def test_forward_frame_uses_metal_and_matches_bits(self):
        torch.manual_seed(2)
        cfg = WanConfig(dim=256, ffn_dim=512, num_heads=2, num_layers=2)
        model = CausalWanModel(cfg).to("mps", dtype=torch.bfloat16).eval().requires_grad_(False)
        latent = torch.randn(1, 1, 16, 8, 8, device="mps", dtype=torch.bfloat16)
        t = torch.tensor([[833.0]], device="mps")
        e0, e_head = model.time_conditioning(t, torch.bfloat16)
        text = model.text_cache([torch.randn(16, 4096, device="mps", dtype=torch.bfloat16)])
        outputs = []
        for mode in ("eager", "metal"):
            cache = allocate_kv_cache(model, 2, 16, device="mps")
            with patch.dict(os.environ, FASTFRAMES_MPS_ROPE=mode), torch.no_grad():
                flow, feat = model.forward_frame(latent, e0, e_head, text, cache, frame=0)
            outputs.append((flow.cpu().view(torch.int16), feat.cpu().view(torch.int16),
                            cache[0]["k"].cpu().view(torch.int16)))
        for a, b in zip(*outputs):
            assert torch.equal(a, b)
