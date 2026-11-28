"""Verify MPS fused RMSNorm accuracy, strided QKV slice handling, and fallbacks."""
import os
from unittest.mock import patch

import pytest
import torch

from apps.lingbot.dit import rms_norm
from fastframes.device import limit_mps_memory
from fastframes.wan.model import WanRMSNorm
from fastframes.wan.mps_rmsnorm import try_rms_norm


def test_cpu_never_loads_shader():
    mod = WanRMSNorm(128, eps=1e-6).to(torch.bfloat16)
    x = torch.randn(2, 4, 128, dtype=torch.bfloat16)
    with patch("fastframes.wan.mps_rmsnorm._kernel", side_effect=AssertionError("unexpected shader load")):
        assert try_rms_norm(x, mod.weight, mod.eps) is None
        out = mod(x)
        assert out.shape == x.shape and out.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
class TestMPSRMSNorm:
    @classmethod
    def setup_class(cls):
        limit_mps_memory(0.45)

    @pytest.mark.parametrize("dim", [128, 1536, 5120])
    def test_contiguous_and_strided_qkv_slices_match_cpu_fp64_and_eager(self, dim):
        torch.manual_seed(0)
        mod_cpu = WanRMSNorm(dim, eps=1e-6).double()
        with torch.no_grad():
            mod_cpu.weight.copy_(torch.randn(dim, dtype=torch.float64) * 0.2 + 1.0)
        mod = WanRMSNorm(dim, eps=1e-6).to("mps", dtype=torch.bfloat16)
        with torch.no_grad():
            mod.weight.copy_(mod_cpu.weight.to(torch.bfloat16))

        # Test strided .chunk(3, -1) views like LingBot QKV under no_grad (inference)!
        qkv = torch.randn(1, 37, 3 * dim, device="mps", dtype=torch.bfloat16)
        for part in qkv.chunk(3, dim=-1):
            with torch.no_grad(), patch.dict(os.environ, FASTFRAMES_MPS_RMSNORM="eager"):
                eager = rms_norm(part, mod)
            with torch.no_grad(), patch.dict(os.environ, FASTFRAMES_MPS_RMSNORM="metal"):
                metal = try_rms_norm(part, mod.weight, mod.eps)
                via_mod = mod(part)
                repeat = try_rms_norm(part, mod.weight, mod.eps)
            assert metal is not None
            assert torch.equal(metal.cpu().view(torch.int16), via_mod.cpu().view(torch.int16))
            assert torch.equal(metal.cpu().view(torch.int16), repeat.cpu().view(torch.int16))
            # Check max difference <= 1 BF16 ULP vs eager and closer/equal vs CPU FP64 reference!
            x64 = part.cpu().double()
            w64 = mod.weight.cpu().double()
            ref64 = (x64 * torch.rsqrt(x64.square().mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).double() * w64
            ref_bf16 = ref64.to(torch.bfloat16)
            ulp_diff = (metal.cpu().view(torch.int16).int() - ref_bf16.view(torch.int16).int()).abs().max().item()
            eager_ulp = (eager.cpu().view(torch.int16).int() - ref_bf16.view(torch.int16).int()).abs().max().item()
            assert ulp_diff <= 1
            assert ulp_diff <= max(1, eager_ulp)

    def test_unsupported_layouts_and_autograd_fallback(self):
        mod = WanRMSNorm(128, eps=1e-6).to("mps", dtype=torch.bfloat16)
        x = torch.randn(2, 6, 128, device="mps", dtype=torch.bfloat16)
        # Non-contiguous batch>1 3D slice falls back cleanly
        strided_b2 = torch.randn(2, 6, 256, device="mps", dtype=torch.bfloat16)[:, :, :128]
        assert try_rms_norm(strided_b2, mod.weight, mod.eps) is None
        with patch.dict(os.environ, FASTFRAMES_MPS_RMSNORM="eager"):
            expected = mod(strided_b2)
        assert torch.equal(mod(strided_b2).cpu().view(torch.int16), expected.cpu().view(torch.int16))

        # FP32 input falls back cleanly
        assert try_rms_norm(x.float(), mod.weight, mod.eps) is None

        # Autograd falls back cleanly
        xg = x.detach().clone().requires_grad_(True)
        assert try_rms_norm(xg, mod.weight, mod.eps) is None
        assert torch.autograd.grad(mod(xg).float().sum(), xg)[0] is not None
