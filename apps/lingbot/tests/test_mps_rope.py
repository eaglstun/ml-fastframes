"""Compare the Metal rotation with the eager MPS operations, including BF16 rounding."""
import os
import unittest
from unittest.mock import patch

import torch

from apps.lingbot.dit import rope_real
from apps.lingbot.mps_rope import try_rope
from fastframes.device import limit_mps_memory


class CPURopeTests(unittest.TestCase):
    def test_cpu_never_loads_shader(self):
        value = torch.ones(1, 2, 1, 128, dtype=torch.bfloat16)
        table = torch.zeros(2, 1, 64, 2)
        table[..., 0] = 1.
        with patch("apps.lingbot.mps_rope._kernel", side_effect=AssertionError("unexpected shader load")):
            self.assertIsNone(try_rope(value, table))
            self.assertTrue(torch.equal(rope_real(value, table), value))


@unittest.skipUnless(torch.backends.mps.is_available(), "requires MPS")
class MPSRopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        limit_mps_memory(0.45)

    def check_bits(self, value, table):
        with patch.dict(os.environ, FASTFRAMES_MPS_ROPE="eager"):
            self.assertIsNone(try_rope(value, table))
            expected = rope_real(value, table).cpu().view(torch.int16)
        with patch.dict(os.environ, FASTFRAMES_MPS_ROPE="metal"):
            for _ in range(3):
                result = try_rope(value, table)
                self.assertIsNotNone(result)
                torch.testing.assert_close(result.cpu().view(torch.int16), expected, rtol=0, atol=0)

    def test_batched_and_offset_slices(self):
        for batch in (1, 30):
            # The iPad-assist tail has a nonzero table offset. An odd BF16 offset also checks alignment.
            shape = (batch, 23, 12, 128)
            value = torch.randn(batch * 23 * 12 * 128 + 1, dtype=torch.bfloat16, device="mps")[1:].view(shape)
            table = torch.randn(27, 1, 64, 2, device="mps")[4:]
            self.check_bits(value, table)

    def test_cancellation_requires_two_rounded_products(self):
        # r*c rounds to s in float32, so r*c - 1*s is exactly zero. FMA can expose its rounding error.
        # Swap the two products on token 1, since either side could be contracted.
        value = torch.tensor([[1.0078125, 1.], [1., 1.0078125]], dtype=torch.bfloat16)
        value = value.repeat(1, 64).view(1, 2, 1, 128)
        c = torch.tensor(0.99990004)
        s = c * value.flatten()[0].float()
        table = torch.stack((c, s, s, c)).reshape(2, 2).repeat(1, 64).view(2, 1, 64, 2)
        self.check_bits(value.to("mps"), table.to("mps"))

    def test_signed_zeros_and_range_boundaries(self):
        pairs = [[0., -0.], [-0., 0.], [float("inf"), 1.], [1., float("-inf")],
                 [torch.finfo(torch.bfloat16).tiny, 2**-133], [torch.finfo(torch.bfloat16).max, -1.]]
        value = torch.tensor(pairs, dtype=torch.bfloat16)[:, None, None, :].repeat(1, 1, 1, 64)
        table = torch.tensor([1., 0., 1., -1., 0.75, 0.5, -0., 0.]).repeat(16).view(1, 1, 64, 2)
        self.check_bits(value.to("mps"), table.to("mps"))

    def test_unsupported_layouts_and_gradients_use_eager(self):
        value = torch.randn(1, 7, 2, 128, device="mps", dtype=torch.bfloat16)
        table = torch.randn(7, 1, 64, 2, device="mps")
        for x, t in ((value[:, ::2], table[::2]), (value, table[..., :1].expand_as(table)),
                     (value.float(), table)):
            self.assertIsNone(try_rope(x, t))
            with patch.dict(os.environ, FASTFRAMES_MPS_ROPE="eager"):
                expected = rope_real(x, t)
            self.assertTrue(torch.equal(expected, rope_real(x, t)))
        value.requires_grad_()
        self.assertIsNone(try_rope(value, table))
        self.assertIsNotNone(torch.autograd.grad(rope_real(value, table).float().sum(), value)[0])
