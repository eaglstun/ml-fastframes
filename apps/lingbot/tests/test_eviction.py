import unittest
from unittest.mock import Mock, patch

import torch

from apps.lingbot import pipeline
from fastframes.device import limit_mps_memory


class EvictionTests(unittest.TestCase):
    def check_evictions(self, device):
        tokens = 5
        shape = (3, 1, 18 * tokens, 2, 8)
        rng = torch.Generator().manual_seed(42)

        def bits(shape):
            return torch.randint(-32768, 32768, shape, dtype=torch.int16, generator=rng)

        expected = [bits(shape), bits(shape)]
        pipe = pipeline.FastFramesLingBot.__new__(pipeline.FastFramesLingBot)
        pipe.cache_k, pipe.cache_v = [x.clone().view(torch.bfloat16).to(device) for x in expected]
        pipe.assist = Mock()
        with patch.object(pipeline, "FRAME_TOKENS", tokens):
            for chunk in range(9):
                pipe.assist.reset_mock()
                pipe._evict(chunk)
                # The first overflow drops two frames, subsequent ones drop four.
                if chunk >= 4:
                    shift = (2 if chunk == 4 else 4) * tokens
                    for x in expected:
                        x[:, :, 6 * tokens:14 * tokens] = x[:, :, 6 * tokens + shift:14 * tokens + shift].clone()
                    pipe.assist.evict.assert_called_once_with(6 * tokens, shift, 8 * tokens)
                else:
                    pipe.assist.evict.assert_not_called()
                for before, actual in zip(expected, (pipe.cache_k, pipe.cache_v)):
                    # Include the sinks, stale tail, and every BF16 bit pattern, including NaNs.
                    self.assertTrue(torch.equal(before, actual.cpu().view(torch.int16)))
                    end = min((chunk + 1) * 4, 18) * tokens
                    update = bits((shape[0], 1, 4 * tokens, 2, 8))
                    before[:, :, end - 4 * tokens:end] = update
                    actual[:, :, end - 4 * tokens:end].copy_(update.view(torch.bfloat16).to(device))

    def test_cpu(self):
        self.check_evictions("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires MPS")
    def test_mps(self):
        limit_mps_memory()
        if pipeline._mps_eviction_kernel() is None:
            self.skipTest("requires runtime Metal shader compilation")
        self.check_evictions("mps")

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires MPS")
    def test_kernel_layout(self):
        limit_mps_memory()
        shape = (3, 1, 90, 2, 8)
        keys = torch.empty(shape, device="mps", dtype=torch.bfloat16)
        values = torch.empty_like(keys)
        self.assertTrue(pipeline._mps_eviction_supported(keys, values))
        for unsupported in (values.float(), values.transpose(-2, -1), values[:, :, 1:], values[..., :4],
                            torch.empty(keys.numel() + 1, device="mps", dtype=keys.dtype)[1:].view(shape)):
            self.assertFalse(pipeline._mps_eviction_supported(unsupported, unsupported))
            self.assertFalse(pipeline._mps_eviction_supported(keys, unsupported))


if __name__ == "__main__":
    unittest.main()
