import torch

from fastframes.wan.causal_model import QuantizedKV, _attended, kv_read, kv_write


def test_roundtrip_error_is_within_half_a_step():
    torch.manual_seed(0)
    x = torch.randn(1, 64, 4, 128, dtype=torch.bfloat16) * 3
    cache = QuantizedKV(x.shape, "cpu")
    kv_write(cache, 0, 64, x)
    back = kv_read(cache, 0, 64)
    assert back.dtype == torch.bfloat16
    # one int8 step is absmax / 127 per token and head; allow BF16 rounding on top
    step = x.float().abs().amax(dim=-1, keepdim=True) / 127
    assert ((back.float() - x.float()).abs() <= step * 0.5 + x.float().abs() / 128).all()


def test_attended_keeps_the_current_frame_exact():
    torch.manual_seed(0)
    prefix, current = torch.randn(1, 8, 2, 16).bfloat16(), torch.randn(1, 4, 2, 16).bfloat16()
    cache = QuantizedKV((1, 12, 2, 16), "cpu")
    kv_write(cache, 0, 8, prefix)
    kv_write(cache, 8, 12, current)
    out = _attended(cache, 8, 12, current)
    assert torch.equal(out[:, 8:], current)
    assert torch.equal(out[:, :8], kv_read(cache, 0, 8))


def test_plain_cache_reads_are_views():
    slot = torch.zeros(1, 6, 2, 4, dtype=torch.bfloat16)
    x = torch.ones(1, 3, 2, 4, dtype=torch.bfloat16)
    kv_write(slot, 3, 6, x)
    assert torch.equal(kv_read(slot, 3, 6), x)
    assert kv_read(slot, 0, 6).data_ptr() == slot.data_ptr()
