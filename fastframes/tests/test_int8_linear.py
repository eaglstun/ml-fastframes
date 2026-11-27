import pytest
import torch
import torch.nn as nn

from fastframes.wan.int8 import Int8Linear, quantize_blocks

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_int8_linear_matches_bf16(device):
    torch.manual_seed(0)
    linear = nn.Linear(256, 384).to(device, torch.bfloat16)
    x = torch.randn(2, 50, 256, device=device, dtype=torch.bfloat16)
    ref = linear(x).float()
    out = Int8Linear(linear)(x)
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    assert ((out.float() - ref).norm() / ref.norm()).item() < 0.02


def test_int8_linear_casts_fp32_input():
    linear = nn.Linear(64, 32)
    out = Int8Linear(linear)(torch.randn(3, 64))
    assert out.dtype == torch.bfloat16 and out.shape == (3, 32)


def test_quantize_blocks_swaps_only_block_linears():
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.k, self.v, self.o = (nn.Linear(32, 32) for _ in range(4))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.cross_attn = Attn(), Attn()
            self.ffn = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))

    model = nn.Module()
    model.blocks = nn.ModuleList([Block(), Block()])
    model.head = nn.Linear(32, 8)
    quantize_blocks(model)
    block = model.blocks[1]
    assert all(isinstance(getattr(a, n), Int8Linear) for a in (block.self_attn, block.cross_attn) for n in "qkvo")
    assert isinstance(block.ffn[0], Int8Linear) and isinstance(block.ffn[2], Int8Linear)
    assert type(model.head) is nn.Linear


def test_streamed_quantized_load_matches_quantize_blocks(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from fastframes.checkpoints import build_wan, load_state_dict
    from fastframes.wan.causal_model import CausalWanModel
    from fastframes.wan.int8 import quantize_state
    from fastframes.wan.model import WAN_CONFIGS, WanConfig

    monkeypatch.setitem(WAN_CONFIGS, "tiny", WanConfig(dim=64, ffn_dim=128, num_heads=4, num_layers=2))
    torch.manual_seed(0)
    with torch.device("meta"):
        shapes = CausalWanModel(WAN_CONFIGS["tiny"]).state_dict()
    ref = build_wan("tiny", {k: torch.randn(v.shape) * 0.02 for k, v in shapes.items()}, device="cpu")
    path = tmp_path / "dit.safetensors"
    save_file({k: v.contiguous() for k, v in ref.state_dict().items()}, str(path))
    streamed = build_wan("tiny", load_state_dict(path, transform=quantize_state), device="cpu", int8=True)
    expected = quantize_blocks(ref).state_dict()
    assert streamed.state_dict().keys() == expected.keys()
    for name, value in streamed.state_dict().items():
        assert value.dtype == expected[name].dtype and torch.equal(value, expected[name]), name
    assert isinstance(streamed.blocks[1].ffn[2], Int8Linear)


@pytest.mark.parametrize("device", DEVICES)
def test_int8_linear_repeats_exactly(device):
    # Guards against torch._weight_int8pack_mm, whose MPS kernel corrupts a tile now and then.
    torch.manual_seed(0)
    layer = Int8Linear(nn.Linear(512, 1024).to(device, torch.bfloat16))
    x = torch.randn(640, 512, device=device, dtype=torch.bfloat16)
    first = layer(x)
    assert all(torch.equal(layer(x), first) for _ in range(50))
