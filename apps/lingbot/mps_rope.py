"""Single-dispatch real RoPE for contiguous BF16 LingBot inference tensors."""
from fastframes.wan.mps_rope import _kernel
from fastframes.wan.mps_rope import try_rope as _try_rope


def try_rope(value, table):
    """Return None for layouts, precisions or PyTorch versions that need eager rotation."""
    return _try_rope(value, table, kernel_fn=_kernel)
