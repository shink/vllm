import torch
import triton
import triton.language as tl
from vllm.v1.spec_decode.eagle import EagleProposer

try:
    import torch_npu
except ImportError:
    pass

from vllm.attention.ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode


def fn():
    device = "cuda" if torch.cuda.is_available() else "npu"

    # Create dummy data
    torch.set_default_device(device)
    x1 = torch.randn((5337, 64, 128), dtype=torch.float16)
    x2 = torch.randn((5337, 64, 128), dtype=torch.float16)
    x3 = torch.randn((5337, 64, 128), dtype=torch.float16)
    x4 = torch.randn((5337, 64, 128), dtype=torch.float16)
    x5 = "auto"
    x6 = torch.randn((640, 64, 16, 32, 8), dtype=torch.float16)
    x7 = torch.randn((640, 64, 128, 32), dtype=torch.float16)
    # RuntimeError: "normal_kernel_cuda" not implemented for 'Long'
    x8 = torch.ones((10, 64), dtype=torch.int64)
    x9 = torch.ones((11,), dtype=torch.int64)
    x10 = torch.ones((10,), dtype=torch.int64)
    x11 = 1024
    x12 = 1
    x13 = torch.randn((), dtype=torch.float32)
    x14 = torch.randn((), dtype=torch.float32)
    args = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]
    kwargs = {"sliding_window": 0}

    # Call the kernel
    print("Triton kernel start...")
    chunked_prefill_paged_decode(*args, **kwargs)
    print("Triton kernel success")


if __name__ == "__main__":
    fn()
