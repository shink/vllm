import torch
import triton
import triton.language as tl
from vllm.attention.ops.triton_decode_attention import decode_attention_fwd

try:
    import torch_npu
except ImportError:
    pass


def fn():
    d = "cuda" if torch.cuda.is_available() else "npu"
    torch.set_default_device(d)

    # Create dummy data
    x1 = torch.randn((5, 32, 576), dtype=torch.bfloat16)
    x2 = torch.randn((1024, 16, 8, 576), dtype=torch.bfloat16)
    x3 = torch.randn((1024, 16, 8, 512), dtype=torch.bfloat16)
    x4 = torch.randn((5, 32, 512), dtype=torch.bfloat16)
    x5 = torch.ones((5, 65, 1), dtype=torch.int64)
    x6 = torch.ones((5,), dtype=torch.int64)
    x7 = torch.randn((5, 32, 8, 513), dtype=torch.float32)
    x8 = 8
    x9 = 0.041666666666666664
    x9 = 16
    args = [x1, x2, x3, x4, x5, x6, x7, x8, x9]

    # Call the kernel
    print("Triton kernel start...")
    decode_attention_fwd(*args)
    print("Triton kernel success")


if __name__ == "__main__":
    fn()
