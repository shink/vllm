import torch
import triton
import triton.language as tl
from vllm.model_executor.layers.mamba.ops import ssd_state_passing

try:
    import torch_npu
except ImportError:
    pass

device = "cuda" if torch.cuda.is_available() else "npu"


def fn():
    # Example parameters
    batch_size = 2
    num_chunks = 4
    chunk_size = 8
    num_heads = 16
    dim = 32

    # Create dummy data
    torch.set_default_device(device)
    states = torch.randn(batch_size, num_chunks, num_heads, dim)
    dA_chunk_cumsum = torch.randn(batch_size, num_heads, num_chunks)

    # Call the kernel
    print("Triton kernel start...")
    ssd_state_passing._state_passing_fwd(states, dA_chunk_cumsum)
    print("Triton kernel success")


if __name__ == "__main__":
    fn()
