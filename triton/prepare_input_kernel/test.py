import torch
import triton
import triton.language as tl
from vllm.v1.spec_decode.eagle import EagleProposer

try:
    import torch_npu
except ImportError:
    pass


def fn():
    d = "cuda" if torch.cuda.is_available() else "npu"
    device = torch.device(d)
    drafter = EagleProposer(None, device)

    # Create dummy data
    # [batch_size + 1]
    cu_target_query_lens = torch.ones(11, device=device, dtype=torch.int64)
    # [batch_size]
    num_rejected_tokens = torch.zeros(10, device=device, dtype=torch.int64)

    # Call the kernel
    print("Triton kernel start...")
    drafter.prepare_inputs(cu_target_query_lens, num_rejected_tokens)
    print("Triton kernel success")


if __name__ == "__main__":
    fn()
