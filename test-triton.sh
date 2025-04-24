#!/bin/bash

set -ex

# torch._inductor.config.compile_threads = 1
export TORCHINDUCTOR_COMPILE_THREADS=1

# scaled_mm_kernel
# vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py
pytest -svx tests/kernels/test_triton_scaled_mm.py::test_scaled_mm

# awq_dequantize_kernel
# vllm/model_executor/layers/quantization/awq_triton.py
pytest -svx tests/kernels/test_awq_triton.py::test_dequantize

# awq_gemm_kernel
# vllm/model_executor/layers/quantization/awq_triton.py
pytest -svx tests/kernels/test_awq_triton.py::test_gemm

# tanh
# _fwd_kernel_stage1
# _fwd_grouped_kernel_stage1
# _fwd_kernel_stage2
pytest -svx tests/kernels/test_triton_decode_attention.py::test_decode_attention

# cdiv_fn
# kernel_paged_attention_2d
# _fwd_kernel
# _fwd_kernel_flash_attn_v2
# _fwd_kernel_alibi
pytest -svx tests/kernels/test_prefix_prefill.py

# merge_attn_states_kernel
pytest -svx tests/kernels/test_cascade_flash_attn.py::test_merge_kernel

# cdiv_fn
# max_fn
# dropout_offsets
# dropout_rng
# dropout_mask
# load_fn
# _attn_fwd_inner
# attn_fwd
# vllm/attention/ops/triton_flash_attention.py
