# Triton Ops in vLLM

> - 60 @triton.jit
>
> - 23 files

### vllm/attention/ops/chunked_prefill_paged_decode.py

- [x] `cdiv_fn`
- [x] `kernel_paged_attention_2d`

### vllm/attention/ops/prefix_prefill.py

- [x] `_fwd_kernel`
- [ ] `_fwd_kernel_flash_attn_v2`
- [x] `_fwd_kernel_alibi`

### vllm/attention/ops/triton_decode_attention.py

- [x] `tanh`
- [x] `_fwd_kernel_stage1`
- [x] `_fwd_grouped_kernel_stage1`
- [x] `_fwd_kernel_stage2`

### vllm/attention/ops/triton_flash_attention.py

- [x] `cdiv_fn`
- [x] `max_fn`
- [x] `dropout_offsets`
- [x] `dropout_rng`
- [x] `dropout_mask`
- [x] `load_fn`
- [x] `_attn_fwd_inner`
- [x] `attn_fwd`

### vllm/attention/ops/triton_merge_attn_states.py

- [x] `merge_attn_states_kernel`

### vllm/attention/ops/blocksparse_attention/blocksparse_attention_kernel.py

- [x] `_fwd_kernel_inner`
- [x] `_fwd_kernel_batch_inference`

### vllm/lora/ops/triton_ops/kernel_utils.py

- [x] `mm_k`
- [x] `do_expand_kernel`
- [x] `do_shrink_kernel`

### vllm/lora/ops/triton_ops/lora_expand.py

- [x] `_lora_expand_kernel`

### vllm/lora/ops/triton_ops/lora_shrink.py

- [x] `_lora_shrink_kernel`

### vllm/model_executor/layers/lightning_attn.py

- [x] `_fwd_diag_kernel`
- [x] `_fwd_kv_parallel`
- [x] `_fwd_kv_reduce`
- [x] `_fwd_none_diag_kernel`
- [x] `_linear_attn_decode_kernel`

### vllm/model_executor/layers/fused_moe/fused_moe.py

- [x] `write_zeros_to_output`
- [x] `fused_moe_kernel_gptq_awq`
- [x] `fused_moe_kernel`

### vllm/model_executor/layers/fused_moe/moe_align_block_size.py

- [x] `moe_align_block_size_stage1`
- [x] `moe_align_block_size_stage2`
- [x] `moe_align_block_size_stage3`
- [x] `moe_align_block_size_stage4`

### vllm/model_executor/layers/mamba/ops/mamba_ssm.py

> tests/kernels/test_mamba_ssm.py

- [x] `softplus`
- [x] `_selective_scan_update_kernel`

### vllm/model_executor/layers/mamba/ops/ssd_bmm.py

- [x] `_bmm_chunk_fwd_kernel`

### vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py

- [x] `_chunk_scan_fwd_kernel`

### vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py

- [x] `_chunk_cumsum_fwd_kernel`
- [x] `_chunk_state_fwd_kernel`
- [x] `_chunk_state_varlen_kernel`

### vllm/model_executor/layers/mamba/ops/ssd_state_passing.py

- [x] `_state_passing_fwd_kernel`

### vllm/model_executor/layers/quantization/awq_triton.py

- [x] `awq_dequantize_kernel`
- [x] `awq_gemm_kernel`

### vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py

- [x] `scaled_mm_kernel`

### vllm/model_executor/layers/quantization/utils/fp8_utils.py

- [x] `_per_token_group_quant_fp8`
- [x] `_per_token_group_quant_fp8_colmajor`
- [x] `_w8a8_block_fp8_matmul`

### vllm/model_executor/layers/quantization/utils/int8_utils.py

- [x] `_per_token_quant_int8`
- [x] `_per_token_group_quant_int8`
- [x] `_w8a8_block_int8_matmul`

### vllm/v1/sample/rejection_sampler.py

- [x] `rejection_greedy_sample_kernel`
- [x] `rejection_random_sample_kernel`
- [x] `expand_kernel`
- [x] `sample_recovered_tokens_kernel`

### vllm/v1/spec_decode/eagle.py

- [x] `prepare_input_kernel`
