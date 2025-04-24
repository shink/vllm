# Triton Ops in vLLM

> https://github.com/shink/vllm/blob/jyh/triton/triton-ascend.md

> - 60 @triton.jit
>
> - 23 files

## Issues

1. NPUOptions

```
E           AttributeError: 'NPUOptions' object has no attribute 'default_dot_input_precision'. Did you mean: 'allowed_dot_input_precisions'?

triton-ascend/triton/python/triton/language/semantic.py:1498: AttributeError
```

解决方法：ascend/backend/compiler.py 中添加 `default_dot_input_precision`:

```python
@dataclass(frozen=True)
class NPUOptions:
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    default_dot_input_precision: str = "ieee"
```

## Ops

### vllm/attention/ops/chunked_prefill_paged_decode.py

- [x] `cdiv_fn`
- [x] `kernel_paged_attention_2d`

```python
pytest -svx tests/kernels/attention/test_prefix_prefill.py -k "chunked_prefill_paged_decode"
```

<details>
<summary>HuaweiCompilationError</summary>

```
E               huawei.HuaweiCompilationError:
E               ///------------------[ERROR][TritonAscend][BEG]------------------
E               [ConvertTritonIRToLinalgIR] encounters error:
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:226:42: remark: [MaskState] Unsupported cmpi scenario
E                           qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
E                                                        ^
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:226:42: note: see current operation: %152 = arith.cmpi sge, %122, %151 : tensor<64x64xi32>
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:226:42: remark: [MaskState] Unsupported cmpi scenario
E                           qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
E                                                        ^
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:226:42: note: see current operation: %141 = arith.cmpi sge, %112, %140 : tensor<64x64xi32>
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:122:21: error: cannot div 0!
E                                        ((start_n + offs_n) // block_size) * stride_b_loc_s,
E                                   ^
E               PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
E               Stack dump:
E               0.      Program arguments: /home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmp329mu6w0/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmp329mu6w0/kernel.ttadapter.mlir
E                #0 0x0000aaaac0144150 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfa4150)
E                #1 0x0000aaaac0141c00 llvm::sys::RunSignalHandlers() (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfa1c00)
E                #2 0x0000aaaac0141d48 SignalHandler(int) Signals.cpp:0:0
E                #3 0x0000ffffa1d0a7c0 (linux-vdso.so.1+0x7c0)
E                #4 0x0000aaaabf47dfd8 mlir::addOpFoldResult(mlir::OpFoldResult const&, mlir::OpFoldResult const&, mlir::Location const&, mlir::OpBuilder&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x2ddfd8)
E                #5 0x0000aaaabf30c008 mlir::triton::BlockData::addBlock(mlir::triton::BlockData&, mlir::triton::BlockData&, mlir::Location, mlir::ConversionPatternRewriter&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16c008)
E                #6 0x0000aaaabf30c224 mlir::triton::BlockDataParser::parseAddPtr(mlir::triton::AddPtrOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16c224)
E                #7 0x0000aaaabf31184c mlir::triton::BlockDataParser::rewriteAddPtr(mlir::triton::AddPtrOp, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>>&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x17184c)
E                #8 0x0000aaaabf2f3210 LoadStoreConverter::AddPtrConverter::matchAndRewrite(mlir::triton::AddPtrOp, mlir::triton::AddPtrOpAdaptor, mlir::ConversionPatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x153210)
E                #9 0x0000aaaabf2df170 mlir::OpConversionPattern<mlir::triton::AddPtrOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::Value>, mlir::ConversionPatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x13f170)
E               #10 0x0000aaaabfda6b2c mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc06b2c)
E               #11 0x0000aaaabfdcd6f0 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc2d6f0)
E               #12 0x0000aaaabfdaa964 (anonymous namespace)::OperationLegalizer::legalize(mlir::Operation*, mlir::ConversionPatternRewriter&) DialectConversion.cpp:0:0
E               #13 0x0000aaaabfdaade8 mlir::OperationConverter::convert(mlir::ConversionPatternRewriter&, mlir::Operation*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc0ade8)
E               #14 0x0000aaaabfdb03c8 mlir::OperationConverter::convertOperations(llvm::ArrayRef<mlir::Operation*>) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc103c8)
E               #15 0x0000aaaabfdb1134 mlir::applyPartialConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc11134)
E               #16 0x0000aaaabf2d6370 (anonymous namespace)::TritonToLinalgPass::runOnOperation() TritonToLinalgPass.cpp:0:0
E               #17 0x0000aaaabfd6c5b4 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbcc5b4)
E               #18 0x0000aaaabfd6ca30 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbcca30)
E               #19 0x0000aaaabfd6d8f8 mlir::PassManager::run(mlir::Operation*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbcd8f8)
E               #20 0x0000aaaabfd61678 performActions(llvm::raw_ostream&, std::shared_ptr<llvm::SourceMgr> const&, mlir::MLIRContext*, mlir::MlirOptMainConfig const&) MlirOptMain.cpp:0:0
E               #21 0x0000aaaabfd61cfc processBuffer(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::MlirOptMainConfig const&, mlir::DialectRegistry&, llvm::ThreadPoolInterface*) MlirOptMain.cpp:0:0
E               #22 0x0000aaaabfd61e38 llvm::LogicalResult llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>::callback_fn<mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&)::'lambda'(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>(long, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&) MlirOptMain.cpp:0:0
E               #23 0x0000aaaac00d64f8 mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>, llvm::raw_ostream&, llvm::StringRef, llvm::StringRef) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xf364f8)
E               #24 0x0000aaaabfd5bcfc mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbbbcfc)
E               #25 0x0000aaaabfd61f48 mlir::MlirOptMain(int, char**, llvm::StringRef, llvm::StringRef, mlir::DialectRegistry&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbc1f48)
E               #26 0x0000aaaabfd6233c mlir::MlirOptMain(int, char**, llvm::StringRef, mlir::DialectRegistry&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbc233c)
E               #27 0x0000aaaabf2ad108 main (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x10d108)
E               #28 0x0000ffffa17773fc __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
E               #29 0x0000ffffa17774cc call_init ./csu/../csu/libc-start.c:128:20
E               #30 0x0000ffffa17774cc __libc_start_main ./csu/../csu/libc-start.c:379:5
E               #31 0x0000aaaabf2cca30 _start (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x12ca30)
E               ///------------------[ERROR][TritonAscend][END]------------------
```

</details>

### vllm/attention/ops/prefix_prefill.py

- [x] `_fwd_kernel`
- [ ] `_fwd_kernel_flash_attn_v2`
- [x] `_fwd_kernel_alibi`

```python
pytest -svx tests/kernels/test_prefix_prefill.py -k "context_attention_fwd"
```

<details>
<summary>HuaweiCompilationError</summary>

```
E               huawei.HuaweiCompilationError:
E               ///------------------[ERROR][TritonAscend][BEG]------------------
E               [ConvertTritonIRToLinalgIR] encounters error:
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:226:42: remark: [MaskState] Unsupported cmpi scenario
E                           qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
E                                                        ^
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:226:42: note: see current operation: %151 = arith.cmpi sge, %121, %150 : tensor<64x64xi32>
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:226:42: remark: [MaskState] Unsupported cmpi scenario
E                           qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
E                                                        ^
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:226:42: note: see current operation: %140 = arith.cmpi sge, %111, %139 : tensor<64x64xi32>
E               /home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:122:21: error: cannot div 0!
E                                        ((start_n + offs_n) // block_size) * stride_b_loc_s,
E                                   ^
E               PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
E               Stack dump:
E               0.      Program arguments: /home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmp44t4v3y9/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmp44t4v3y9/kernel.ttadapter.mlir
E                #0 0x0000aaaada0b4150 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfa4150)
E                #1 0x0000aaaada0b1c00 llvm::sys::RunSignalHandlers() (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfa1c00)
E                #2 0x0000aaaada0b1d48 SignalHandler(int) Signals.cpp:0:0
E                #3 0x0000ffff92aa27c0 (linux-vdso.so.1+0x7c0)
E                #4 0x0000aaaad93edfd8 mlir::addOpFoldResult(mlir::OpFoldResult const&, mlir::OpFoldResult const&, mlir::Location const&, mlir::OpBuilder&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x2ddfd8)
E                #5 0x0000aaaad927c008 mlir::triton::BlockData::addBlock(mlir::triton::BlockData&, mlir::triton::BlockData&, mlir::Location, mlir::ConversionPatternRewriter&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16c008)
E                #6 0x0000aaaad927c224 mlir::triton::BlockDataParser::parseAddPtr(mlir::triton::AddPtrOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16c224)
E                #7 0x0000aaaad928184c mlir::triton::BlockDataParser::rewriteAddPtr(mlir::triton::AddPtrOp, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>>&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x17184c)
E                #8 0x0000aaaad9263210 LoadStoreConverter::AddPtrConverter::matchAndRewrite(mlir::triton::AddPtrOp, mlir::triton::AddPtrOpAdaptor, mlir::ConversionPatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x153210)
E                #9 0x0000aaaad924f170 mlir::OpConversionPattern<mlir::triton::AddPtrOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::Value>, mlir::ConversionPatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x13f170)
E               #10 0x0000aaaad9d16b2c mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc06b2c)
E               #11 0x0000aaaad9d3d6f0 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc2d6f0)
E               #12 0x0000aaaad9d1a964 (anonymous namespace)::OperationLegalizer::legalize(mlir::Operation*, mlir::ConversionPatternRewriter&) DialectConversion.cpp:0:0
E               #13 0x0000aaaad9d1ade8 mlir::OperationConverter::convert(mlir::ConversionPatternRewriter&, mlir::Operation*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc0ade8)
E               #14 0x0000aaaad9d203c8 mlir::OperationConverter::convertOperations(llvm::ArrayRef<mlir::Operation*>) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc103c8)
E               #15 0x0000aaaad9d21134 mlir::applyPartialConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc11134)
E               #16 0x0000aaaad9246370 (anonymous namespace)::TritonToLinalgPass::runOnOperation() TritonToLinalgPass.cpp:0:0
E               #17 0x0000aaaad9cdc5b4 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbcc5b4)
E               #18 0x0000aaaad9cdca30 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbcca30)
E               #19 0x0000aaaad9cdd8f8 mlir::PassManager::run(mlir::Operation*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbcd8f8)
E               #20 0x0000aaaad9cd1678 performActions(llvm::raw_ostream&, std::shared_ptr<llvm::SourceMgr> const&, mlir::MLIRContext*, mlir::MlirOptMainConfig const&) MlirOptMain.cpp:0:0
E               #21 0x0000aaaad9cd1cfc processBuffer(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::MlirOptMainConfig const&, mlir::DialectRegistry&, llvm::ThreadPoolInterface*) MlirOptMain.cpp:0:0
E               #22 0x0000aaaad9cd1e38 llvm::LogicalResult llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>::callback_fn<mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&)::'lambda'(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>(long, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&) MlirOptMain.cpp:0:0
E               #23 0x0000aaaada0464f8 mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>, llvm::raw_ostream&, llvm::StringRef, llvm::StringRef) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xf364f8)
E               #24 0x0000aaaad9ccbcfc mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbbbcfc)
E               #25 0x0000aaaad9cd1f48 mlir::MlirOptMain(int, char**, llvm::StringRef, llvm::StringRef, mlir::DialectRegistry&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbc1f48)
E               #26 0x0000aaaad9cd233c mlir::MlirOptMain(int, char**, llvm::StringRef, mlir::DialectRegistry&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbc233c)
E               #27 0x0000aaaad921d108 main (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x10d108)
E               #28 0x0000ffff925173fc __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
E               #29 0x0000ffff925174cc call_init ./csu/../csu/libc-start.c:128:20
E               #30 0x0000ffff925174cc __libc_start_main ./csu/../csu/libc-start.c:379:5
E               #31 0x0000aaaad923ca30 _start (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x12ca30)
E               ///------------------[ERROR][TritonAscend][END]------------------
```

</details>

### vllm/attention/ops/triton_decode_attention.py

- [x] `tanh`
- [x] `_fwd_kernel_stage1`
- [x] `_fwd_grouped_kernel_stage1`
- [x] `_fwd_kernel_stage2`

```python
pytest -svx tests/kernels/attention/test_triton_decode_attention.py
```

<details>
<summary>HuaweiCompilationError</summary>

```
E               huawei.HuaweiCompilationError:
E               ///------------------[ERROR][TritonAscend][BEG]------------------
E               [ConvertTritonIRToLinalgIR] encounters error:
E               PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
E               Stack dump:
E               0.      Program arguments: /home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmp5r9cvwnj/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmp5r9cvwnj/kernel.ttadapter.mlir
E               #0 0x0000aaaaba374150 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfa4150)
E               #1 0x0000aaaaba371c00 llvm::sys::RunSignalHandlers() (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfa1c00)
E               #2 0x0000aaaaba371d48 SignalHandler(int) Signals.cpp:0:0
E               #3 0x0000ffff972f97c0 (linux-vdso.so.1+0x7c0)
E               #4 0x0000ffff96dd7d64 ./string/../sysdeps/aarch64/multiarch/../memcpy.S:261:0
E               #5 0x0000aaaab953da48 mlir::triton::BlockDataParser::parseExpandDims(mlir::triton::ExpandDimsOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16da48)
E               #6 0x0000ffffd0cded50
E               ///------------------[ERROR][TritonAscend][END]------------------
```

</details>

### vllm/attention/ops/triton_flash_attention.py

- [x] `cdiv_fn`
- [x] `max_fn`
- [x] `dropout_offsets`
- [x] `dropout_rng`
- [x] `dropout_mask`
- [x] `load_fn`
- [x] `_attn_fwd_inner`
- [x] `attn_fwd`

> 无测试用例

### vllm/attention/ops/triton_merge_attn_states.py

- [x] `merge_attn_states_kernel`

```python
pytest -svx tests/kernels/test_cascade_flash_attn.py
```

<details>
<summary>vllm-ascend 问题</summary>

```
______________________________________________ ERROR collecting tests/kernels/test_cascade_flash_attn.py ______________________________________________
ImportError while importing test module '/home/devuser/workspace/vllm-project/vllm/tests/kernels/test_cascade_flash_attn.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../.conda/envs/triton/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/kernels/test_cascade_flash_attn.py:12: in <module>
    from vllm.vllm_flash_attn import (fa_version_unsupported_reason,
E   ImportError: cannot import name 'fa_version_unsupported_reason' from 'vllm.vllm_flash_attn' (unknown location)
```

</details>

### vllm/attention/ops/blocksparse_attention/blocksparse_attention_kernel.py

- [x] `_fwd_kernel_inner`
- [x] `_fwd_kernel_batch_inference`

> 无测试用例

### vllm/lora/ops/triton_ops/kernel_utils.py

- [x] `mm_k`
- [x] `do_expand_kernel`
- [x] `do_shrink_kernel`

> 复用 `test_punica_ops.py` 测试用例

### vllm/lora/ops/triton_ops/lora_expand.py

- [x] `_lora_expand_kernel`

```python
pytest -svx tests/lora/test_punica_ops.py -k "expand"
```

<details>
<summary>vllm-ascend 问题</summary>

```
The operator 'vllm::lora_expand' is not currently supported on the NPU backend and will fall back to run on the CPU.
```

```
E       NotImplementedError: Could not run 'vllm::lora_expand' with arguments from the 'CPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build).
```

</details>

### vllm/lora/ops/triton_ops/lora_shrink.py

- [x] `_lora_shrink_kernel`

```python
pytest -svx tests/lora/test_punica_ops.py -k "shrink"
```

<details>
<summary>vllm-ascend 问题</summary>

```
Warning: CAUTION: The operator 'vllm::lora_shrink' is not currently supported on the NPU backend and will fall back to run on the CPU.
```

```
E       NotImplementedError: Could not run 'vllm::lora_shrink' with arguments from the 'CPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build).
```

</details>

### vllm/model_executor/layers/lightning_attn.py

- [x] `_fwd_diag_kernel`
- [x] `_fwd_kv_parallel`
- [x] `_fwd_kv_reduce`
- [x] `_fwd_none_diag_kernel`
- [x] `_linear_attn_decode_kernel`

```python
pytest -svx tests/kernels/test_lightning_attn.py
```

<details>
<summary>HuaweiCompilationError</summary>

```
E               huawei.HuaweiCompilationError:
E               ///------------------[ERROR][TritonAscend][BEG]------------------
E               [ConvertTritonIRToLinalgIR] encounters error:
E               /home/devuser/workspace/vllm-project/vllm/vllm/model_executor/layers/lightning_attn.py:507:0: error: 'func.return' op has 1 operands, but enclosing function (@_linear_attn_decode_kernel) returns 0
E               /home/devuser/workspace/vllm-project/vllm/vllm/model_executor/layers/lightning_attn.py:507:0: note: see current operation: "func.return"(%9) : (i1) -> ()
E               ///------------------[ERROR][TritonAscend][END]------------------
```

</details>

### vllm/model_executor/layers/fused_moe/fused_moe.py

- [x] `write_zeros_to_output`
- [x] `fused_moe_kernel_gptq_awq`
- [x] `fused_moe_kernel`

> 无测试用例

<details>
<summary>HuaweiCompilationError</summary>

</details>

### vllm/model_executor/layers/fused_moe/moe_align_block_size.py

- [x] `moe_align_block_size_stage1`
- [x] `moe_align_block_size_stage2`
- [x] `moe_align_block_size_stage3`
- [x] `moe_align_block_size_stage4`

> 无测试用例

<details>
<summary>HuaweiCompilationError</summary>

</details>

### vllm/model_executor/layers/mamba/ops/mamba_ssm.py

- [x] `softplus`
- [x] `_selective_scan_update_kernel`

```python
pytest -svx tests/kernels/test_mamba_ssm.py
```

<details>
<summary>PyTorch 问题</summary>

```
E               AttributeError: '_OpNamespace' '_C' object has no attribute 'selective_scan_fwd'

../../../.conda/envs/triton/lib/python3.10/site-packages/torch/_ops.py:1225: AttributeError
```

</details>

### vllm/model_executor/layers/mamba/ops/ssd_bmm.py

- [x] `_bmm_chunk_fwd_kernel`

> 无测试用例

<details>
<summary></summary>

</details>

### vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py

- [x] `_chunk_scan_fwd_kernel`

> 无测试用例

<details>
<summary></summary>

</details>

### vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py

- [x] `_chunk_cumsum_fwd_kernel`
- [x] `_chunk_state_fwd_kernel`
- [x] `_chunk_state_varlen_kernel`

> 无测试用例

<details>
<summary></summary>

</details>

### vllm/model_executor/layers/mamba/ops/ssd_state_passing.py

- [x] `_state_passing_fwd_kernel`

```python
python triton/test_state_passing_fwd.py
```

<details>
<summary>HuaweiCompilationError</summary>

```
huawei.HuaweiCompilationError:
///------------------[ERROR][TritonAscend][BEG]------------------
[ConvertLinalgRToBinary] encounters error:
error: strides must not be zero
npuc: /usr1/BiSheng/mlir/include/mlir/IR/StorageUniquerSupport.h:181: static ConcreteT mlir::detail::StorageUserBase<mlir::StridedLayoutAttr, mlir::Attribute, mlir::detail::StridedLayoutAttrStorage, mlir::detail::AttributeUniquer, Trait>::get(mlir::MLIRContext *, Args &&...) [ConcreteT = mlir::StridedLayoutAttr, BaseT = mlir::Attribute, StorageT = mlir::detail::StridedLayoutAttrStorage, UniquerT = mlir::detail::AttributeUniquer, Traits = <Trait>, Args = <long, llvm::ArrayRef<long>>]: Assertion `succeeded(ConcreteT::verify(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.      Program arguments: /home/devuser/program/bisheng/compiler/npuc /tmp/tmpc0e0vex4/kernel.ttadapter.mlir --enable-auto-multi-buffer=true -o /tmp/tmpc0e0vex4/kernel
```

</details>

### vllm/model_executor/layers/quantization/awq_triton.py

- [x] `awq_dequantize_kernel`
- [x] `awq_gemm_kernel`

```python
pytest -svx tests/kernels/test_awq_triton.py
```

<details>
<summary>HuaweiCompilationError</summary>

```
E               huawei.HuaweiCompilationError:
E               ///------------------[ERROR][TritonAscend][BEG]------------------
E               [ConvertLinalgRToBinary] encounters error:
E               loc("/tmp/tmp8ijeyefm/kernel.ttadapter.mlir":94:11): error: 'hivm.hir.vshr' op failed to verify that operand at index 1 is scalar-only
E               loc("/tmp/tmp8ijeyefm/kernel.ttadapter.mlir":2:1): error: run BiShengHIR pipeline failed
E
E               loc("/tmp/tmp8ijeyefm/kernel.ttadapter.mlir":94:11): error: 'hivm.hir.vshr' op failed to verify that operand at index 1 is scalar-only
E               loc("/tmp/tmp8ijeyefm/kernel.ttadapter.mlir":2:1): error: run BiShengHIR pipeline failed
E
E               loc("/tmp/tmp8ijeyefm/kernel.ttadapter.mlir":94:11): error: 'hivm.hir.vshr' op failed to verify that operand at index 1 is scalar-only
E               loc("/tmp/tmp8ijeyefm/kernel.ttadapter.mlir":2:1): error: run BiShengHIR pipeline failed
E
E               loc("/tmp/tmp8ijeyefm/kernel.ttadapter.mlir":94:11): error: 'hivm.hir.vshr' op failed to verify that operand at index 1 is scalar-only
E               loc("/tmp/tmp8ijeyefm/kernel.ttadapter.mlir":2:1): error: run BiShengHIR pipeline failed
E
E               loc("/tmp/tmp8ijeyefm/kernel.ttadapter.mlir":94:11): error: 'hivm.hir.vshr' op failed to verify that operand at index 1 is scalar-only
E               loc("/tmp/tmp8ijeyefm/kernel.ttadapter.mlir":2:1): error: run BiShengHIR pipeline failed
E
E               Error run BiShengIR pipeline pipeline
E               ///------------------[ERROR][TritonAscend][END]------------------
```

</details>

### vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py

- [x] `scaled_mm_kernel`

```python
pytest -svx tests/kernels/test_triton_scaled_mm.py
```

<details>
<summary>HuaweiCompilationError</summary>

```
E               huawei.HuaweiCompilationError:
E               ///------------------[ERROR][TritonAscend][BEG]------------------
E               [ConvertTritonIRToLinalgIR] encounters error:
E               PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
E               Stack dump:
E               0.      Program arguments: /home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmp_ma9v6t0/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmp_ma9v6t0/kernel.ttadapter.mlir
E                #0 0x0000aaaab6d72400 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfa2400)
E                #1 0x0000aaaab6d6feb0 llvm::sys::RunSignalHandlers() (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xf9feb0)
E                #2 0x0000aaaab6d6fff8 SignalHandler(int) Signals.cpp:0:0
```

</details>

### vllm/model_executor/layers/quantization/utils/fp8_utils.py

- [x] `_per_token_group_quant_fp8`
- [x] `_per_token_group_quant_fp8_colmajor`
- [x] `_w8a8_block_fp8_matmul`

> 无测试用例

<details>
<summary></summary>

</details>

### vllm/model_executor/layers/quantization/utils/int8_utils.py

- [x] `_per_token_quant_int8`
- [x] `_per_token_group_quant_int8`
- [x] `_w8a8_block_int8_matmul`

> 无测试用例

<details>
<summary></summary>

</details>

### vllm/v1/sample/rejection_sampler.py

- [x] `rejection_greedy_sample_kernel`
- [x] `rejection_random_sample_kernel`
- [x] `expand_kernel`
- [x] `sample_recovered_tokens_kernel`

```python
pytest -svx tests/v1/sample/test_rejection_sampler.py
```

<details>
<summary>HuaweiCompilationError</summary>

```
E               huawei.HuaweiCompilationError:
E               ///------------------[ERROR][TritonAscend][BEG]------------------
E               [ConvertTritonIRToLinalgIR] encounters error:
E               /home/devuser/workspace/vllm-project/vllm/vllm/v1/sample/rejection_sampler.py:569:0: error: 'func.return' op has 1 operands, but enclosing function (@sample_recovered_tokens_kernel) returns 0
E               /home/devuser/workspace/vllm-project/vllm/vllm/v1/sample/rejection_sampler.py:569:0: note: see current operation: "func.return"(%12) : (i1) -> ()
E               ///------------------[ERROR][TritonAscend][END]------------------

```

</details>

### vllm/v1/spec_decode/eagle.py

- [x] `prepare_input_kernel` ✅

```python
python triton/test_prepare_input_kernel.py
```
