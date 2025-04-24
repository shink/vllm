## `context_attention_fwd`

1. _fwd_kernel

```python
python triton/context_attention_fwd/test.py
```

Result: Failed ❌

```
Traceback (most recent call last):
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/compiler/compiler.py", line 289, in compile
    next_module = compile_ir(module, metadata)
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 310, in <lambda>
    stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options, named_ops=True)
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 60, in ttir_to_linalg
    ret = subprocess.run(cmd_list, capture_output=True, check=True)
  File "/home/devuser/.conda/envs/triton/lib/python3.10/subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt', '/tmp/tmp22lqcbaw/kernel.ttir.mlir', '--triton-to-linalg=global-kernel=false named-ops=True', '-o', '/tmp/tmp22lqcbaw/kernel.ttadapter.mlir']' died with <Signals.SIGSEGV: 11>.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/devuser/workspace/vllm-project/vllm/triton/kernel_paged_attention_2d/test.py", line 44, in <module>
    fn()
  File "/home/devuser/workspace/vllm-project/vllm/triton/kernel_paged_attention_2d/test.py", line 39, in fn
    chunked_prefill_paged_decode(*args, **kwargs)
  File "/home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/chunked_prefill_paged_decode.py", line 236, in chunked_prefill_paged_decode
    context_attention_fwd(
  File "/home/devuser/.conda/envs/triton/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py", line 850, in context_attention_fwd
    _fwd_kernel[grid](
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/runtime/jit.py", line 330, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/runtime/jit.py", line 623, in run
    kernel = self.compile(
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/compiler/compiler.py", line 297, in compile
    raise MLIRCompilationError(stage_name, e.stderr.decode('utf-8'))
triton.compiler.errors.MLIRCompilationError:
///------------------[ERROR][Triton][BEG]------------------
[ConvertTritonIRToLinalgIR] encounters error:
/home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:244:42: remark: [MaskState] Unsupported cmpi scenario
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                                         ^
/home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:244:42: note: see current operation: %142 = arith.cmpi sge, %107, %141 : tensor<128x64xi32>
/home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:244:42: remark: [MaskState] Unsupported cmpi scenario
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                                         ^
/home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:244:42: note: see current operation: %131 = arith.cmpi sge, %98, %130 : tensor<128x64xi32>
/home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/prefix_prefill.py:156:26: error: cannot div 0!
                K_cache + off_k,
                         ^
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.      Program arguments: /home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmp22lqcbaw/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmp22lqcbaw/kernel.ttadapter.mlir
 #0 0x0000aaaad662ef80 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaef80)
 #1 0x0000aaaad662ca30 llvm::sys::RunSignalHandlers() (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaca30)
 #2 0x0000aaaad662cb78 SignalHandler(int) Signals.cpp:0:0
 #3 0x0000ffffac6767c0 (linux-vdso.so.1+0x7c0)
 #4 0x0000aaaad59657e0 mlir::mulOpFoldResult(mlir::OpFoldResult const&, mlir::Value const&, mlir::Location const&, mlir::OpBuilder&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x2e57e0)
 #5 0x0000aaaad57ec3a4 mlir::triton::BlockData::mulBlock(mlir::triton::BlockData&, mlir::triton::BlockData&, mlir::Location, mlir::ConversionPatternRewriter&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16c3a4)
 #6 0x0000aaaad57ef254 mlir::triton::BlockDataParser::parseMul(mlir::arith::MulIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f254)
 #7 0x0000aaaad57ee74c mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e74c)
 #8 0x0000aaaad57ee9dc mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e9dc)
 #9 0x0000aaaad57ef06c mlir::triton::BlockDataParser::parseAdd(mlir::arith::AddIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f06c)
#10 0x0000aaaad57ee764 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e764)
#11 0x0000aaaad57f0160 mlir::triton::BlockDataParser::parseBroadcast(mlir::triton::BroadcastOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x170160)
#12 0x0000aaaad57ee988 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e988)
#13 0x0000aaaad57ef040 mlir::triton::BlockDataParser::parseAdd(mlir::arith::AddIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f040)
#14 0x0000aaaad57ee764 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e764)
#15 0x0000aaaad57ef040 mlir::triton::BlockDataParser::parseAdd(mlir::arith::AddIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f040)
#16 0x0000aaaad57ee764 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e764)
#17 0x0000aaaad57edeb0 mlir::triton::BlockDataParser::parseAddPtr(mlir::triton::AddPtrOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16deb0)
#18 0x0000aaaad57f34fc mlir::triton::BlockDataParser::rewriteAddPtr(mlir::triton::AddPtrOp, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>>&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x1734fc)
#19 0x0000aaaad57d4ed0 LoadStoreConverter::AddPtrConverter::matchAndRewrite(mlir::triton::AddPtrOp, mlir::triton::AddPtrOpAdaptor, mlir::ConversionPatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x154ed0)
#20 0x0000aaaad57c0cb0 mlir::OpConversionPattern<mlir::triton::AddPtrOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::Value>, mlir::ConversionPatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x140cb0)
#21 0x0000aaaad628eeec mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc0eeec)
#22 0x0000aaaad62b5ab0 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc35ab0)
#23 0x0000aaaad6292d24 (anonymous namespace)::OperationLegalizer::legalize(mlir::Operation*, mlir::ConversionPatternRewriter&) DialectConversion.cpp:0:0
#24 0x0000aaaad62931a8 mlir::OperationConverter::convert(mlir::ConversionPatternRewriter&, mlir::Operation*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc131a8)
#25 0x0000aaaad6298788 mlir::OperationConverter::convertOperations(llvm::ArrayRef<mlir::Operation*>) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc18788)
#26 0x0000aaaad62994f4 mlir::applyPartialConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc194f4)
#27 0x0000aaaad57b7848 (anonymous namespace)::TritonToLinalgPass::runOnOperation() TritonToLinalgPass.cpp:0:0
#28 0x0000aaaad6254974 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd4974)
#29 0x0000aaaad6254df0 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd4df0)
#30 0x0000aaaad6255cb8 mlir::PassManager::run(mlir::Operation*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd5cb8)
#31 0x0000aaaad6249a38 performActions(llvm::raw_ostream&, std::shared_ptr<llvm::SourceMgr> const&, mlir::MLIRContext*, mlir::MlirOptMainConfig const&) MlirOptMain.cpp:0:0
#32 0x0000aaaad624a0bc processBuffer(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::MlirOptMainConfig const&, mlir::DialectRegistry&, llvm::ThreadPoolInterface*) MlirOptMain.cpp:0:0
#33 0x0000aaaad624a1f8 llvm::LogicalResult llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>::callback_fn<mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&)::'lambda'(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>(long, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&) MlirOptMain.cpp:0:0
#34 0x0000aaaad65c1328 mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>, llvm::raw_ostream&, llvm::StringRef, llvm::StringRef) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xf41328)
#35 0x0000aaaad62440bc mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbc40bc)
#36 0x0000aaaad624a308 mlir::MlirOptMain(int, char**, llvm::StringRef, llvm::StringRef, mlir::DialectRegistry&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbca308)
#37 0x0000aaaad624a6fc mlir::MlirOptMain(int, char**, llvm::StringRef, mlir::DialectRegistry&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbca6fc)
#38 0x0000aaaad578e548 main (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x10e548)
#39 0x0000ffffac0e73fc __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
#40 0x0000ffffac0e74cc call_init ./csu/../csu/libc-start.c:128:20
#41 0x0000ffffac0e74cc __libc_start_main ./csu/../csu/libc-start.c:379:5
#42 0x0000aaaad57ade70 _start (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x12de70)
///------------------[ERROR][Triton][END]------------------

[ERROR] 2025-04-24-08:29:30 (PID:2666241, Device:0, RankID:-1) ERR99999 UNKNOWN application exception
```
