## `triton/kernel_paged_attention_2d`

```python
python triton/kernel_paged_attention_2d/test.py
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
subprocess.CalledProcessError: Command '['/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt', '/tmp/tmpj5_yku23/kernel.ttir.mlir', '--triton-to-linalg=global-kernel=false named-ops=True', '-o', '/tmp/tmpj5_yku23/kernel.ttadapter.mlir']' died with <Signals.SIGSEGV: 11>.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/devuser/workspace/vllm-project/vllm/triton/kernel_paged_attention_2d/test.py", line 44, in <module>
    fn()
  File "/home/devuser/workspace/vllm-project/vllm/triton/kernel_paged_attention_2d/test.py", line 39, in fn
    chunked_prefill_paged_decode(*args, **kwargs)
  File "/home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/chunked_prefill_paged_decode.py", line 327, in chunked_prefill_paged_decode
    kernel_paged_attention_2d[(
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/runtime/jit.py", line 330, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/runtime/jit.py", line 623, in run
    kernel = self.compile(
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/compiler/compiler.py", line 297, in compile
    raise MLIRCompilationError(stage_name, e.stderr.decode('utf-8'))
triton.compiler.errors.MLIRCompilationError:
///------------------[ERROR][Triton][BEG]------------------
[ConvertTritonIRToLinalgIR] encounters error:
/home/devuser/workspace/vllm-project/vllm/vllm/attention/ops/chunked_prefill_paged_decode.py:133:41: error: cannot div 0!
        K_load = tl.load(key_cache_ptr + k_offset,
                                        ^
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.      Program arguments: /home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt /tmp/tmpj5_yku23/kernel.ttir.mlir "--triton-to-linalg=global-kernel=false named-ops=True" -o /tmp/tmpj5_yku23/kernel.ttadapter.mlir
 #0 0x0000aaaad2a1ef80 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaef80)
 #1 0x0000aaaad2a1ca30 llvm::sys::RunSignalHandlers() (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xfaca30)
 #2 0x0000aaaad2a1cb78 SignalHandler(int) Signals.cpp:0:0
 #3 0x0000ffff99ca77c0 (linux-vdso.so.1+0x7c0)
 #4 0x0000aaaad1d557e0 mlir::mulOpFoldResult(mlir::OpFoldResult const&, mlir::Value const&, mlir::Location const&, mlir::OpBuilder&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x2e57e0)
 #5 0x0000aaaad1bdc3a4 mlir::triton::BlockData::mulBlock(mlir::triton::BlockData&, mlir::triton::BlockData&, mlir::Location, mlir::ConversionPatternRewriter&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16c3a4)
 #6 0x0000aaaad1bdf254 mlir::triton::BlockDataParser::parseMul(mlir::arith::MulIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f254)
 #7 0x0000aaaad1bde74c mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e74c)
 #8 0x0000aaaad1bdf06c mlir::triton::BlockDataParser::parseAdd(mlir::arith::AddIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f06c)
 #9 0x0000aaaad1bde764 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e764)
#10 0x0000aaaad1be0160 mlir::triton::BlockDataParser::parseBroadcast(mlir::triton::BroadcastOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x170160)
#11 0x0000aaaad1bde988 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e988)
#12 0x0000aaaad1bdf040 mlir::triton::BlockDataParser::parseAdd(mlir::arith::AddIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f040)
#13 0x0000aaaad1bde764 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e764)
#14 0x0000aaaad1bdf040 mlir::triton::BlockDataParser::parseAdd(mlir::arith::AddIOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16f040)
#15 0x0000aaaad1bde764 mlir::triton::BlockDataParser::parse(mlir::Value, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16e764)
#16 0x0000aaaad1bddeb0 mlir::triton::BlockDataParser::parseAddPtr(mlir::triton::AddPtrOp, mlir::triton::BlockData&, mlir::Location const&, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>> const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x16deb0)
#17 0x0000aaaad1be34fc mlir::triton::BlockDataParser::rewriteAddPtr(mlir::triton::AddPtrOp, mlir::ConversionPatternRewriter&, llvm::SmallDenseMap<mlir::Value, mlir::triton::BlockData, 4u, llvm::DenseMapInfo<mlir::Value, void>, llvm::detail::DenseMapPair<mlir::Value, mlir::triton::BlockData>>&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x1734fc)
#18 0x0000aaaad1bc4ed0 LoadStoreConverter::AddPtrConverter::matchAndRewrite(mlir::triton::AddPtrOp, mlir::triton::AddPtrOpAdaptor, mlir::ConversionPatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x154ed0)
#19 0x0000aaaad1bb0cb0 mlir::OpConversionPattern<mlir::triton::AddPtrOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::Value>, mlir::ConversionPatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x140cb0)
#20 0x0000aaaad267eeec mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc0eeec)
#21 0x0000aaaad26a5ab0 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc35ab0)
#22 0x0000aaaad2682d24 (anonymous namespace)::OperationLegalizer::legalize(mlir::Operation*, mlir::ConversionPatternRewriter&) DialectConversion.cpp:0:0
#23 0x0000aaaad26831a8 mlir::OperationConverter::convert(mlir::ConversionPatternRewriter&, mlir::Operation*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc131a8)
#24 0x0000aaaad2688788 mlir::OperationConverter::convertOperations(llvm::ArrayRef<mlir::Operation*>) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc18788)
#25 0x0000aaaad26894f4 mlir::applyPartialConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xc194f4)
#26 0x0000aaaad1ba7848 (anonymous namespace)::TritonToLinalgPass::runOnOperation() TritonToLinalgPass.cpp:0:0
#27 0x0000aaaad2644974 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd4974)
#28 0x0000aaaad2644df0 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd4df0)
#29 0x0000aaaad2645cb8 mlir::PassManager::run(mlir::Operation*) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbd5cb8)
#30 0x0000aaaad2639a38 performActions(llvm::raw_ostream&, std::shared_ptr<llvm::SourceMgr> const&, mlir::MLIRContext*, mlir::MlirOptMainConfig const&) MlirOptMain.cpp:0:0
#31 0x0000aaaad263a0bc processBuffer(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::MlirOptMainConfig const&, mlir::DialectRegistry&, llvm::ThreadPoolInterface*) MlirOptMain.cpp:0:0
#32 0x0000aaaad263a1f8 llvm::LogicalResult llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>::callback_fn<mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&)::'lambda'(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>(long, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&) MlirOptMain.cpp:0:0
#33 0x0000aaaad29b1328 mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::function_ref<llvm::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>, llvm::raw_ostream&, llvm::StringRef, llvm::StringRef) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xf41328)
#34 0x0000aaaad26340bc mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbc40bc)
#35 0x0000aaaad263a308 mlir::MlirOptMain(int, char**, llvm::StringRef, llvm::StringRef, mlir::DialectRegistry&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbca308)
#36 0x0000aaaad263a6fc mlir::MlirOptMain(int, char**, llvm::StringRef, mlir::DialectRegistry&) (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0xbca6fc)
#37 0x0000aaaad1b7e548 main (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x10e548)
#38 0x0000ffff997173fc __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
#39 0x0000ffff997174cc call_init ./csu/../csu/libc-start.c:128:20
#40 0x0000ffff997174cc __libc_start_main ./csu/../csu/libc-start.c:379:5
#41 0x0000aaaad1b9de70 _start (/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/triton-adapter-opt+0x12de70)
///------------------[ERROR][Triton][END]------------------

[ERROR] 2025-04-24-09:00:37 (PID:2670925, Device:0, RankID:-1) ERR99999 UNKNOWN application exception
```
