## `_state_passing_fwd_kernel`

> vllm/model_executor/layers/mamba/ops/ssd_state_passing.py

```python
python triton/state_passing_fwd/test.py
```

Result: Failed ❌

```
Traceback (most recent call last):
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/compiler/compiler.py", line 289, in compile
    next_module = compile_ir(module, metadata)
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 310, in <lambda>
    stages["npubin"] = lambda src, metadata: linalg_to_bin_enable_npu_compile(src, metadata, options)
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/backends/huawei/compiler.py", line 181, in linalg_to_bin_enable_npu_compile
    ret = subprocess.run(cmd_list, capture_output=True, check=True)
  File "/home/devuser/.conda/envs/triton/lib/python3.10/subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/home/devuser/program/bisheng/compiler/npuc', '/tmp/tmp0uq92uv2/kernel.ttadapter.mlir', '--enable-auto-multi-buffer=True', '-o', '/tmp/tmp0uq92uv2/kernel']' died with <Signals.SIGABRT: 6>.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/devuser/workspace/vllm-project/vllm/triton/state_passing_fwd/test.py", line 34, in <module>
    fn()
  File "/home/devuser/workspace/vllm-project/vllm/triton/state_passing_fwd/test.py", line 29, in fn
    ssd_state_passing._state_passing_fwd(states, dA_chunk_cumsum)
  File "/home/devuser/workspace/vllm-project/vllm/vllm/model_executor/layers/mamba/ops/ssd_state_passing.py", line 178, in _state_passing_fwd
    _state_passing_fwd_kernel[grid](
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/runtime/jit.py", line 330, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/runtime/autotuner.py", line 194, in run
    ret = self.fn.run(
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/runtime/jit.py", line 623, in run
    kernel = self.compile(
  File "/home/devuser/workspace/ascend/triton-ascend/triton/python/triton/compiler/compiler.py", line 297, in compile
    raise MLIRCompilationError(stage_name, e.stderr.decode('utf-8'))
triton.compiler.errors.MLIRCompilationError:
///------------------[ERROR][Triton][BEG]------------------
[ConvertLinalgRToBinary] encounters error:
error: strides must not be zero
npuc: /home/JenkinsStub/llvm-project/mlir/include/mlir/IR/StorageUniquerSupport.h:181: static ConcreteT mlir::detail::StorageUserBase<mlir::StridedLayoutAttr, mlir::Attribute, mlir::detail::StridedLayoutAttrStorage, mlir::detail::AttributeUniquer, Trait>::get(mlir::MLIRContext *, Args &&...) [ConcreteT = mlir::StridedLayoutAttr, BaseT = mlir::Attribute, StorageT = mlir::detail::StridedLayoutAttrStorage, UniquerT = mlir::detail::AttributeUniquer, Traits = <Trait>, Args = <long, llvm::ArrayRef<long>>]: Assertion `succeeded(ConcreteT::verify(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.      Program arguments: /home/devuser/program/bisheng/compiler/npuc /tmp/tmp0uq92uv2/kernel.ttadapter.mlir --enable-auto-multi-buffer=True -o /tmp/tmp0uq92uv2/kernel
 #0 0x0000000004295f18 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/home/devuser/program/bisheng/compiler/npuc+0x4295f18)
 #1 0x0000000004293c30 llvm::sys::RunSignalHandlers() (/home/devuser/program/bisheng/compiler/npuc+0x4293c30)
 #2 0x0000000004296660 SignalHandler(int) Signals.cpp:0:0
 #3 0x0000ffff84df17c0 (linux-vdso.so.1+0x7c0)
 #4 0x0000ffff8482f1f0 __pthread_kill_implementation ./nptl/pthread_kill.c:44:76
 #5 0x0000ffff847ea67c gsignal ./signal/../sysdeps/posix/raise.c:27:6
 #6 0x0000ffff847d7130 abort ./stdlib/abort.c:81:7
 #7 0x0000ffff847e3fd4 __assert_fail_base ./assert/assert.c:91:7
 #8 0x0000ffff847e404c (/lib/aarch64-linux-gnu/libc.so.6+0x3404c)
 #9 0x0000000003e83840 mlir::StridedLayoutAttr::getOffset() const (/home/devuser/program/bisheng/compiler/npuc+0x3e83840)
#10 0x0000000003e83768 mlir::StridedLayoutAttr::get(mlir::MLIRContext*, long, llvm::ArrayRef<long>) (/home/devuser/program/bisheng/compiler/npuc+0x3e83768)
#11 0x0000000003795784 ReinterpretCastReturnTypeCanonicalizer::operator()(mlir::memref::ReinterpretCastOp, llvm::ArrayRef<mlir::OpFoldResult>, llvm::ArrayRef<mlir::OpFoldResult>, llvm::ArrayRef<mlir::OpFoldResult>) MemRefOps.cpp:0:0
#12 0x0000000003795500 mlir::OpWithOffsetSizesAndStridesConstantArgumentFolder<mlir::memref::ReinterpretCastOp, ReinterpretCastReturnTypeCanonicalizer, ReinterpretCastCanonicalizer>::matchAndRewrite(mlir::memref::ReinterpretCastOp, mlir::PatternRewriter&) const MemRefOps.cpp:0:0
#13 0x00000000037949d4 mlir::detail::OpOrInterfaceRewritePatternBase<mlir::memref::ReinterpretCastOp>::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const MemRefOps.cpp:0:0
#14 0x0000000003a2a538 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<mlir::LogicalResult (mlir::Pattern const&)>)::$_2::operator()() const PatternApplicator.cpp:0:0
#15 0x0000000003a275dc mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<mlir::LogicalResult (mlir::Pattern const&)>) (/home/devuser/program/bisheng/compiler/npuc+0x3a275dc)
#16 0x0000000003a1597c (anonymous namespace)::GreedyPatternRewriteDriver::processWorklist() GreedyPatternRewriteDriver.cpp:0:0
#17 0x0000000003a120d8 mlir::applyPatternsAndFoldGreedily(mlir::Region&, mlir::FrozenRewritePatternSet const&, mlir::GreedyRewriteConfig, bool*) (/home/devuser/program/bisheng/compiler/npuc+0x3a120d8)
#18 0x00000000039c3758 (anonymous namespace)::Canonicalizer::runOnOperation() Canonicalizer.cpp:0:0
#19 0x0000000003a6d888 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/home/devuser/program/bisheng/compiler/npuc+0x3a6d888)
#20 0x0000000003a6dfe8 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (/home/devuser/program/bisheng/compiler/npuc+0x3a6dfe8)
#21 0x0000000003a70028 mlir::PassManager::run(mlir::Operation*) (/home/devuser/program/bisheng/compiler/npuc+0x3a70028)
#22 0x00000000016b391c bishengir::runPipeline(mlir::ModuleOp, std::function<void (mlir::PassManager&, bishengir::BiShengPipelineOptions const&)> const&, bishengir::BiShengPipelineOptions const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>> const&) (/home/devuser/program/bisheng/compiler/npuc+0x16b391c)
#23 0x00000000016bd944 bishengir::runBiShengPipeline(mlir::ModuleOp, bishengir::BiShengPipelineOptions const&) (/home/devuser/program/bisheng/compiler/npuc+0x16bd944)
#24 0x0000000001649ab0 main (/home/devuser/program/bisheng/compiler/npuc+0x1649ab0)
#25 0x0000ffff847d73fc __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
#26 0x0000ffff847d74cc call_init ./csu/../csu/libc-start.c:128:20
#27 0x0000ffff847d74cc __libc_start_main ./csu/../csu/libc-start.c:379:5
#28 0x0000000001647c6c _start (/home/devuser/program/bisheng/compiler/npuc+0x1647c6c)
///------------------[ERROR][Triton][END]------------------

[ERROR] 2025-04-24-07:18:12 (PID:2650301, Device:0, RankID:-1) ERR99999 UNKNOWN application exception
```
