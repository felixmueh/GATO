# Explicit CPU/GPU shared-RAM experiment

This isolated branch starts at public LDLT candidate `16dd835`. It prepares
Indy7 N128 float B1 for manual Jetson experiments. It does not execute on Jetson,
change power/clocks, restore a private TIAGo fix, or claim Jetson performance.

Enable the experiment explicitly with CMake `-DPRECOND_MANAGED_S=ON`. The option
defaults OFF and is applied only to `bsqpN128_indy7`. A narrow allocation macro
at the one BSQP Schur-S allocation callsite selects the requested allocator:

| Method | Schur S allocation | CPU/GPU factor handoff |
|---|---|---|
| `normal` | original `cudaMalloc` | original native GPU preconditioner |
| `normal_shared` | explicit `cudaMallocManaged` | original native GPU preconditioner; allocation control |
| `block_ldlt` | original `cudaMalloc` | explicit D2H S and H2D factors; copy control |
| `block_ldlt_shared` | explicit `cudaMallocManaged` | CPU reads original S; host writes managed L/Dinv; GPU reads same pointers |

Select the method before constructing BSQP and use a fresh process per repeat.
Shared requests require runtime `managedMemory=1` and the explicit build option;
unsupported requests fail without fallback. `normal` retains its original S
allocation even in a managed-enabled binary. The CUDA per-block PCG shared-memory
size, recurrence, arithmetic, stopping, budgets and factor mathematics stay as
in the public LDLT candidate. Factor setup has no shift/floor/rescue.

Minimal native API order (the portable task runner supplies the controller):

```python
native.set_method("block_ldlt_shared")  # or unchanged baseline "normal"
native.set_end_to_end_mode(True)
native.prepare()                      # workspaces outside task timing
before = native.shared_memory_info()
# Construct public MPC/BSQP; time the COMPLETE public task, including warmup.
# Run task, then export metrics and native.shared_memory_info().
```

End-to-end mode defaults OFF to preserve the desktop instrumented mode. Explicit
`True` makes native copy hooks pass through their original CUDA calls, recording
only byte/call counts, with no additional per-copy fences or timers. The normal
and normal_shared PCG hooks call the original native solver without experimental
PCG events or a pre-solve fence. LDLT disables inner event timing and completion
waits in this mode; the ordinary BSQP completion remains. Metadata marks isolated
PCG wall/events unavailable. **Actual `whole_task_wall_us` is primary; subtract
nothing.** All remaining native I/O, the small skip-flag read, CPU factorization,
host factor writes, runtime/cache/migration effects and synchronization stay in
that task wall. Construction and workspace allocation are separately recorded
setup costs. Dummy transfer calibration is rejected in end-to-end mode.

Managed memory is allocated explicitly for S and shared factors; physical shared
RAM is not inferred just from running on a Jetson. `shared_memory_info()` records
`integrated`, `managedMemory`, `concurrentManagedAccess`,
`directManagedMemAccessFromHost`, `pageableMemoryAccess`,
`pageableMemoryAccessUsesHostPageTables`, allocation mode/count, actual last S
allocation and a managed-pointer verification at shared solve entry. A discrete
GPU can validate mechanics but cannot establish Jetson performance. No prefetch
or memory advice hides runtime handoff costs.

The CPU waits for `cudaDeviceSynchronize` before reading managed S or touching
managed factors. S is read-only during CPU setup. Factor construction uses the
existing CPU strip LDLT; `std::copy` writes its values to the managed L/Dinv
arrays inside task timing. A CPU release fence and device synchronization hand
ownership back before any GPU launch. This permits devices reporting
`concurrentManagedAccess=0`; simultaneous CPU/GPU managed access is not used.
See [CUDA for Tegra memory management](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management)
and [CUDA Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html).

Root owns source review, desktop build and functional mechanics checks. Jetson
execution is manual, inside the existing container; MODE_40W is user-confirmed,
not freshly measured. No host power/clock queries or changes are required.
Source preparation performed no builds, tests, GPU calls or Jetson execution.
