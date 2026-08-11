from pathlib import Path


def test_persisting_l2_setup_is_checked_current_device_and_unsupported_safe():
    root = Path(__file__).resolve().parents[2]
    source = (root / "gato/utils/cuda.cuh").read_text()
    begin = source.index("void setL2PersistingAccess")
    end = source.index("void resetL2PersistingAccess", begin)
    function = source[begin:end]

    assert "#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000" in function
    assert "gpuErrchk(cudaGetDevice(&device));" in function
    assert "gpuErrchk(cudaGetDeviceProperties(&prop, device));" in function
    assert function.count("cudaGetDevice(&device)") == 1
    assert function.count("cudaGetDeviceProperties(&prop, device)") == 1
    unsupported = function.index("if (prop.persistingL2CacheMaxSize == 0)")
    early_return = function.index("return;", unsupported)
    checked_limit = function.index(
        "gpuErrchk(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size));"
    )
    assert unsupported < early_return < checked_limit
    assert function.count("cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize") == 1
    assert "cudaGetLastError" not in function
    assert "cudaPeekAtLastError" not in function


def test_both_python_solver_constructors_use_the_portable_helper():
    root = Path(__file__).resolve().parents[2]
    bindings = (root / "python/bindings.cu").read_text()
    assert bindings.count("setL2PersistingAccess(1.0);") == 2
    assert "cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize" not in bindings


def test_persisting_l2_reset_is_checked_and_unsupported_safe():
    root = Path(__file__).resolve().parents[2]
    source = (root / "gato/utils/cuda.cuh").read_text()
    begin = source.index("void resetL2PersistingAccess")
    function = source[begin:]

    assert "#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000" in function
    assert "gpuErrchk(cudaGetDevice(&device));" in function
    assert "gpuErrchk(cudaGetDeviceProperties(&prop, device));" in function
    assert function.count("cudaGetDevice(&device)") == 1
    assert function.count("cudaGetDeviceProperties(&prop, device)") == 1
    unsupported = function.index("if (prop.persistingL2CacheMaxSize == 0)")
    early_return = function.index("return;", unsupported)
    checked_limit = function.index(
        "gpuErrchk(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));"
    )
    checked_reset = function.index("gpuErrchk(cudaCtxResetPersistingL2Cache());")
    assert unsupported < early_return < checked_limit < checked_reset
    assert function.count("cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize") == 1
    assert function.count("cudaCtxResetPersistingL2Cache()") == 1
    assert "cudaGetLastError" not in function
    assert "cudaPeekAtLastError" not in function
