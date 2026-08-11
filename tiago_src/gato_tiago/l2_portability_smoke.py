"""Frozen static contract for the post-fix CUDA L2 portability smoke."""

from pathlib import Path


PROTOCOL_VERSION = "tiago_tool_center_l2_portability_smoke_1"
SMOKE_EXECUTION_AUTHORIZATION = None
AUTHORIZED_OUTPUT_PATH = Path(
    "/tmp/tiago-tool-center-l2-portability-smoke-authorized-once/smoke.json"
)
AUTHORIZED_CWD = Path("/workspace/GATO")
AUTHORIZED_ORIG_ARGV = (
    "python",
    "-B",
    "-m",
    "gato_tiago.l2_portability_smoke_runner",
    "--execute",
    "--output",
    str(AUTHORIZED_OUTPUT_PATH),
)
FROZEN_BUILD_HEAD = "154556ab7b45a5137a6d9c107c1e6e433b9cc057"
FROZEN_CUDA_ARCH = "61-real"
FROZEN_BUILD_SOURCE_HASHES = {
    "gato/utils/cuda.cuh": "0c61877b0f94bd7c5c0e863cfabb1e456efe4ea6e4938201dbedad9c15321494",
    "python/bindings.cu": "13d909ff8e6f498e435d0c5314a435808fa886fd9990a97bf3a87ea6903a24d0",
    "gato/bsqp/kernels/tool_position.cuh": "99fe1c402663bbc4e2a73f9ed5bae0550420b2c4d3ef29980bce9f047607ae25",
    "tools/build.sh": "5fa16321e426c5903e0b26626877bed6ecb806b7d31ebc25e8ddbb2de107fcc1",
}
FROZEN_MODULES = (
    {
        "module_name": "bsqp.bsqpN64_tiago_right_multimodal_toll",
        "extension_path": "python/bsqp/bsqpN64_tiago_right_multimodal_toll.cpython-310-x86_64-linux-gnu.so",
        "extension_sha256": "dbf7a016b644d75a36ebaddb6c25a456dec51814d5f794f5f2c5cfaa2b184ccb",
        "extension_size_bytes": 6686384,
        "reference_size": 10,
    },
    {
        "module_name": "bsqp.bsqpN64_tiago_right",
        "extension_path": "python/bsqp/bsqpN64_tiago_right.cpython-310-x86_64-linux-gnu.so",
        "extension_sha256": "3a97a4682c949eba483ad7be39fc64416032abae0141e6f049af58405f8b8f9f",
        "extension_size_bytes": 6764208,
        "reference_size": 6,
    },
    {
        "module_name": "bsqp.bsqpN64_tiago_right_multimodal",
        "extension_path": "python/bsqp/bsqpN64_tiago_right_multimodal.cpython-310-x86_64-linux-gnu.so",
        "extension_sha256": "988ef42c36dedd35227e54294ad1e2cbfb99f9c8f5da290aa8683b2e0a6de978",
        "extension_size_bytes": 6678192,
        "reference_size": 6,
    },
)
EXPECTED_CONSTRUCTOR_CALLS_PER_MODULE = 2
EXPECTED_FK_CALLS_PER_MODULE = 2
EXPECTED_GLOBAL_CONSTRUCTOR_CALLS = 6
EXPECTED_GLOBAL_FK_CALLS = 6
EXPECTED_WORKER_ARRAY_NAMES = frozenset(
    {"q_float32", "q16_float32", "fk_b1_float32", "fk_b16_float32"}
)
FORBIDDEN_CALL_COUNTS = {
    "v4_artifact_loads": 0,
    "task_rng_calls": 0,
    "task_construction_calls": 0,
    "initializer_calls": 0,
    "pin_external_model_calls": 0,
    "sim_forward_calls": 0,
    "solve_calls": 0,
    "sqp_calls": 0,
}
REQUIRED_SOURCE_PATHS = {
    "schema": "tiago_src/gato_tiago/l2_portability_smoke.py",
    "runner": "tiago_src/gato_tiago/l2_portability_smoke_runner.py",
    "worker": "tiago_src/gato_tiago/l2_portability_smoke_worker.py",
    "tests": "tests/python/test_l2_portability_smoke.py",
    "cuda_helper": "gato/utils/cuda.cuh",
    "bindings": "python/bindings.cu",
    "tool_position_kernel": "gato/bsqp/kernels/tool_position.cuh",
    "build_script": "tools/build.sh",
}


def frozen_metadata():
    return {
        "protocol_version": PROTOCOL_VERSION,
        "execution_authorized": SMOKE_EXECUTION_AUTHORIZATION is not None,
        "authorized_output_path": str(AUTHORIZED_OUTPUT_PATH),
        "authorized_cwd": str(AUTHORIZED_CWD),
        "authorized_orig_argv": list(AUTHORIZED_ORIG_ARGV),
        "frozen_build_head": FROZEN_BUILD_HEAD,
        "frozen_build_source_hashes": dict(FROZEN_BUILD_SOURCE_HASHES),
        "frozen_cuda_arch": FROZEN_CUDA_ARCH,
        "modules": [dict(row) for row in FROZEN_MODULES],
        "q": [0.0] * 7,
        "q_dtype": "float32",
        "constructor_calls_per_module": EXPECTED_CONSTRUCTOR_CALLS_PER_MODULE,
        "fk_calls_per_module": EXPECTED_FK_CALLS_PER_MODULE,
        "global_constructor_calls": EXPECTED_GLOBAL_CONSTRUCTOR_CALLS,
        "global_fk_calls": EXPECTED_GLOBAL_FK_CALLS,
        "forbidden_call_counts": dict(FORBIDDEN_CALL_COUNTS),
        "optimization_evidence": False,
        "timing_evidence": False,
    }
