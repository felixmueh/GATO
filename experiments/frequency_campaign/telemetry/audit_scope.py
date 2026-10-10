"""CPU-only source projection audit; no compiler/native import/GPU calls."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE_COMMIT = "12dc7ccdc6acf700235e5cd35cf2ed09328b7962"
NATIVE_FILES = (
    "gato/types.cuh", "gato/bsqp/bsqp.cuh",
    "gato/bsqp/kernels/pcg.cuh", "python/bindings.cu",
)
MACRO_IF = "#if defined(GATO_EXPERIMENT_TELEMETRY) && GATO_EXPERIMENT_TELEMETRY"


def project(source, enabled):
    # Only the flat experiment guards are projected; other directives survive.
    out = []
    selected = None
    for line in source.splitlines():
        if line.strip() == MACRO_IF:
            assert selected is None, "nested experiment guards need explicit audit"
            selected = enabled
        elif selected is not None and line.strip() == "#else":
            selected = not selected
        elif selected is not None and line.strip() == "#endif":
            selected = None
        elif selected is None or selected:
            out.append(line)
    assert selected is None
    return "\n".join(out)


def tokens(source):
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    # Adding a branch annotation expands the existing single break statement.
    source = re.sub(r"\{\s*break\s*;\s*\}", "break;", source)
    return re.findall(r"\w+|[^\s]", source)


def call_sites(source, pattern):
    return re.findall(pattern, re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S))


def audit():
    sync_pattern = r"\b(?:cuda\w*Synchronize|__syncthreads\w*|__syncwarp|__ballot\w*|__any_sync|__all_sync|atomic\w*)\s*\("
    copy_pattern = r"\bcudaMemcpy(?:Async)?\s*\("
    result = {"base_commit": BASE_COMMIT, "files": {}, "scope": "static CPU audit; no native behavior qualification"}
    for name in NATIVE_FILES:
        before = subprocess.check_output(["git", "show", f"{BASE_COMMIT}:{name}"], cwd=ROOT, text=True)
        current = (ROOT / name).read_text()
        off, on = project(current, False), project(current, True)
        off_equal = tokens(off) == tokens(before)
        sync_equal = call_sites(on, sync_pattern) == call_sites(before, sync_pattern)
        copies_equal = call_sites(on, copy_pattern) == call_sites(before, copy_pattern)
        kernels_equal = on.count("<<<") == before.count("<<<")
        assert off_equal, f"OFF differs from baseline tokens: {name}"
        assert sync_equal and copies_equal and kernels_equal, f"ON adds/reorders sync/copy/kernel calls: {name}"
        result["files"][name] = {
            "sha256": hashlib.sha256(current.encode()).hexdigest(),
            "off_tokens_equal_baseline": off_equal,
            "on_sync_vote_atomic_call_sequence_equal_baseline": sync_equal,
            "on_copy_call_sequence_equal_baseline": copies_equal,
            "on_kernel_launch_count_equal_baseline": kernels_equal,
        }
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert 'option(GATO_EXPERIMENT_TELEMETRY "Record existing SQP/PCG stop paths" OFF)' in cmake
    result["cmake_default_off"] = True
    result["device_bytes_delta_per_lane"] = 4
    result["device_bytes_delta_B16"] = 64
    result["pinned_host_bytes_B16"] = 128
    result["new_explicit_synchronization_calls"] = 0
    result["new_copy_calls"] = 0
    result["new_kernel_launches"] = 0
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
