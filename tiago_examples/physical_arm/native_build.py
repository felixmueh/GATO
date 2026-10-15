"""Prepare and select isolated N32 native modules for manual TIAGo tasks.

Preparation compiles only; loading never builds or probes a GPU. Each process
must choose one profile before constructing the normal BSQP/MPC controller.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "build/physical-arm-native"
TARGET = "bsqpN32_tiago_right"
MODULE = "bsqp." + TARGET
PROFILES = {
    "tracking": {"TIAGO_COST_JOINT_MARGIN_RAD": "0.0",
                 "TIAGO_COST_VELOCITY_LIMIT_FRACTION": "1.0",
                 "TIAGO_COST_CONTROL_LIMIT_FRACTION": "1.0"},
    "reaching": {"TIAGO_COST_JOINT_MARGIN_RAD": "0.20",
                 "TIAGO_COST_VELOCITY_LIMIT_FRACTION": "0.80",
                 "TIAGO_COST_CONTROL_LIMIT_FRACTION": "0.85"},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity() -> dict:
    # Hash actual working-tree bytes, including uncommitted relevant edits and
    # generated TIAGo GRiD headers; unrelated studies/results do not invalidate.
    paths = [ROOT / "CMakeLists.txt", ROOT / "python/bindings.cu",
             ROOT / "python/bsqp/interface.py", Path(__file__).resolve()]
    for directory in (ROOT / "gato", ROOT / "GRiD"):
        paths.extend(p for p in directory.rglob("*") if p.is_file()
                     and p.suffix in {".h", ".hpp", ".cuh", ".cu", ".cpp", ".cmake"}
                     and not any(part in {"indy7", "iiwa14", "build", ".git"}
                                 for part in p.relative_to(directory).parts))
    files = {str(p.relative_to(ROOT)): _sha256(p) for p in sorted(set(paths))}
    return {"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "files": files}


def _python_identity() -> dict:
    return {"executable": str(Path(sys.executable).resolve()), "version": sys.version,
            "soabi": sysconfig.get_config_var("SOABI"),
            "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX")}


def _platform_identity() -> dict:
    return {"system": platform.system(), "machine": platform.machine()}


def _config(profile: str, architecture: str) -> dict:
    return {"profile": profile, "cuda_architecture": architecture,
            "build_type": "Release", "knots": 32, "plant": "tiago_right",
            "definitions": PROFILES[profile], "build_testing": False,
            "cuda_flags": "", "cxx_flags": "", "linker_flags": ""}


def _profile_dir(profile: str) -> Path:
    if profile not in PROFILES:
        raise ValueError(f"Unknown native profile {profile!r}; choose tracking or reaching")
    return CACHE / profile


def _prepare_command(architecture: str | None = None) -> str:
    arch = architecture if architecture in ("87", "120") else "<120-for-RTX5060-or-87-for-Jetson>"
    return (f"{shlex.quote(sys.executable)} "
            f"{shlex.quote(str(Path(__file__).resolve()))} --cuda-architecture {arch}")


def _invalid(message: str, architecture: str | None = None) -> RuntimeError:
    return RuntimeError(f"{message}. Prepare native modules explicitly:\n{_prepare_command(architecture)}")


def verified_receipt(profile: str) -> dict:
    """Validate source, runtime identity and artifact bytes without importing CUDA."""
    directory = _profile_dir(profile)
    architecture = None
    try:
        receipt = json.loads((directory / "receipt.json").read_text())
        architecture = receipt["config"]["cuda_architecture"]
        if architecture not in ("87", "120"):
            raise ValueError("unsupported CUDA architecture")
        expected = {"schema": 1, "config": _config(profile, architecture),
                    "source": _source_identity(), "python": _python_identity(),
                    "platform": _platform_identity()}
        for key, value in expected.items():
            if receipt[key] != value:
                raise ValueError(f"stale or mismatched {key}")
        module = directory / "module" / (TARGET + _python_identity()["extension_suffix"])
        if receipt["module"]["path"] != str(module):
            raise ValueError("unexpected module path")
        if receipt["module"]["sha256"] != _sha256(module):
            raise ValueError("module hash changed")
        build = receipt["build"]
        if build["directory"] != str(directory / "cmake-build"):
            raise ValueError("unexpected build directory")
        if build["cmake_cache_sha256"] != _sha256(directory / "cmake-build/CMakeCache.txt"):
            raise ValueError("build configuration changed")
        return receipt
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise _invalid(f"Native {profile} profile is unavailable or stale ({exc})", architecture) from exc


def load_profile(profile: str) -> dict:
    """Load the verified profile into the existing bsqp.interface import path.

    Returns the receipt for inclusion in run metadata. Repeat selection of the
    same unchanged module is allowed; switching profiles requires a new process.
    """
    receipt = verified_receipt(profile)
    identity = (profile, receipt["source"]["sha256"], receipt["module"]["sha256"])
    loaded = sys.modules.get(MODULE)
    if loaded is not None:
        if getattr(loaded, "_physical_arm_native_identity", None) != identity:
            raise RuntimeError(f"{MODULE} is already loaded with a different or unverified profile; "
                               "start a fresh Python process and load the requested profile before MPC")
        return receipt
    python_path = str(ROOT / "python")
    if python_path not in sys.path:
        sys.path.insert(0, python_path)
    package = importlib.import_module("bsqp")
    spec = importlib.util.spec_from_file_location(MODULE, receipt["module"]["path"])
    if spec is None or spec.loader is None:
        raise _invalid(f"Cannot load native {profile} extension", receipt["config"]["cuda_architecture"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = module
    try:
        spec.loader.exec_module(module)
        module._physical_arm_native_identity = identity
        setattr(package, TARGET, module)
    except BaseException:
        sys.modules.pop(MODULE, None)
        raise
    return receipt


def prepare(profile: str, architecture: str, jobs: int = 2) -> dict:
    """Explicit compilation into this checkout's ignored, profile-private cache."""
    directory = _profile_dir(profile)
    if architecture not in {"87", "120"}:
        raise ValueError("CUDA architecture must be explicitly 120 (RTX5060) or 87 (Jetson)")
    if jobs < 1:
        raise ValueError("jobs must be positive")
    wrapper = directory / "cmake-source"
    build_dir = directory / "cmake-build"
    output = directory / "module"
    wrapper.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    # Invalidate any previous receipt before touching its artifact/build state.
    (directory / "receipt.json").unlink(missing_ok=True)
    source = _source_identity()
    definitions = " ".join(f"{key}={value}" for key, value in PROFILES[profile].items())
    # CMAKE_SOURCE_DIR in the repository points at this wrapper. Supply the
    # repository include path explicitly; all other source paths are relative to
    # the repository subdirectory. No source-tree or installed module is written.
    (wrapper / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.24)\n'
        'project(PhysicalArmNative LANGUAGES CXX CUDA)\n'
        'add_subdirectory("${GATO_ROOT}" "${CMAKE_BINARY_DIR}/gato")\n'
        f'target_include_directories({TARGET} PRIVATE "${{GATO_ROOT}}/gato")\n'
        f'target_compile_definitions({TARGET} PRIVATE {definitions})\n'
        f'set_target_properties({TARGET} PROPERTIES\n'
        '  LIBRARY_OUTPUT_DIRECTORY "${PROFILE_OUTPUT}"\n'
        '  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROFILE_OUTPUT}")\n')
    configure = ["cmake", "-S", str(wrapper), "-B", str(build_dir), "-G", "Unix Makefiles",
                 f"-DGATO_ROOT={ROOT}", f"-DPROFILE_OUTPUT={output}",
                 "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CUDA_ARCHITECTURES={architecture}",
                 f"-DPython3_EXECUTABLE={sys.executable}", "-DKNOTS=32", "-DPLANT=tiago_right",
                 "-DBUILD_TESTING=OFF", "-DCMAKE_CXX_FLAGS=", "-DCMAKE_CUDA_FLAGS=",
                 "-DCMAKE_SHARED_LINKER_FLAGS=", "-DCMAKE_MODULE_LINKER_FLAGS="]
    build = ["cmake", "--build", str(build_dir), "--config", "Release",
             "--target", TARGET, "--parallel", str(jobs)]
    subprocess.run(configure, cwd=ROOT, check=True)
    subprocess.run(build, cwd=ROOT, check=True)
    if _source_identity() != source:
        raise RuntimeError("Relevant source changed during compilation; rerun preparation")
    module = output / (TARGET + _python_identity()["extension_suffix"])
    receipt = {"schema": 1, "prepared_at": datetime.now(timezone.utc).isoformat(),
               "config": _config(profile, architecture), "source": source,
               "python": _python_identity(), "platform": _platform_identity(),
               "module": {"path": str(module), "sha256": _sha256(module)},
               "build": {"directory": str(build_dir), "configure_command": configure,
                         "build_command": build,
                         "cmake_cache_sha256": _sha256(build_dir / "CMakeCache.txt")}}
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return verified_receipt(profile)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-architecture", required=True, choices=("120", "87"),
                        help="explicit target: 120 RTX5060, 87 Jetson; no GPU probing")
    parser.add_argument("--profile", choices=tuple(PROFILES), help="default: prepare both profiles")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    for profile in ([args.profile] if args.profile else PROFILES):
        receipt = prepare(profile, args.cuda_architecture, args.jobs)
        print(f"Prepared {profile}: {receipt['module']['path']} sha256={receipt['module']['sha256']}")


if __name__ == "__main__":
    main()
