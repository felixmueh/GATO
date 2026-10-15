"""CPU-only native preparation/selection checks; never compile or import CUDA."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "tiago_examples/physical_arm/native_build.py"
SPEC = importlib.util.spec_from_file_location("physical_native_build_test", SCRIPT)
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "ROOT", tmp_path)
    monkeypatch.setattr(native, "CACHE", tmp_path / "build/physical-arm-native")
    for name in ("CMakeLists.txt", "python/bindings.cu", "python/bsqp/interface.py",
                 "gato/settings.h", "gato/dynamics/tiago_right/tiago_right_grid.cuh",
                 "tiago_examples/physical_arm/native_build.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original source\n")
    monkeypatch.setattr(native, "__file__", str(tmp_path / "tiago_examples/physical_arm/native_build.py"))
    monkeypatch.delitem(sys.modules, native.MODULE, raising=False)
    return tmp_path


@pytest.fixture
def compiler(checkout, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs == {"cwd": checkout, "check": True}
        if command[1] == "-S":
            build_dir = Path(command[command.index("-B") + 1])
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\n")
        else:
            build_dir = Path(command[2])
            output = build_dir.parent / "module"
            artifact = output / (native.TARGET + native._python_identity()["extension_suffix"])
            artifact.write_bytes(build_dir.parent.name.encode())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(native.subprocess, "run", run)
    return calls


def test_prepare_separates_profiles_and_redirects_all_artifacts(checkout, compiler):
    installed = checkout / "python/bsqp" / (native.TARGET + ".so")
    installed.write_bytes(b"existing installed module")
    tracking = native.prepare("tracking", "120")
    reaching = native.prepare("reaching", "120")
    assert installed.read_bytes() == b"existing installed module"
    assert tracking["module"]["path"] != reaching["module"]["path"]
    assert tracking["module"]["sha256"] != reaching["module"]["sha256"]
    assert tracking["config"]["definitions"] == native.PROFILES["tracking"]
    assert reaching["config"]["definitions"] == native.PROFILES["reaching"]
    for profile in native.PROFILES:
        directory = native._profile_dir(profile)
        wrapper = (directory / "cmake-source/CMakeLists.txt").read_text()
        assert 'add_subdirectory("${GATO_ROOT}"' in wrapper
        assert 'LIBRARY_OUTPUT_DIRECTORY "${PROFILE_OUTPUT}"' in wrapper
        assert 'LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROFILE_OUTPUT}"' in wrapper
        for key, value in native.PROFILES[profile].items():
            assert f"{key}={value}" in wrapper
    assert len(compiler) == 4
    for command in compiler[::2]:
        assert "-DCMAKE_CUDA_ARCHITECTURES=120" in command
        assert "-DCMAKE_BUILD_TYPE=Release" in command
        assert "-DBUILD_TESTING=OFF" in command
    assert all("--target" in command and native.TARGET in command for command in compiler[1::2])


@pytest.mark.parametrize("mutation", ["source", "generated_header", "module", "receipt", "flags", "cache", "missing"])
def test_stale_or_missing_profile_has_explicit_prepare_command(checkout, compiler, mutation):
    receipt = native.prepare("reaching", "87")
    directory = native._profile_dir("reaching")
    if mutation == "source":
        (checkout / "gato/settings.h").write_text("dirty relevant source")
    elif mutation == "generated_header":
        (checkout / "gato/dynamics/tiago_right/tiago_right_grid.cuh").write_text("new GRiD")
    elif mutation == "module":
        Path(receipt["module"]["path"]).write_bytes(b"different module")
    elif mutation == "receipt":
        (directory / "receipt.json").write_text("invalid json")
    elif mutation == "flags":
        receipt["config"]["definitions"] = native.PROFILES["tracking"]
        (directory / "receipt.json").write_text(json.dumps(receipt))
    elif mutation == "cache":
        (directory / "cmake-build/CMakeCache.txt").write_text("changed flags")
    else:
        (directory / "receipt.json").unlink()
    with pytest.raises(RuntimeError, match="Prepare native modules explicitly") as exc:
        native.load_profile("reaching")
    assert "--cuda-architecture" in str(exc.value)
    assert len(compiler) == 2  # loading never triggers a build


def test_loading_same_profile_repeats_but_other_profile_is_forbidden(checkout, compiler, monkeypatch):
    native.prepare("tracking", "120")
    native.prepare("reaching", "120")
    package = ModuleType("bsqp")
    module = ModuleType(native.MODULE)
    executions = []
    loader = SimpleNamespace(exec_module=lambda selected: executions.append(selected))
    monkeypatch.setattr(native.importlib, "import_module", lambda name: package)
    monkeypatch.setattr(native.importlib.util, "spec_from_file_location",
                        lambda name, path: SimpleNamespace(loader=loader))
    monkeypatch.setattr(native.importlib.util, "module_from_spec", lambda spec: module)
    tracking = native.load_profile("tracking")
    assert native.load_profile("tracking") == tracking
    assert sys.modules[native.MODULE] is module
    assert getattr(package, native.TARGET) is module
    assert executions == [module]
    with pytest.raises(RuntimeError, match="different or unverified profile"):
        native.load_profile("reaching")
    assert len(executions) == 1


def test_previously_imported_unverified_module_is_rejected(checkout, compiler, monkeypatch):
    native.prepare("tracking", "87")
    monkeypatch.setitem(sys.modules, native.MODULE, ModuleType(native.MODULE))
    with pytest.raises(RuntimeError, match="fresh Python process"):
        native.load_profile("tracking")


def test_unrelated_results_do_not_invalidate_and_failed_build_removes_receipt(checkout, compiler, monkeypatch):
    native.prepare("tracking", "87")
    (checkout / "study-results.txt").write_text("unrelated study")
    native.verified_receipt("tracking")

    def fail(*args, **kwargs):
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(native.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="compiler failed"):
        native.prepare("tracking", "87")
    assert not (native._profile_dir("tracking") / "receipt.json").exists()


def test_architecture_requires_explicit_supported_choice(checkout, compiler):
    with pytest.raises(ValueError, match="explicitly"):
        native.prepare("tracking", "native")
    assert not compiler
