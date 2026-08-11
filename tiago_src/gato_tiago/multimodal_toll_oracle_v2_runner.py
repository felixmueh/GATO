"""Transactional full-campaign runner for Tiago tool-center Oracle V2.

The accepted V1 scientific implementation is called under an explicit V2
namespace adapter.  The adapter changes no scientific constant or numerical
function: it adds the accepted prerequisite artifact gate, uses producer-native
hashing only for predecessor summaries, and owns the V2 artifact/worker paths.
All execution capabilities are disabled in this static checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping

import numpy as np

from gato_tiago import multimodal_toll_oracle_v1_runner as _v1
from gato_tiago.multimodal_toll_oracle_v2 import producer_array_hash
from gato_tiago.multimodal_toll_oracle_v2_full import *  # noqa: F401,F403
from gato_tiago import multimodal_toll_oracle_v2_worker as _worker


RUNNER_EXECUTION_AUTHORIZATION = None
RUNNER_PROTOCOL_VERSION = ORACLE_PROTOCOL_VERSION + "_runner_1"
_PRODUCTION_TOKEN = object()
WORKER_MODULE = "gato_tiago.multimodal_toll_oracle_v2_worker"
_V1_CERTIFY_PROVENANCE = _v1.certify_provenance
_V1_RECERTIFY_RETAINED_ORACLE = _v1.recertify_retained_oracle
PREREQUISITE_RECERTIFICATION_GATE_NAMES = (
    "task_producer_and_pure_recert",
    "model_producer_and_pure_recert",
    "cross_bind",
    "stored_detail_equals_independent_recomputation",
    "retained_document_and_checkpoint_chain",
)

REQUIRED_SOURCE_PATHS = {
    **{
        key: value
        for key, value in _v1.REQUIRED_SOURCE_PATHS.items()
        if key not in {"oracle_schema", "oracle_runner", "oracle_worker", "oracle_tests"}
    },
    "oracle_v2_schema": "tiago_src/gato_tiago/multimodal_toll_oracle_v2_full.py",
    "oracle_v2_runner": "tiago_src/gato_tiago/multimodal_toll_oracle_v2_runner.py",
    "oracle_v2_worker": "tiago_src/gato_tiago/multimodal_toll_oracle_v2_worker.py",
    "oracle_v2_tests": "tests/python/test_tiago_multimodal_toll_oracle_v2.py",
    "accepted_v1_schema": "tiago_src/gato_tiago/multimodal_toll_oracle_v1.py",
    "accepted_v1_runner": "tiago_src/gato_tiago/multimodal_toll_oracle_v1_runner.py",
    "accepted_v1_worker": "tiago_src/gato_tiago/multimodal_toll_oracle_v1_worker.py",
    "prerequisite_schema": "tiago_src/gato_tiago/multimodal_toll_oracle_v2.py",
    "prerequisite_runner": "tiago_src/gato_tiago/multimodal_toll_oracle_v2_prerequisite_runner.py",
}

# Public aliases make the byte-identical scientific implementation explicit.
PinOracleModel = _v1.PinOracleModel
ProductionOracleDependenciesV1 = _v1.ProductionOracleDependencies
OriginalOracleNLP = _v1.OriginalOracleNLP
solve_anchored_acquisition = _v1.solve_anchored_acquisition
solve_unanchored_polish = _v1.solve_unanchored_polish
certify_retained_row = _v1.certify_retained_row
aggregate_campaign = _v1.aggregate_campaign
bootstrap_heldout_gaps = _v1.bootstrap_heldout_gaps


def sha256_file(path):
    return _v1.sha256_file(path)


def authenticate_summary(summary, arrays, *, kind):
    """Authenticate only predecessor summaries with their producer convention."""

    expected = TASK_ARRAY_COUNT if kind == "task" else MODEL_ARRAY_COUNT
    certificate_key = (
        "all_v4_gates_pass" if kind == "task" else "all_model_preflight_gates_pass"
    )
    names = summary.get("array_names")
    hashes = summary.get("array_hashes")
    return bool(
        summary.get("incomplete") is False
        and summary.get(certificate_key) is True
        and summary.get("certificate", {}).get(certificate_key) is True
        and isinstance(names, list)
        and names == sorted(arrays)
        and len(names) == len(set(names)) == expected
        and isinstance(hashes, Mapping)
        and set(hashes) == set(arrays)
        and all(hashes[name] == producer_array_hash(arrays[name]) for name in names)
    )


def authenticate_prerequisite_artifact():  # pragma: no cover
    """Hard-pin and independently recertify the accepted CPU preflight."""

    from gato_tiago.multimodal_toll_oracle_v2_prerequisite_runner import (
        recertify_retained_prerequisite,
    )

    pin_results = {}
    for name, (path, digest) in PREREQUISITE_ARTIFACT_PINS.items():
        actual = sha256_file(path) if path.is_file() else None
        pin_results[name] = {
            "path": str(path), "expected_sha256": digest,
            "actual_sha256": actual, "passes": actual == digest,
        }
    recertification = recertify_retained_prerequisite()
    result = {
        "pin_results": pin_results,
        "all_four_pins_pass": all(row["passes"] for row in pin_results.values()),
        "independent_recertification": recertification,
        "passes": bool(
            all(row["passes"] for row in pin_results.values())
            and recertification.get("passes") is True
        ),
    }
    if not result["passes"]:
        raise RuntimeError("Oracle V2 accepted prerequisite authentication failed")
    return result


def certify_prerequisite_authentication_detail(detail):
    """Require the complete canonical result of the accepted CPU preflight."""

    if not isinstance(detail, Mapping) or set(detail) != {
        "pin_results",
        "all_four_pins_pass",
        "independent_recertification",
        "passes",
    }:
        return False
    pin_results = detail.get("pin_results")
    if not isinstance(pin_results, Mapping) or set(pin_results) != set(
        PREREQUISITE_ARTIFACT_PINS
    ):
        return False
    for name, (path, digest) in PREREQUISITE_ARTIFACT_PINS.items():
        row = pin_results.get(name)
        if not isinstance(row, Mapping) or set(row) != {
            "path", "expected_sha256", "actual_sha256", "passes"
        }:
            return False
        if row != {
            "path": str(path),
            "expected_sha256": digest,
            "actual_sha256": digest,
            "passes": True,
        }:
            return False
    recertification = detail.get("independent_recertification")
    if not isinstance(recertification, Mapping) or set(recertification) != {
        "gates", "passes"
    }:
        return False
    gates = recertification.get("gates")
    return bool(
        isinstance(gates, Mapping)
        and tuple(gates) == PREREQUISITE_RECERTIFICATION_GATE_NAMES
        and all(gates[name] is True for name in PREREQUISITE_RECERTIFICATION_GATE_NAMES)
        and recertification.get("passes") is True
        and detail.get("all_four_pins_pass") is True
        and detail.get("passes") is True
    )


def start_provenance(root=None, *, orig_argv=None):
    with _v2_runner_context(include_certifier=False):
        provenance = _v1.start_provenance(root, orig_argv=orig_argv)
    provenance.update(
        {
            "prerequisite_artifact_pins": {
                name: {"path": str(path), "sha256": digest}
                for name, (path, digest) in PREREQUISITE_ARTIFACT_PINS.items()
            },
            "prerequisite_artifact_loads": 0,
            "prerequisite_artifact_authentication": None,
            "rejected_v1_artifact_loads": 0,
            "v1_rejected_report_only": V1_REJECTED_REPORT_ONLY,
        }
    )
    return provenance


def certify_provenance(provenance):
    with _v2_runner_context(include_certifier=False):
        inherited = _V1_CERTIFY_PROVENANCE(provenance)
    detail = provenance.get("prerequisite_artifact_authentication")
    return bool(
        inherited
        and provenance.get("prerequisite_artifact_pins")
        == {
            name: {"path": str(path), "sha256": digest}
            for name, (path, digest) in PREREQUISITE_ARTIFACT_PINS.items()
        }
        and provenance.get("prerequisite_artifact_loads") == 1
        and certify_prerequisite_authentication_detail(detail)
        and provenance.get("rejected_v1_artifact_loads") == 0
        and provenance.get("v1_rejected_report_only") == V1_REJECTED_REPORT_ONLY
    )


def certify_worker_row_boundary(worker, retained, public, identity=None):
    expected = {
        "captured_x0_float32": np.asarray(public["public_x0_float32"], np.float32),
        "captured_reference_float32": np.asarray(public["public_reference_float32"], np.float32),
        "captured_controls_float32": np.asarray(retained["controls_float64"], np.float32),
        "captured_joint_lower_float64": np.asarray(public["joint_lower_float64"], np.float64),
        "captured_joint_upper_float64": np.asarray(public["joint_upper_float64"], np.float64),
        "captured_velocity_limit_float64": np.asarray(public["velocity_limit_float64"], np.float64),
        "captured_effort_limit_float64": np.asarray(public["effort_limit_float64"], np.float64),
        "cuda_knot_states_float32": np.asarray(retained["cuda_dense_state_float32"])[::DENSE_SUBSTEPS],
        "cuda_dense_states_float32": np.asarray(retained["cuda_dense_state_float32"]),
        "cuda_dense_tool_float32": np.asarray(retained["cuda_dense_tool_float32"]),
        "dense_time_float64": np.asarray(retained["dense_time_float64"]),
    }
    hashes = worker.get("array_hashes", {})
    artifact = worker.get("artifact", {})
    command = worker.get("command", ())
    return bool(
        worker.get("protocol") == _worker.WORKER_PROTOCOL_VERSION
        and isinstance(command, list)
        and command[1:4] == ["-B", "-m", WORKER_MODULE]
        and command[4:] == ["--request", artifact.get("request_path"), "--output", artifact.get("json_path")]
        and (identity is None or worker.get("identity") == list(identity))
        and worker.get("cwd") == str(AUTHORIZED_CWD)
        and worker.get("exit_code") == 0
        and worker.get("constructor_calls") == 2
        and worker.get("sim_forward_calls") == DENSE_SAMPLES - 1
        and worker.get("tool_position_calls") == ((DENSE_SAMPLES + 15) // 16)
        and worker.get("solve_calls") == worker.get("sqp_calls") == 0
        and set(hashes) == set(expected)
        and all(hashes[name] == _worker.array_hash(value) for name, value in expected.items())
        and set(artifact) >= {"request_path", "request_sha256", "input_path", "input_sha256", "json_path", "json_sha256", "npz_path", "npz_sha256"}
    )


@contextmanager
def _v2_runner_context(*, include_certifier=True):
    replacements = {
        "ORACLE_PROTOCOL_VERSION": ORACLE_PROTOCOL_VERSION,
        "ORACLE_OUTPUT_PATH": ORACLE_OUTPUT_PATH,
        "AUTHORIZED_ORIG_ARGV": AUTHORIZED_ORIG_ARGV,
        "RUNNER_PROTOCOL_VERSION": RUNNER_PROTOCOL_VERSION,
        "REQUIRED_SOURCE_PATHS": REQUIRED_SOURCE_PATHS,
        "WORKER_PROTOCOL_VERSION": _worker.WORKER_PROTOCOL_VERSION,
        "EXPECTED_REPLAY_ARRAY_NAMES": _worker.EXPECTED_REPLAY_ARRAY_NAMES,
        "EXPECTED_WORKER_SUMMARY_KEYS": _worker.EXPECTED_WORKER_SUMMARY_KEYS,
        "validate_request": _worker.validate_request,
        "certify_replay": _worker.certify_replay,
        "authenticate_summary": authenticate_summary,
        "certify_worker_row_boundary": certify_worker_row_boundary,
    }
    if include_certifier:
        replacements["certify_provenance"] = certify_provenance
    previous = {name: getattr(_v1, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(_v1, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(_v1, name, value)


class ProductionOracleDependencies(_v1.ProductionOracleDependencies):  # pragma: no cover
    """V1 numerical dependencies with an isolated V2 worker subprocess."""

    def replay(self, identity, polish_arrays, task_index, base_arrays):
        index = EXPECTED_LEDGER.index(tuple(identity))
        stem = f"{self.output.stem}.worker.{index:04d}"
        request_path = self.output.with_name(stem + ".request.json")
        input_path = self.output.with_name(stem + ".input.npz")
        output_json = self.output.with_name(stem + ".json")
        output_npz = self.output.with_name(stem + ".npz")
        if any(path.exists() for path in (request_path, input_path, output_json, output_npz)):
            raise RuntimeError("Oracle V2 worker side artifact exists")
        worker_input = {
            "x0_float32": np.asarray(base_arrays["public_x0_float32"][task_index], np.float32),
            "reference_float32": np.asarray(base_arrays["public_reference_float32"][task_index], np.float32),
            "controls_float32": np.asarray(polish_arrays["controls_float64"], np.float32),
            "joint_lower_float64": base_arrays["joint_lower_float64"],
            "joint_upper_float64": base_arrays["joint_upper_float64"],
            "velocity_limit_float64": base_arrays["velocity_limit_float64"],
            "effort_limit_float64": base_arrays["effort_limit_float64"],
        }
        _v1._atomic_npz(input_path, worker_input)
        request = {
            "identity": list(identity), "request_json": str(request_path),
            "input_npz": str(input_path), "input_npz_sha256": sha256_file(input_path),
            "extension_module": CUDA_EXTENSION_MODULE,
            "extension_path": str(CUDA_EXTENSION_PATH),
            "extension_sha256": CUDA_EXTENSION_SHA256,
            "worker_protocol": _worker.WORKER_PROTOCOL_VERSION,
            "output_json": str(output_json), "output_npz": str(output_npz),
        }
        _v1._atomic_json(request_path, request)
        command = [sys.executable, "-B", "-m", WORKER_MODULE, "--request", str(request_path), "--output", str(output_json)]
        completed = subprocess.run(
            command, cwd=AUTHORIZED_CWD,
            env={**os.environ, "PYTHONPATH": "tiago_src:python"},
            text=True, capture_output=True,
        )
        self.provenance["worker_subprocess_calls"] += 1
        if completed.returncode != 0:
            raise RuntimeError(f"Oracle V2 worker failed: {completed.returncode}: {completed.stderr}")
        summary = json.loads(output_json.read_text())
        with np.load(output_npz, allow_pickle=False) as archive:
            worker_arrays = {name: archive[name] for name in archive.files}
        if not _worker.validate_request(request) or not _worker.certify_replay(summary, worker_arrays)["passes"]:
            raise RuntimeError("Oracle V2 worker boundary recertification failed")
        self.provenance["cuda_replay_calls"] += 1
        artifact = {
            "request_path": str(request_path), "request_sha256": sha256_file(request_path),
            "input_path": str(input_path), "input_sha256": sha256_file(input_path),
            "json_path": str(output_json), "json_sha256": sha256_file(output_json),
            "npz_path": str(output_npz), "npz_sha256": sha256_file(output_npz),
        }
        summary["artifact"] = artifact
        summary.update({"command": command, "cwd": str(AUTHORIZED_CWD), "exit_code": completed.returncode,
                        "stdout": completed.stdout, "stderr": completed.stderr})
        return summary, {
            "cuda_dense_state_float32": worker_arrays["cuda_dense_states_float32"],
            "cuda_dense_tool_float32": worker_arrays["cuda_dense_tool_float32"],
        }


def _run_pipeline(output, *, provenance, prerequisite_authenticator: Callable,
                  task_loader: Callable, model_loader: Callable,
                  acquisition_solver: Callable, polish_solver: Callable,
                  replay_solver: Callable, finish_provenance: Callable,
                  test_override: bool, **kwargs):
    """Authenticate the accepted CPU preflight before entering V1 science."""

    authentication = prerequisite_authenticator()
    if authentication.get("passes") is not True:
        raise RuntimeError("Oracle V2 prerequisite authentication failed")
    provenance["prerequisite_artifact_loads"] = 1
    provenance["prerequisite_artifact_authentication"] = authentication
    with _v2_runner_context():
        return _v1._run_pipeline(
            output, provenance=provenance, task_loader=task_loader,
            model_loader=model_loader, acquisition_solver=acquisition_solver,
            polish_solver=polish_solver, replay_solver=replay_solver,
            finish_provenance=finish_provenance, test_override=test_override,
            **kwargs,
        )


def recertify_retained_oracle(summary_path, npz_path, manifest_path, pointer_path):  # pragma: no cover
    """Fail-closed disk recertification with fresh prerequisite cross-binding."""

    try:
        authentication = authenticate_prerequisite_artifact()
        summary = json.loads(Path(summary_path).read_text())
        retained = summary.get("provenance", {}).get(
            "prerequisite_artifact_authentication"
        )
        prerequisite_exact = bool(
            certify_prerequisite_authentication_detail(authentication)
            and retained == authentication
        )
        if not prerequisite_exact:
            return {
                "prerequisite_artifact_authentication_exact": False,
                "all_disk_recertification_gates_pass": False,
            }
        with _v2_runner_context():
            result = dict(
                _V1_RECERTIFY_RETAINED_ORACLE(
                    summary_path, npz_path, manifest_path, pointer_path
                )
            )
        result["prerequisite_artifact_authentication_exact"] = True
        result["all_disk_recertification_gates_pass"] = bool(
            result.get("all_disk_recertification_gates_pass") is True
        )
        return result
    except Exception:
        return {
            "prerequisite_artifact_authentication_exact": False,
            "all_disk_recertification_gates_pass": False,
        }


def _production_pipeline(output, provenance):  # pragma: no cover
    dependencies = ProductionOracleDependencies(output, provenance)
    return _run_pipeline(
        output, provenance=provenance,
        prerequisite_authenticator=authenticate_prerequisite_artifact,
        task_loader=dependencies.task_loader,
        model_loader=dependencies.model_loader,
        acquisition_solver=dependencies.acquisition,
        polish_solver=dependencies.polish,
        replay_solver=dependencies.replay,
        finish_provenance=_v1.finish_runner_provenance,
        test_override=False,
        final_certifier=lambda rows, arrays, current: _v1.certify_final_oracle(
            rows, arrays, current, dependencies.problem_factory, reenumerate=True
        ),
        before_finish=dependencies.before_finish,
        campaign_deadline_setter=dependencies.set_campaign_deadline,
    )


def execute_oracle(output, authorization=None):  # pragma: no cover
    if (
        RUNNER_EXECUTION_AUTHORIZATION is None
        or authorization is not RUNNER_EXECUTION_AUTHORIZATION
    ):
        raise RuntimeError("Oracle V2 execution is blocked")
    output = Path(output)
    if output.resolve() != ORACLE_OUTPUT_PATH.resolve():
        raise RuntimeError("Oracle V2 output path not authorized")
    if output.parent.exists():
        raise RuntimeError("Oracle V2 output root must be absent")
    with _v2_runner_context():
        root = _v1.repository_root()
        provenance = start_provenance(root)
    if not provenance["tracked_tree_clean_at_start"]:
        raise RuntimeError("Oracle V2 requires tracked-clean source")
    return _production_pipeline(output, provenance)


def main(argv=None):  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not args.execute or RUNNER_EXECUTION_AUTHORIZATION is None:
        raise SystemExit("Oracle V2 execution is blocked")
    execute_oracle(args.output, RUNNER_EXECUTION_AUTHORIZATION)


if __name__ == "__main__":
    main()
