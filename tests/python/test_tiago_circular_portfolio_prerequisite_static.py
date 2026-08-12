import copy
import inspect
import json

import numpy as np
import pytest

from gato_tiago import circular_portfolio as schema
from gato_tiago import circular_portfolio_prerequisite_runner as prerequisite


def test_only_cpu_prerequisite_capability_is_enabled_and_path_is_frozen():
    assert prerequisite.RUNNER_EXECUTION_AUTHORIZATION is not None
    assert prerequisite.OUTPUT == prerequisite.Path(
        "/tmp/tiago-tool-center-circular-portfolio-p1-prerequisite-authorized-once/prerequisite.json"
    )
    with pytest.raises(RuntimeError, match="blocked"):
        prerequisite.execute(
            prerequisite.OUTPUT,
            authorization=object(),
        )
    with pytest.raises(RuntimeError, match="not authorized"):
        prerequisite.execute(
            prerequisite.OUTPUT.with_name("wrong.json"),
            authorization=prerequisite.RUNNER_EXECUTION_AUTHORIZATION,
        )


def _synthetic_bundle():
    task_arrays = {}
    rows = []
    for index, identity in enumerate(schema.TASK_IDENTITIES):
        seed = identity[1]
        x0 = np.zeros(14, np.float32)
        x0[1] = np.float32(index * 0.001)
        history = np.zeros((9, 7), np.float64)
        history[:, 1] = index * 0.001
        history[8, 0] = 0.1
        reference = np.zeros(10, np.float32)
        reference[:3] = np.asarray((0.1, index * 0.001, 0.0), np.float32)
        task_arrays[f"task_{seed}_x0_float32"] = x0
        task_arrays[f"task_{seed}_reference_float32"] = reference
        task_arrays[f"task_{seed}_history_q_float64"] = history
        rows.append({"public_task": {"default_side": -1 if index % 2 else 1}})
    model_arrays = {
        "model_lower_float64": -np.ones(7),
        "model_upper_float64": np.ones(7),
        "model_velocity_float64": np.ones(7),
        "model_effort_float64": 10 * np.ones(7),
    }
    def kinematics(q):
        return np.asarray(q[:3], np.float64), np.c_[np.eye(3), np.zeros((3, 4))]
    return {"rows": rows}, task_arrays, model_arrays, kinematics


def _provenance(final=False):
    sources = {path: "a" * 64 for path in prerequisite.SOURCE_PATHS}
    return {
        "protocol": prerequisite.PROTOCOL, "cwd": str(prerequisite.AUTHORIZED_CWD),
        "orig_argv": list(prerequisite.AUTHORIZED_ORIG_ARGV),
        "exact_command": __import__("shlex").join(prerequisite.AUTHORIZED_ORIG_ARGV),
        "git_head_at_start": "b" * 40, "git_head_at_end": "b" * 40 if final else None,
        "tracked_clean_at_start": True, "tracked_clean_at_end": True if final else None,
        "source_hashes_at_start": sources, "source_hashes_at_end": sources if final else None,
        "build_head": prerequisite.FROZEN_BUILD_HEAD,
        "cuda_arch": prerequisite.FROZEN_CUDA_ARCH,
        "extension_at_start": prerequisite._extension_measurement(),
        "extension_at_end": prerequisite._extension_measurement() if final else None,
        "runtime_versions": prerequisite._runtime_versions(),
    }


def test_build_pin_and_cpu_only_boundary_are_exact_and_authorized_once(tmp_path):
    assert prerequisite.RUNNER_EXECUTION_AUTHORIZATION is not None
    assert prerequisite.FROZEN_EXTENSION == {
        "path": "/workspace/GATO/python/bsqp/bsqpN96_tiago_right_circular_portfolio_toll.cpython-310-x86_64-linux-gnu.so",
        "sha256": "b079410ade9e7de19ed3d4b7ed6f6ace27172cd442ccea0d5a46bb7277067a2f",
        "size_bytes": 6_690_480, "KNOT_POINTS": 96, "REFERENCE_SIZE": 10,
        "TOOL_POSITION_FRAME": "arm_right_tool_joint_origin", "TOOL_POSITION_SIZE": 3,
        "B1": True, "B16": True,
    }
    with pytest.raises(RuntimeError, match="blocked"):
        prerequisite.execute(tmp_path / "prerequisite.json", authorization=object())
    source = inspect.getsource(prerequisite.execute)
    assert source.index("publish_checkpoint") < source.index("authenticate_task_artifact")
    assert "production_kinematics" in source
    assert "optimizer" not in source and "worker" not in source and "cuda" not in source.lower()


def test_exact12_geometry_and_all192_references_reconstruct_and_mutate_fail():
    summary, task, model, kinematics = _synthetic_bundle()
    arrays, details, primary_calls = prerequisite.construct_geometry_arrays(
        summary, task, model, kinematics
    )
    certificate = prerequisite.certify_geometry_arrays(arrays, summary, task, model, kinematics)
    assert certificate["passes"] and len(details) == 12
    assert primary_calls == certificate["recert_call_counts"] == {
        "pin_fk_calls": 24, "geometry_constructions": 204,
        "reference_constructions": 204,
    }
    assert arrays["route_tool_reference_float64"].shape == (192, 96, 3)
    assert np.array_equal(arrays["ledger_seed_int64"], np.asarray([row[1] for row in schema.EXPECTED_LEDGER]))
    for name in (
        "public_x0_float32", "accepted_task_reference_float32", "quarantined_q8_float64",
        "public_default_side_int8", "q0_tool_float64", "q8_tool_float64",
        "portfolio_reference_float32", "route_tool_reference_float64",
        "joint_lower_float64", "velocity_limit_float64",
    ):
        bad = {key: value.copy() for key, value in arrays.items()}
        bad[name].flat[0] += 1
        assert not prerequisite.certify_geometry_arrays(
            bad, summary, task, model, kinematics
        )["passes"], name


def test_checkpoint_stages_have_exact_counters_auth_and_array_schema():
    summary, task, model, kinematics = _synthetic_bundle()
    arrays, _, _ = prerequisite.construct_geometry_arrays(summary, task, model, kinematics)
    geometry = prerequisite.certify_geometry_arrays(arrays, summary, task, model, kinematics)
    task_auth = {"passes": True}; model_auth = {"passes": True}; cross = {"passes": True}
    for generation, stage in enumerate((
        "gen0", "task_authenticated", "model_cross_bound", "geometry_authenticated", "honest_end"
    )):
        retained = {} if generation < 3 else arrays
        doc = prerequisite.checkpoint_document(
            generation, stage, prerequisite._expected_counters(generation),
            _provenance(final=generation == 4), retained,
            task_auth=None if generation == 0 else task_auth,
            model_auth=None if generation < 2 else model_auth,
            cross_bind=None if generation < 2 else cross,
            geometry_certificate=None if generation < 3 else geometry,
        )
        owned = {"task": None if generation == 0 else task_auth,
                 "model": None if generation < 2 else model_auth,
                 "cross_bind": None if generation < 2 else cross,
                 "geometry": None if generation < 3 else geometry}
        assert prerequisite.certify_checkpoint(doc, retained, generation, owned)
        bad = copy.deepcopy(doc); bad["counters"]["optimizer_calls"] = 1
        assert not prerequisite.certify_checkpoint(bad, retained, generation, owned)
        if generation >= 1:
            bad = copy.deepcopy(doc); bad["task_authentication"]["fresh_lie"] = True
            assert not prerequisite.certify_checkpoint(bad, retained, generation, owned)
        if generation >= 3:
            bad = {name: value.copy() for name, value in retained.items()}
            bad["route_tool_reference_float64"][0, 1, 0] += 1
            assert not prerequisite.certify_checkpoint(doc, bad, generation, owned)


def test_geometry_call_ledger_spies_reject_extra_or_missing_calls(monkeypatch):
    summary, task, model, kinematics = _synthetic_bundle()
    calls = {"fk": 0}
    def spy(q):
        calls["fk"] += 1
        return kinematics(q)
    arrays, _details, primary = prerequisite.construct_geometry_arrays(summary, task, model, spy)
    assert calls["fk"] == primary["pin_fk_calls"] == 24
    certificate = prerequisite.certify_geometry_arrays(arrays, summary, task, model, spy)
    assert certificate["passes"] and calls["fk"] == 48
    original = prerequisite.construct_geometry_arrays
    for delta in (-1, 1):
        def altered(*args, _delta=delta, **kwargs):
            regenerated, details, retained = original(*args, **kwargs)
            retained["pin_fk_calls"] += _delta
            return regenerated, details, retained
        monkeypatch.setattr(prerequisite, "construct_geometry_arrays", altered)
        assert not prerequisite.certify_geometry_arrays(
            arrays, summary, task, model, spy
        )["passes"]
        monkeypatch.setattr(prerequisite, "construct_geometry_arrays", original)


def test_extension_and_runtime_provenance_are_measured_not_self_reported():
    provenance = _provenance(final=True)
    assert prerequisite.certify_provenance(provenance, final=True)
    for mutation in (
        lambda value: value["extension_at_start"].__setitem__("sha256", "0" * 64),
        lambda value: value["extension_at_end"].__setitem__("size_bytes", 1),
        lambda value: value["extension_at_start"]["accepted_attributes"].__setitem__("KNOT_POINTS", 64),
        lambda value: value["runtime_versions"].__setitem__("numpy", "wrong"),
    ):
        bad = copy.deepcopy(provenance); mutation(bad)
        assert not prerequisite.certify_provenance(bad, final=True)
    wrong = copy.deepcopy(prerequisite._extension_measurement())
    wrong.update({"path": "/tmp/self-consistent-wrong.so", "sha256": "e" * 64,
                  "size_bytes": 123})
    assert not prerequisite.certify_extension_measurement(wrong)
    self_consistent = copy.deepcopy(provenance)
    self_consistent["extension_at_start"] = wrong
    self_consistent["extension_at_end"] = copy.deepcopy(wrong)
    assert not prerequisite.certify_provenance(self_consistent, final=True)


def test_provenance_rejects_monkeypatched_self_consistent_wrong_live_measurement(monkeypatch):
    provenance = _provenance(final=True)
    wrong = copy.deepcopy(provenance["extension_at_start"])
    wrong["path"] = "/tmp/replaced-before-start.so"
    wrong["sha256"] = "f" * 64
    wrong["size_bytes"] += 1
    provenance["extension_at_start"] = copy.deepcopy(wrong)
    provenance["extension_at_end"] = copy.deepcopy(wrong)
    monkeypatch.setattr(prerequisite, "_extension_measurement", lambda: copy.deepcopy(wrong))
    assert not prerequisite.certify_provenance(provenance, final=True)


def test_final_certificate_binds_candidate_npz_path_hash_and_exact_fresh_details(tmp_path):
    task_summary, task_arrays, model_arrays, kinematics = _synthetic_bundle()
    arrays, _details, _calls = prerequisite.construct_geometry_arrays(
        task_summary, task_arrays, model_arrays, kinematics
    )
    semantic = prerequisite.certify_geometry_arrays(
        arrays, task_summary, task_arrays, model_arrays, kinematics
    )
    task_auth = {"passes": True, "array_hashes": {"task": "a" * 64}}
    model_auth = {"passes": True, "array_hashes": {"model": "b" * 64}}
    cross = {"passes": True, "q8": True}
    task_bundle = (task_auth, task_summary, task_arrays, {}, {})
    model_bundle = (model_auth, cross, {}, model_arrays, {}, {})
    npz_path = (tmp_path / "prerequisite.npz").resolve()
    npz_sha = "c" * 64
    summary = {
        "protocol": prerequisite.PROTOCOL, "incomplete": False,
        "overall_pass": True, "prerequisite_auth_only": True,
        "oracle_evidence": False, "benchmark_evidence": False,
        "task_authentication": task_auth, "model_authentication": model_auth,
        "cross_bind": cross, "geometry_certificate": semantic,
        "counters": prerequisite._expected_counters(4),
        "provenance": _provenance(final=True), "array_names": sorted(arrays),
        "array_hashes": {name: prerequisite._array_hash(arrays[name]) for name in sorted(arrays)},
        "npz_path": str(npz_path), "npz_sha256": npz_sha, "checkpoint_count": 5,
    }
    assert prerequisite.certify_final(
        summary, arrays, task_bundle, model_bundle, semantic, npz_path, npz_sha
    )["passes"]
    for field, value in (("npz_path", str(tmp_path / "lie.npz")),
                         ("npz_sha256", "d" * 64)):
        bad = copy.deepcopy(summary); bad[field] = value
        assert not prerequisite.certify_final(
            bad, arrays, task_bundle, model_bundle, semantic, npz_path, npz_sha
        )["passes"]
    bad = copy.deepcopy(summary); bad["geometry_certificate"]["fresh_lie"] = True
    assert not prerequisite.certify_final(
        bad, arrays, task_bundle, model_bundle, semantic, npz_path, npz_sha
    )["passes"]


def test_retained_disk_recert_exactly_binds_fresh_checkpoint_detail_and_documents(tmp_path, monkeypatch):
    task_summary, task_arrays, model_arrays, kinematics = _synthetic_bundle()
    arrays, _details, _calls = prerequisite.construct_geometry_arrays(
        task_summary, task_arrays, model_arrays, kinematics
    )
    semantic = prerequisite.certify_geometry_arrays(
        arrays, task_summary, task_arrays, model_arrays, kinematics
    )
    task_auth = {"passes": True, "array_hashes": {"task": "a" * 64}}
    model_auth = {"passes": True, "array_hashes": {"model": "b" * 64}}
    cross = {"passes": True, "q8": True}
    task_bundle = (task_auth, task_summary, task_arrays, {}, {})
    model_bundle = (model_auth, cross, {}, model_arrays, {}, {})
    monkeypatch.setattr(prerequisite, "authenticate_task_artifact", lambda: task_bundle)
    monkeypatch.setattr(
        prerequisite, "authenticate_model_artifact", lambda *_args: model_bundle
    )
    monkeypatch.setattr(prerequisite, "production_kinematics", lambda: kinematics)
    output = (tmp_path / "prerequisite.json").resolve()
    owned = {"task": task_auth, "model": model_auth, "cross_bind": cross,
             "geometry": semantic}
    checkpoint_pairs = []
    for generation, stage in enumerate((
        "gen0", "task_authenticated", "model_cross_bound",
        "geometry_authenticated", "honest_end",
    )):
        retained = {} if generation < 3 else arrays
        doc = prerequisite.checkpoint_document(
            generation, stage, prerequisite._expected_counters(generation),
            _provenance(final=generation == 4), retained,
            task_auth=None if generation == 0 else task_auth,
            model_auth=None if generation < 2 else model_auth,
            cross_bind=None if generation < 2 else cross,
            geometry_certificate=None if generation < 3 else semantic,
        )
        checkpoint_pairs.append(prerequisite.publish_checkpoint(
            output, doc, retained, owned_detail=owned
        ))
    npz_path = output.with_suffix(".npz")
    prerequisite._write_npz_exclusive(npz_path, arrays)
    summary = {
        "protocol": prerequisite.PROTOCOL, "incomplete": False,
        "overall_pass": True, "prerequisite_auth_only": True,
        "oracle_evidence": False, "benchmark_evidence": False,
        "task_authentication": task_auth, "model_authentication": model_auth,
        "cross_bind": cross, "geometry_certificate": semantic,
        "counters": prerequisite._expected_counters(4),
        "provenance": _provenance(final=True), "array_names": sorted(arrays),
        "array_hashes": {name: prerequisite._array_hash(arrays[name]) for name in sorted(arrays)},
        "npz_path": str(npz_path), "npz_sha256": prerequisite._sha256_file(npz_path),
        "checkpoint_count": 5,
    }
    prerequisite._write_json_exclusive(output, summary)
    sides = [{"path": str(path), "sha256": prerequisite._sha256_file(path)}
             for pair in checkpoint_pairs for path in pair]
    manifest_path = output.with_name(output.stem + ".manifest.json")
    manifest = {"protocol": prerequisite.PROTOCOL, "json_path": str(output),
                "json_sha256": prerequisite._sha256_file(output),
                "npz_path": str(npz_path), "npz_sha256": prerequisite._sha256_file(npz_path),
                "side_artifacts": sides}
    prerequisite._write_json_exclusive(manifest_path, manifest)
    pointer_path = output.with_name(output.stem + ".partial.latest.json")
    pointer = {"protocol": prerequisite.PROTOCOL, "incomplete": False,
               "json_path": str(output), "json_sha256": prerequisite._sha256_file(output),
               "npz_path": str(npz_path), "npz_sha256": prerequisite._sha256_file(npz_path),
               "manifest_path": str(manifest_path),
               "manifest_sha256": prerequisite._sha256_file(manifest_path)}
    pointer_path.unlink()
    prerequisite._write_json_exclusive(pointer_path, pointer)
    assert prerequisite.recertify_retained_prerequisite(output)["passes"]

    gen3 = output.with_name(output.stem + ".partial.gen3.json")
    row = json.loads(gen3.read_text()); row["task_authentication"]["fresh_lie"] = True
    gen3.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    for item in manifest["side_artifacts"]:
        if item["path"] == str(gen3): item["sha256"] = prerequisite._sha256_file(gen3)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    pointer["manifest_sha256"] = prerequisite._sha256_file(manifest_path)
    pointer_path.write_text(json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n")
    assert not prerequisite.recertify_retained_prerequisite(output)["passes"]


def test_prerequisite_source_has_no_task_rng_n96_import_optimizer_or_worker_call():
    source = inspect.getsource(prerequisite)
    assert "default_rng(" not in source and "generate_task(" not in source
    assert "import_module(" not in source and "BSQP_" not in source
    assert "minimize(" not in source and "subprocess.run([sys.executable" not in source
    assert prerequisite._expected_counters(4)["optimizer_calls"] == 0
    assert prerequisite._expected_counters(4)["cuda_calls"] == 0
