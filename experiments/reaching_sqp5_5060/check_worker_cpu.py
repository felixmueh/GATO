#!/usr/bin/env python3
"""Bounded fake-solver check of cold start, t0 arrival, RK4 and compact work."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


w = load("worker_contract", HERE / "run_case.py")
adapter = load("adapter_contract", HERE / "adapter.py")
model = pin.buildModelFromUrdf(str(SOURCE / w.MODEL))
data = model.createData()
initial = np.r_[[-.5, -1.4, -.3, -1.7, 0., -.9, 0.], np.zeros(7)]
witness = initial[:7].copy()
witness[0] += .3


def packed(state):
    return np.r_[np.tile(np.r_[state, np.zeros(7)], 31), state]


results = []
for arm, (mode, _) in w.ARMS.items():
    for already_arrived in (False, True):
        events, steps = [], []
        goal = w.pose(pin, model, data, initial[:7] if already_arrived else witness)
        reference = np.tile(np.r_[goal.translation, pin.rpy.matrixToRpy(goal.rotation)], 32).astype(np.float32)[None]
        w.task = lambda *args: (initial.copy(), witness.copy(), goal, reference,
                                dict(source="CPU fake-solver contract", already_inside_arrival_region=already_arrived))

        class Native:
            def set_tiago_joint_cost_mode(self, requested, q):
                events.append("set")
                self.mode, self.q = requested, q

            def get_tiago_joint_cost_mode(self):
                events.append("get")
                return dict(mode=self.mode, q_reference=self.q, joint_margin_rad=.20)

        class Solver:
            def set_f_ext_B(self, value):
                events.append("force")

            def reset_dual(self):
                events.append("dual")

            def reset_rho(self):
                events.append("rho")

            def solve(self, actual, ref, seed):
                events.append("solve")
                assert np.array_equal(seed[0], packed(initial).astype(np.float32))
                assert np.array_equal(actual[0], initial)
                return seed.copy(), 1.

            def get_stats(self):
                return dict(ls_num_iters=1, pcg_iters=np.array([0]), sqp_iters=np.array([1]),
                            step_size=np.array([0.]), min_merit=np.array([1.]), kkt_converged=np.array([0]),
                            initial_merit=np.array([1.]), final_merit=np.array([1.]))

        class MPC:
            def __init__(self, *args, **kwargs):
                events.append("constructor")
                self.solver, self.data, self.actual_f_ext = Solver(), data, None

        def fake_rk4(model, data, q, v, u, h, force):
            steps.append(h)
            return witness.copy(), np.zeros(7)

        common = SimpleNamespace(initialize_warm_start=lambda *args: packed(initial), rk4=fake_rk4)
        protocol = json.loads((HERE / "inputs" / (arm + ".json")).read_text())
        with tempfile.TemporaryDirectory(prefix="gato-sqp5-contract-") as temp:
            out = Path(temp)
            args = SimpleNamespace(source=SOURCE, output=out, preflight_input=None, family="cone")
            with contextlib.redirect_stdout(io.StringIO()):
                result = w.run_episode((np, pin, MPC, common, Native()), args, protocol, 0)
            assert result["task_success"] and result["physical_pass"], result
            assert result["feedback_solves"] == (0 if already_arrived else 1)
            assert abs(result["reached_seconds"] - (0. if already_arrived else .03)) < 1e-12
            assert events[:3] == ["set", "get", "constructor"]
            assert events.count("dual") == events.count("rho") == 1
            assert np.allclose(steps, [] if already_arrived else [.003] * 10, rtol=0, atol=1e-12)
            assert not (out / "pair_00000/horizons.npz").exists()
            with np.load(out / "pair_00000/work_stats.npz") as stats:
                assert (not stats.files) if already_arrived else stats["pcg_iters"].tolist() == [0]
            with np.load(out / "pair_00000/executed.npz") as saved:
                assert len(saved["times"]) == (1 if already_arrived else 11)
                assert saved["controls"].shape == ((0 if already_arrived else 10), 7)
        results.append(dict(arm=arm, already_arrived=already_arrived, feedback_solves=result["feedback_solves"]))

negative = dict(complete=True, execution_error=None, interrupted=False,
                physical=dict(finite=True, physical_pass=False), task_success=False)
assert adapter.benchmark_readiness(negative)
assert not adapter.benchmark_readiness(dict(negative, execution_error="setup failed"))
assert not adapter.benchmark_readiness(dict(negative, physical=dict(finite=False)))
print(json.dumps(dict(status="PASS", cases=results, canonical_negative_outcomes_retained=True,
                      native_solver_imports=0, GPU_calls=0), indent=2))
