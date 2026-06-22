import importlib.util
import os

import numpy as np
import pytest


PLANT = "tiago_right"
KNOTS = 16
URDF_PATH = "gato/dynamics/tiago_right/tiago_right_arm.urdf"


def _require_cuda_tiago_terminal_test_enabled():
    if os.environ.get("GATO_RUN_TRACKING_TESTS") != "1":
        pytest.skip("set GATO_RUN_TRACKING_TESTS=1 to run CUDA tracking tests")

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available to torch in this environment")

    module_name = f"bsqp.bsqpN{KNOTS}_{PLANT}"
    if importlib.util.find_spec(module_name) is None:
        pytest.skip(f"built solver extension is not available: {module_name}")


def test_tiago_terminal_tracking_responds_to_n_cost():
    _require_cuda_tiago_terminal_test_enabled()

    import pinocchio as pin

    from bsqp.config import TIAGO_RIGHT_START_CONFIGS, TIAGO_TRACKING_SOLVER_PARAMS
    from bsqp.interface import BSQP

    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    torso_id = model.getFrameId("torso_lift_link")
    tool_id = model.getFrameId("arm_right_tool_link")

    def ee_pos(q):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return (data.oMf[torso_id].inverse() * data.oMf[tool_id]).translation.copy()

    q0 = TIAGO_RIGHT_START_CONFIGS["comfortable"].astype(np.float64)
    x0 = np.hstack([q0, np.zeros(model.nv)]).astype(np.float32)
    goal = ee_pos(q0) + np.array([0.11, -0.085, 0.035], dtype=np.float64)
    reference = np.tile(
        np.hstack([goal, np.zeros(3)]).astype(np.float32),
        KNOTS,
    ).reshape(1, -1)

    nx = model.nq + model.nv
    nu = model.nv

    def terminal_distance(n_cost):
        solver_params = dict(TIAGO_TRACKING_SOLVER_PARAMS)
        solver_params.update(
            max_sqp_iters=20,
            max_pcg_iters=160,
            pcg_tol=1e-4,
            N_cost=n_cost,
            vel_lim_cost=0.0,
            ctrl_lim_cost=0.0,
        )
        solver = BSQP(
            model_path=URDF_PATH,
            batch_size=1,
            N=KNOTS,
            dt=0.03,
            plant_type=PLANT,
            **solver_params,
        )
        warm_start = np.zeros((1, KNOTS * (nx + nu) - nu), dtype=np.float32)
        for knot in range(KNOTS):
            start = knot * (nx + nu)
            warm_start[0, start : start + nx] = x0
        solver.reset_dual()
        trajectory, _ = solver.solve(x0.reshape(1, -1), reference, warm_start)
        terminal_start = (KNOTS - 1) * (nx + nu)
        terminal_q = trajectory[0, terminal_start : terminal_start + model.nq]
        return float(np.linalg.norm(ee_pos(terminal_q.astype(np.float64)) - goal))

    low_terminal_distance = terminal_distance(0.0)
    high_terminal_distance = terminal_distance(300.0)

    assert high_terminal_distance < 0.5 * low_terminal_distance
    assert high_terminal_distance < 0.01
