from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "tiago_src")]

from tiago_examples.iiwa_multimodal_endpoint_validity import (
    AGREEMENT_TOLERANCE_M,
    DE_SEED,
    DE_SETTINGS,
    LS_RANDOM_SEED,
    LS_RANDOM_START_COUNT,
    LS_SETTINGS,
    TASK_SEED,
    benchmark_position,
    classify_endpoint,
)
from gato_tiago.iiwa_multimodal_sqp import generate_instance, load_model


def test_endpoint_audit_protocol_is_exact_and_cpu_oracle_only():
    assert TASK_SEED == 170
    assert LS_RANDOM_SEED == 20260812
    assert LS_RANDOM_START_COUNT == 100
    assert LS_SETTINGS == {
        "jac": "3-point",
        "max_nfev": 2000,
        "xtol": 1e-12,
        "ftol": 1e-12,
        "gtol": 1e-12,
    }
    assert DE_SEED == 20260813
    assert DE_SETTINGS == {
        "workers": 1,
        "updating": "immediate",
        "polish": False,
        "popsize": 20,
        "maxiter": 2000,
        "tol": 1e-10,
        "atol": 1e-12,
        "mutation": (0.5, 1.0),
        "recombination": 0.7,
    }
    assert AGREEMENT_TOLERANCE_M == 1e-5
    source = (
        REPO_ROOT / "tiago_examples/iiwa_multimodal_endpoint_validity.py"
    ).read_text()
    assert '"oracle_only": True' in source
    assert '"benchmark_seed_eligible": False' in source
    assert '"gato_or_cuda_calls": 0' in source
    assert '"route_planner_calls": 0' in source
    assert '"dynamics_solve_calls": 0' in source
    assert "differential_evolution" in source
    assert "BSQP" not in source


def test_validity_fk_is_exact_benchmark_joint_origin_not_contact_frame():
    model = load_model()
    instance = generate_instance(TASK_SEED, model=model)
    q0 = instance["q0"].astype(np.float64)
    benchmark = benchmark_position(model, q0)
    # The task generator forms start/goal before its final float32 q0 cast.
    np.testing.assert_allclose(
        benchmark, instance["start_position_m"], rtol=0.0, atol=5e-9
    )
    data = model.createData()
    import pinocchio as pin

    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    contact = data.oMf[model.getFrameId("contact")].translation
    assert np.isclose(np.linalg.norm(contact - benchmark), 0.04, atol=1e-12)


def test_endpoint_classification_fails_closed_at_frozen_thresholds():
    invalid = classify_endpoint(0.03396472, 0.03396473)
    assert invalid["decision"] == "empirically_unreachable_invalid_frozen_endpoint"
    assert invalid["abort_route_and_dynamics_search"]
    reachable = classify_endpoint(0.019, 0.019001)
    assert reachable["decision"] == "endpoint_reachable_route_search_may_resume"
    assert not reachable["abort_route_and_dynamics_search"]
    for left, right in ((0.021, 0.021), (0.034, 0.03402)):
        ambiguous = classify_endpoint(left, right)
        assert ambiguous["decision"] == "requires_verifier_ruling"
        assert ambiguous["abort_route_and_dynamics_search"]
