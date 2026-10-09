"""Control horizon boundaries for TIAGo-sized packed trajectories; CPU only."""

import numpy as np
import pytest

from bsqp.common import sample_packed_trajectory_control


@pytest.mark.parametrize('elapsed,expected_knot', [
    (0., 0), (.015, 0), (.045, 1), (.915, 30), (.929, 30),
    (.930, 30), (.936, 30), (1.2, 30),
])
def test_control_sampling_holds_final_available_torque(elapsed, expected_knot):
    knots, nx, nu, dt = 32, 14, 7, .03
    packed = np.full(knots * (nx + nu) - nu, -10000.)
    controls = np.arange((knots - 1) * nu).reshape(knots - 1, nu)
    for knot, torque in enumerate(controls):
        start = knot * (nx + nu) + nx
        packed[start:start+nu] = torque
    actual = sample_packed_trajectory_control(packed, nx, nu, knots, elapsed, dt)
    assert actual.shape == (nu,)
    np.testing.assert_array_equal(actual, controls[expected_knot])


@pytest.mark.parametrize('knots,dt', [(1, .03), (32, 0.), (32, -.03)])
def test_control_sampling_rejects_missing_horizon(knots, dt):
    with pytest.raises(ValueError):
        sample_packed_trajectory_control(np.zeros(14), 14, 7, knots, 0., dt)
