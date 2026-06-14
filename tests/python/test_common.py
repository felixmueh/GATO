import numpy as np

from bsqp.common import (
    figure8,
    initialize_warm_start,
    sample_reference,
    sample_reference_horizon,
    shift_packed_trajectory_warm_start,
)


def test_warm_start_repeats_current_state_at_every_knot_with_zero_controls():
    nx = 4
    nu = 2
    knots = 3
    x_start = np.array([0.25, -0.5, 1.5, 2.0])

    warm = initialize_warm_start(x_start, N=knots, nx=nx, nu=nu)

    assert warm.shape == (knots * (nx + nu) - nu,)
    for knot in range(knots):
        knot_start = knot * (nx + nu)
        np.testing.assert_allclose(warm[knot_start:knot_start + nx], x_start)

        control_start = knot_start + nx
        control_stop = min(control_start + nu, warm.size)
        np.testing.assert_allclose(warm[control_start:control_stop], 0.0)


def test_figure8_returns_finite_pose_references_for_requested_cycles():
    dt = 0.1
    period = 2.0
    cycles = 3

    traj = figure8(dt=dt, period=period, cycles=cycles)

    expected_points_per_cycle = int(period / dt)
    expected_shape = (expected_points_per_cycle * cycles * 6,)
    assert traj.shape == expected_shape

    reshaped = traj.reshape(-1, 6)
    assert np.isfinite(reshaped).all()
    np.testing.assert_allclose(reshaped[:, 3:], 0.0)


def test_figure8_repeats_the_same_closed_reference_each_cycle():
    dt = 0.05
    period = 1.0
    cycles = 2

    traj = figure8(dt=dt, period=period, cycles=cycles)

    points_per_cycle = int(period / dt)
    cycle_a, cycle_b = traj.reshape(cycles, points_per_cycle, 6)
    np.testing.assert_allclose(cycle_a, cycle_b)


def test_reference_sampling_interpolates_continuous_time():
    reference = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 6.0, 0.0, 0.0, 0.0],
            [4.0, 8.0, 12.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    ).reshape(-1)

    sampled = sample_reference(reference, np.array([0.0, 0.5, 1.0]), dt=1.0)

    np.testing.assert_allclose(
        sampled[:, :3],
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
    )

    horizon = sample_reference_horizon(reference, start_time=0.5, dt=1.0, knots=2)
    np.testing.assert_allclose(
        horizon.reshape(-1, 6)[:, :3],
        [[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]],
    )


def test_shift_packed_trajectory_warm_start_reanchors_and_samples_future_knots():
    nx = 2
    nu = 1
    knots = 4
    dt = 1.0
    packed = np.array(
        [
            0.0, 0.0, 10.0,
            1.0, 2.0, 20.0,
            2.0, 4.0, 30.0,
            3.0, 6.0,
        ],
        dtype=np.float32,
    )

    shifted = shift_packed_trajectory_warm_start(
        packed,
        x_current=np.array([100.0, 200.0], dtype=np.float32),
        nx=nx,
        nu=nu,
        knots=knots,
        elapsed=0.5,
        dt=dt,
    )

    shifted_states = np.array(
        [
            shifted[0:2],
            shifted[3:5],
            shifted[6:8],
            shifted[9:11],
        ]
    )
    shifted_controls = np.array([shifted[2], shifted[5], shifted[8]])

    np.testing.assert_allclose(
        shifted_states,
        [
            [100.0, 200.0],
            [1.5, 3.0],
            [2.5, 5.0],
            [3.0, 6.0],
        ],
    )
    np.testing.assert_allclose(shifted_controls, [10.0, 20.0, 30.0])
