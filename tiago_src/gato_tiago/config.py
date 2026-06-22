"""TIAGo-specific experiment configuration."""

import numpy as np


TIAGO_RIGHT_START_CONFIGS = {
    "zero": np.zeros(7),
    # Default start configuration from simulation environment.
    # This pose does not clear the default conservative safety margin.
    "comfortable": np.array([-0.36, -1.83, -0.47, -2.35, 0.0, -1.20, 0.04]),
    # Start Configuration with a little bit more clearance of the arm to the body.
    "comfortable_high_clearance": np.array(
        [-0.39, -1.73, -0.38, -2.35, 0.0, -1.21, 0.04]
    ),
}

TIAGO_RIGHT_START_CONFIGS["combortable_high_clearance"] = TIAGO_RIGHT_START_CONFIGS[
    "comfortable_high_clearance"
]

TIAGO_RIGHT_DEFAULT_START_CONFIG = "comfortable_high_clearance"

TIAGO_TRACKING_SOLVER_PARAMS = {
    "max_sqp_iters": 5,
    "kkt_tol": 1e-3,
    "max_pcg_iters": 120,
    "pcg_tol": 1e-3,
    "solve_ratio": 1.0,
    "mu": 1.0,
    "q_cost": 4.0,
    "qd_cost": 1e-2,
    "u_cost": 1e-4,
    "N_cost": 80.0,
    # Orientation is disabled by default.
    "ee_orient_cost": 0.0,
    "ee_orient_N_cost": 0.0,
    "q_lim_cost": 0.01,
    "vel_lim_cost": 0.001,
    "ctrl_lim_cost": 0.003,
    "rho": 0.01,
}
