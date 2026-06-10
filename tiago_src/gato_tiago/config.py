"""TIAGo-specific experiment configuration."""

import numpy as np


TIAGO_RIGHT_START_CONFIGS = {
    "zero": np.zeros(7),
    "comfortable": np.array([-0.36, -1.83, -0.47, -2.35, 0.0, -1.20, 0.04]),
}

TIAGO_TRACKING_SOLVER_PARAMS = {
    "max_sqp_iters": 5,
    "kkt_tol": 1e-3,
    "max_pcg_iters": 120,
    "pcg_tol": 1e-3,
    "solve_ratio": 1.0,
    "mu": 1.0,
    "q_cost": 4.0,
    "qd_cost": 1e-2,
    "u_cost": 1e-5,
    "N_cost": 80.0,
    "q_lim_cost": 0.01,
    "vel_lim_cost": 0.001,
    "ctrl_lim_cost": 0.003,
    "rho": 0.01,
}
