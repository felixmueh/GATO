"""One-way, per-goal parameter switch for the existing reaching MPC."""
import numpy as np

from gato_tiago.config import (
    TIAGO_REACHING_FINAL_APPROACH_PARAMS,
    TIAGO_REACHING_FINAL_APPROACH_TRIGGER,
)

COST_KEYS = ("q_cost", "qd_cost", "u_cost", "N_cost", "ee_orient_cost",
             "ee_orient_N_cost", "q_lim_cost", "vel_lim_cost", "ctrl_lim_cost")
PARAMETER_KEYS = (*COST_KEYS, "max_sqp_iters", "mu", "pcg_tol")


class FinalApproachSwitch:
    """Change parameters in place at the first eligible feedback observation.

    ``params`` uses the same cost/SQP/mu/PCG names as ``solver_params``.
    Unspecified values inherit the approach; overrides merge into the measured
    final-approach preset. ``trigger`` uses metres and maximum absolute joint
    speed in rad/s. Reset before each new goal (and each new run).
    """

    def __init__(self, approach, *, enabled=False, params=None, trigger=None):
        if not isinstance(enabled, bool):
            raise ValueError("final_approach must be a boolean")
        unknown = set(params or {}) - set(PARAMETER_KEYS)
        if unknown:
            raise ValueError(f"Unsupported final-approach parameters: {sorted(unknown)}")
        unknown = set(trigger or {}) - set(TIAGO_REACHING_FINAL_APPROACH_TRIGGER)
        if unknown:
            raise ValueError(f"Unsupported final-approach trigger fields: {sorted(unknown)}")
        self.enabled = enabled
        self.approach = {key: approach[key] for key in PARAMETER_KEYS}
        self.after = {**self.approach, **TIAGO_REACHING_FINAL_APPROACH_PARAMS,
                      **(params or {})}
        self.trigger = {**TIAGO_REACHING_FINAL_APPROACH_TRIGGER, **(trigger or {})}
        for configuration in (self.approach, self.after, self.trigger):
            for key, value in configuration.items():
                if key == "max_sqp_iters":
                    if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                            or not 0 < value <= np.iinfo(np.uint32).max):
                        raise ValueError("max_sqp_iters must be a positive uint32 integer")
                elif (not np.isscalar(value) or isinstance(value, (bool, str))
                      or not np.isfinite(value) or value < 0
                      or value > np.finfo(np.float32).max):
                    raise ValueError(f"{key} must be finite and nonnegative")
        self.active = False

    @staticmethod
    def check_solver(solver):
        """Fail before control execution if the extension needs rebuilding."""
        required = ("set_cost_weights", "set_max_sqp_iters", "set_mu_batch", "set_pcg_tol_batch")
        if any(not callable(getattr(solver.solver, key, None)) for key in required):
            raise RuntimeError("Final approach needs native parameter setters; rebuild the BSQP extension")

    @staticmethod
    def _apply(solver, params):
        FinalApproachSwitch.check_solver(solver)
        solver.set_max_sqp_iters(params["max_sqp_iters"])
        solver.solver.set_mu_batch(np.full(solver.batch_size, params["mu"], dtype=np.float32))
        solver.solver.set_pcg_tol_batch(np.full(solver.batch_size, params["pcg_tol"], dtype=np.float32))
        solver.set_cost_weights(**{key: params[key] for key in COST_KEYS})

    def update(self, solver, *, time_s, goal_index, distance_m, joint_velocities):
        if not self.enabled or self.active:
            return None
        speed = float(np.max(np.abs(joint_velocities)))
        if (not np.isfinite(distance_m)
                or not np.isfinite(speed) or distance_m < 0
                or distance_m > self.trigger["distance_m"]
                or speed > self.trigger["max_joint_speed_rad_s"]):
            return None
        self._apply(solver, self.after)
        self.active = True
        return dict(time_s=float(time_s), goal_index=int(goal_index),
                    distance_m=float(distance_m), max_joint_speed_rad_s=speed,
                    before=dict(self.approach), after=dict(self.after))

    def reset(self, solver):
        """Restore approach parameters only if this goal actually switched."""
        if self.active:
            self._apply(solver, self.approach)
        self.active = False
