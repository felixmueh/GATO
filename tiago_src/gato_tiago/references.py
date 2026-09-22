"""Horizontal tool-position reference shared by TIAGo experiments."""
import numpy as np
from bsqp.config import FIG8_DEFAULT_PARAMS

TIAGO_HORIZONTAL_FIG8_PARAMS = {
    "x_span": 0.32,
    "y_amplitude": 0.08,
    "edge_clearance": 0.02,
    "period": 24.0,
    "cycles": FIG8_DEFAULT_PARAMS["cycles"],
}


def tiago_horizontal_figure8(dt, start_ee, *, cycles=None):
    params = TIAGO_HORIZONTAL_FIG8_PARAMS
    cycles = params["cycles"] if cycles is None else cycles
    steps_per_cycle = int(params["period"] / dt)
    phase = np.linspace(0.0, 2.0 * np.pi, steps_per_cycle, endpoint=False)
    left_edge = np.asarray(start_ee, dtype=np.float64).copy()
    left_edge[0] += params["edge_clearance"]

    cycle = np.zeros((steps_per_cycle, 6), dtype=np.float64)
    cycle[:, 0] = left_edge[0] + 0.5 * params["x_span"] * (1.0 - np.cos(phase))
    cycle[:, 1] = left_edge[1] + params["y_amplitude"] * np.sin(2.0 * phase)
    cycle[:, 2] = left_edge[2]
    reference = np.tile(cycle, (int(cycles), 1))
    return reference.reshape(-1), {
        "reference_mode": "tiago_horizontal",
        "start_edge_clearance_m": float(params["edge_clearance"]),
        "x_span_m": float(params["x_span"]),
        "y_amplitude_m": float(params["y_amplitude"]),
        "period_sec": float(params["period"]),
        "cycles": int(cycles),
        "left_edge": left_edge.tolist(),
    }
