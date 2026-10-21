"""Validation shared by registration and per-solver passive-loss settings."""
import numpy as np


def smoothing_widths(value, nq):
    """Return independent float32-representable, positive finite joint widths."""
    if isinstance(value, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in value):
        raise ValueError("joint_friction_smoothing requires a numeric scalar or vector")
    try:
        raw = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("joint_friction_smoothing requires a numeric scalar or vector") from exc
    if raw.dtype.kind not in "iuf" or raw.ndim > 1:
        raise ValueError("joint_friction_smoothing requires a numeric scalar or vector")
    with np.errstate(over="ignore", under="ignore"):
        widths = np.asarray(raw, dtype=np.float32)
    if widths.ndim == 0:
        widths = np.full(nq, widths, dtype=np.float32)
    if widths.shape != (nq,) or not np.all(np.isfinite(widths) & (widths > 0)):
        raise ValueError("joint_friction_smoothing must give positive finite velocity widths for every joint (representable in float32)")
    return np.ascontiguousarray(widths).copy()
