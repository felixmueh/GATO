"""CPU orchestration contracts; no dynamics, optimizer or ROS implementation."""
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np


def frozen(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class WrenchConvention:
    # Reference point is required: a frame label alone cannot specify a wrench.
    hypothesis_frame: str
    hypothesis_point: str
    backend_frame: str
    backend_point: str
    components: str = "Fx,Fy,Fz,Tx,Ty,Tz"
    force_units: str = "N"
    torque_units: str = "N*m"


@dataclass(frozen=True)
class AppliedSegment:
    """Actually applied control on [start_s, end_s), in the plant clock."""
    start_s: float
    end_s: float
    control: np.ndarray


@dataclass(frozen=True)
class TickInputs:
    previous_time_s: float
    observed_time_s: float
    previous_state: np.ndarray
    observed_state: np.ndarray
    applied_history: tuple[AppliedSegment, ...]
    candidate_states: np.ndarray
    candidate_references: np.ndarray
    candidate_warm_seeds: np.ndarray


@dataclass(frozen=True)
class PredictionRequest:
    inputs: TickInputs
    hypothesis_wrenches: np.ndarray
    convention: WrenchConvention
    transform_wrench: Callable


@dataclass(frozen=True)
class TORequest:
    candidate_indices: tuple[int, ...]
    states: np.ndarray
    references: np.ndarray
    warm_seeds: np.ndarray
    hypothesis_wrenches: np.ndarray
    backend_wrenches: np.ndarray
    convention: WrenchConvention


def freeze_inputs(inputs):
    history = tuple(AppliedSegment(s.start_s, s.end_s, frozen(s.control))
                    for s in inputs.applied_history)
    result = TickInputs(inputs.previous_time_s, inputs.observed_time_s,
                        frozen(inputs.previous_state), frozen(inputs.observed_state),
                        history, frozen(inputs.candidate_states),
                        frozen(inputs.candidate_references),
                        frozen(inputs.candidate_warm_seeds))
    if not np.isfinite([result.previous_time_s, result.observed_time_s]).all():
        raise ValueError("history timestamps must be finite")
    if result.observed_time_s <= result.previous_time_s or not history:
        raise ValueError("prediction requires a positive recorded interval and controls")
    cursor = result.previous_time_s
    for segment in history:
        if (not np.isfinite([segment.start_s, segment.end_s]).all()
                or segment.start_s != cursor or segment.end_s <= segment.start_s):
            raise ValueError("applied history must be ordered, gap-free and nonoverlapping")
        cursor = segment.end_s
    if cursor != result.observed_time_s:
        raise ValueError("applied history must cover the entire observation interval")
    if result.previous_state.ndim != 1 or result.observed_state.shape != result.previous_state.shape:
        raise ValueError("previous and observed states must be matching vectors")
    return result


def run_tick(mode, inputs, *, estimator, predictor, optimizer, convention,
             transform_wrench, observer):
    """Run one isolated comparison tick and emit one raw record even on failure.

    predictor(request) -> (B,nx) predictions, replaying ALL applied segments.
    optimizer(request) -> (batch,...) plans; native timing may be saved by backend.
    estimator.generate_batch() -> (B,6); update_frozen(index,errors,batch) -> None.
    observer(record) persists raw arrays/nonfinite values; runs outside all timers.
    transform_wrench(state,wrench,convention) -> six backend components, including
    reference-point torque shift. Predictor calls it at every dynamics evaluation
    using that hypothesis's evolving state. TO backend handles pose-dependent
    wrench dynamics within its solve; conversion here only defines initial data.
    """
    record = {"mode": mode, "inputs_submitted": inputs, "convention": convention,
              "timings_s": {}, "memory_bytes": None,
              "memory_reason": "allocation/peak memory not measured by CPU orchestration",
              "status": "started", "events": [], "error": None}
    started = perf_counter()

    def measured(name, operation):
        begin = perf_counter()
        record["events"].append(name)
        try:
            return operation()
        finally:
            record["timings_s"][name] = perf_counter() - begin

    try:
        if mode not in ("original", "adapted"):
            raise ValueError("mode must be original or adapted")
        if (not all((convention.hypothesis_frame, convention.hypothesis_point,
                     convention.backend_frame, convention.backend_point))
                or convention.components != "Fx,Fy,Fz,Tx,Ty,Tz"
                or convention.force_units != "N" or convention.torque_units != "N*m"):
            raise ValueError("explicit frame/point and SI wrench convention required")
        tick = freeze_inputs(inputs)
        record["inputs"] = tick
        forces = frozen(estimator.generate_batch())
        record["hypotheses"] = forces
        if forces.ndim != 2 or forces.shape[1] != 6 or len(forces) == 0:
            raise ValueError("hypotheses must have shape (B,6)")
        batch = len(forces)
        if (tick.candidate_states.shape != (batch, len(tick.observed_state))
                or tick.candidate_references.ndim < 2
                or tick.candidate_warm_seeds.ndim < 2
                or len(tick.candidate_references) != batch
                or len(tick.candidate_warm_seeds) != batch):
            raise ValueError("state/reference/warm seed must have the same B candidate axis")

        def solve(indices):
            ids = list(indices)
            backend = frozen([transform_wrench(tick.candidate_states[i], forces[i], convention)
                              for i in ids])
            if backend.shape != (len(ids), 6):
                raise ValueError("wrench transform must return six components")
            request = TORequest(tuple(ids), frozen(tick.candidate_states[ids]),
                                frozen(tick.candidate_references[ids]),
                                frozen(tick.candidate_warm_seeds[ids]),
                                frozen(forces[ids]), backend, convention)
            record["to_request"] = request
            plans = frozen(measured("to_backend", lambda: optimizer(request)))
            record["plans"] = plans
            if plans.ndim < 2 or len(plans) != len(ids):
                raise ValueError("optimizer plans must retain the requested candidate axis")
            return plans

        if mode == "original":
            plans = solve(range(batch))
        prediction_request = PredictionRequest(tick, forces, convention, transform_wrench)
        record["prediction_request"] = prediction_request
        predictions = frozen(measured("candidate_forward", lambda: predictor(prediction_request)))
        record["predictions"] = predictions
        if predictions.shape != (batch, len(tick.observed_state)):
            raise ValueError("predictor must return shape (B,nx)")

        def select():
            errors = frozen(np.linalg.norm(predictions - tick.observed_state[None, :], axis=1))
            record["errors"] = errors
            if not np.isfinite(predictions).all() or not np.isfinite(errors).all():
                raise ValueError("nonfinite prediction/error; selection rejected")
            index = int(np.argmin(errors))  # unchanged NumPy policy, earliest finite tie
            record.update(errors=errors, selected_index=index, selected_force=frozen(forces[index]))
            return index

        selected = measured("selection", select)
        measured("estimator_update", lambda: estimator.update_frozen(selected, record["errors"], forces))
        if mode == "adapted":
            plans = solve((selected,))
            record["selected_plan"] = frozen(plans[0])
        else:
            record["selected_plan"] = frozen(plans[selected])
        if not np.isfinite(plans).all():
            raise ValueError("nonfinite optimizer plan retained")
        record["status"] = "complete"
        return record
    except Exception as error:
        record["status"] = "failed"
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        elapsed = perf_counter() - started
        record["timings_s"]["total"] = elapsed
        record["timings_s"]["wrapper"] = elapsed - sum(
            record["timings_s"].get(k, 0.0) for k in ("candidate_forward", "to_backend", "selection"))
        observer(record)
