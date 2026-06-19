"""Collision-safety helpers for live TIAGo controller experiments.

The module is intentionally ROS-free. ROS-facing tools pass named joint-state
snapshots in, and this module owns the Pinocchio collision model, state mapping,
distance checks, blacklist handling, and URDF joint-limit checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import pinocchio as pin

from gato_tiago.ros_tiago import RIGHT_ARM_JOINTS


REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_URDF_PATH = REPO_ROOT / "TiagoProURDF" / "tiago_pro.urdf"
_WORKSPACE_URDF_PATH = Path("/workspace/GATO/TiagoProURDF/tiago_pro.urdf")
DEFAULT_URDF_PATH = (
    _LOCAL_URDF_PATH if _LOCAL_URDF_PATH.is_file() else _WORKSPACE_URDF_PATH
)
# Simplified collision model used by the live safety monitor.
DEFAULT_COLLISION_URDF_PATH = (
    REPO_ROOT
    / "tiago_configs"
    / "collision_alpha_wrap_a3_o25_cap001"
    / "tiago_pro_collision_alpha_wrap_a3_o25_cap001.urdf"
)
# Reviewed ignore list for known-safe/self-overlapping pairs.
DEFAULT_COLLISION_BLACKLIST_PATH = (
    REPO_ROOT / "tiago_configs" / "comfortable_pose_collision_blacklist.json"
)
# These joints are fixed at their reference value in the runtime reduced
# collision model, so safety does not require position/velocity state for them.
TIAGO_RUNTIME_STATE_FIXED_JOINT_NAMES = (
    "wheel_front_left_joint",
    "wheel_front_right_joint",
    "wheel_rear_left_joint",
    "wheel_rear_right_joint",
    "gripper_left_finger_joint",
    "gripper_left_inner_finger_left_joint",
    "gripper_left_fingertip_left_joint",
    "gripper_left_inner_finger_right_joint",
    "gripper_left_fingertip_right_joint",
    "gripper_left_outer_finger_right_joint",
    "gripper_right_finger_joint",
    "gripper_right_inner_finger_left_joint",
    "gripper_right_fingertip_left_joint",
    "gripper_right_inner_finger_right_joint",
    "gripper_right_fingertip_right_joint",
    "gripper_right_outer_finger_right_joint",
)


@dataclass(frozen=True)
class NamedJointState:
    """A ROS-independent named joint-state snapshot."""

    position: dict[str, float]
    velocity: dict[str, float]
    stamp_sec: float | None = None

@dataclass(frozen=True)
class CollisionSafetySettings:
    enabled: bool = True
    urdf_path: Path = DEFAULT_COLLISION_URDF_PATH
    blacklist_path: Path | None = DEFAULT_COLLISION_BLACKLIST_PATH
    controlled_joint_names: tuple[str, ...] = ()
    min_distance_m: float = 0.04
    check_timeout_sec: float = 0.05
    max_monitored_geometry_speed_m_s: float = 1.0
    joint_position_margin_rad: float = 0.0
    joint_velocity_scale: float = 1.0


@dataclass(frozen=True)
class SafetyCheckState:
    q: np.ndarray
    qd: np.ndarray
    stamp_sec: float
    received_monotonic_sec: float
    seq: int
    joint_positions: dict[str, float]
    joint_velocities: dict[str, float]


@dataclass(frozen=True)
class _SafetyCheckResult:
    seq: int
    ok: bool
    error: str | None = None
    fault_kind: str = ""
    elapsed_sec: float = 0.0
    kind: str = "check"


class _SafetyFault(RuntimeError):
    def __init__(self, fault_kind: str, message: str) -> None:
        super().__init__(message)
        self.fault_kind = fault_kind


@dataclass(frozen=True)
class CollisionBodySpeedReport:
    """Conservative point-speed bound for one collision geometry object."""

    geometry: str
    link: str
    parent_joint: str
    linear_speed_m_s: float
    angular_speed_rad_s: float
    radius_m: float
    speed_bound_m_s: float


@dataclass(frozen=True)
class CollisionMarginViolation:
    """Thresholded collision result for the runtime hot path."""

    index: int
    geometry_a: str
    geometry_b: str
    link_a: str
    link_b: str
    margin_m: float
    distance_lower_bound_m: float

    @property
    def approximate_distance_m(self) -> float:
        return self.distance_lower_bound_m + self.margin_m


@dataclass(frozen=True)
class JointLimitViolation:
    """One joint position or velocity limit violation."""

    joint: str
    kind: str
    value: float
    lower: float | None = None
    upper: float | None = None
    margin: float = 0.0
    limit: float | None = None
    scale: float = 1.0


class _SafetyChecker:
    def __init__(
        self,
        *,
        settings: CollisionSafetySettings,
        initial_state: SafetyCheckState,
    ) -> None:
        self.settings = settings

        blacklist_path = settings.blacklist_path
        if blacklist_path is not None and not Path(blacklist_path).is_file():
            raise FileNotFoundError(f"collision blacklist not found: {blacklist_path}")
        if not settings.controlled_joint_names:
            raise ValueError("collision safety requires controlled_joint_names")

        self.collision_model = build_tiago_collision_model(
            urdf_path=settings.urdf_path,
            state_fixed_joint_names=TIAGO_RUNTIME_STATE_FIXED_JOINT_NAMES,
            reference_positions=initial_state.joint_positions,
            blacklist_path=blacklist_path,
            controlled_joint_names=settings.controlled_joint_names,
        )
        self._last_checked_seq = -1

    def check(self, state: SafetyCheckState) -> None:
        if int(state.seq) == self._last_checked_seq:
            return
        joint_state = NamedJointState(
            position=state.joint_positions,
            velocity=state.joint_velocities,
            stamp_sec=float(state.stamp_sec),
        )
        q, qd = state_to_qv(self.collision_model.model, joint_state)
        joint_violations = check_joint_limits(
            self.collision_model.model,
            joint_state,
            position_margin=self.settings.joint_position_margin_rad,
            velocity_scale=self.settings.joint_velocity_scale,
            joint_names=self.settings.controlled_joint_names,
        )
        if joint_violations:
            violation = joint_violations[0]
            raise _SafetyFault(
                "joint_limit",
                "joint safety fault: "
                f"{violation.joint} {violation.kind} value={violation.value:.6f}",
            )

        margin_violation = compute_collision_margin_violation(
            self.collision_model,
            q,
            margin_m=self.settings.min_distance_m,
            stop_at_first=True,
        )
        if margin_violation is not None:
            raise _SafetyFault(
                "collision_margin",
                "collision safety fault: "
                f"margin violation below "
                f"min_distance_m={self.settings.min_distance_m:.6f} for "
                f"{margin_violation.geometry_a} <-> {margin_violation.geometry_b}",
            )

        speeds = compute_collision_body_speeds(self.collision_model, q, qd)
        if not speeds:
            raise _SafetyFault(
                "state_invalid",
                "collision safety fault: no monitored geometry for speed check",
            )
        fastest = max(speeds, key=lambda report: report.speed_bound_m_s)
        if fastest.speed_bound_m_s > self.settings.max_monitored_geometry_speed_m_s:
            raise _SafetyFault(
                "collision_body_speed",
                "collision safety fault: "
                f"monitored geometry speed {fastest.speed_bound_m_s:.6f} m/s above "
                "max_monitored_geometry_speed_m_s="
                f"{self.settings.max_monitored_geometry_speed_m_s:.6f} for "
                f"{fastest.geometry}",
            )
        self._last_checked_seq = int(state.seq)


class AsyncSafetyMonitor:
    def __init__(
        self,
        *,
        settings: CollisionSafetySettings,
        initial_state: SafetyCheckState,
    ) -> None:
        self.timeout_sec = float(settings.check_timeout_sec)
        self._ctx = mp.get_context("spawn")
        self._request_q: mp.Queue = self._ctx.Queue(maxsize=1)
        self._response_q: mp.Queue = self._ctx.Queue(maxsize=8)
        self._stop_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_safety_worker_main,
            kwargs={
                "settings": settings,
                "initial_state": initial_state,
                "request_q": self._request_q,
                "response_q": self._response_q,
                "stop_event": self._stop_event,
            },
        )
        self._last_checked_seq = -1
        self._closed = False
        self._process.start()
        try:
            result = self._response_q.get(timeout=15.0)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError("collision safety worker did not initialize within 15.000s") from exc
        if result.kind != "ready" or not result.ok:
            self.close()
            raise RuntimeError(result.error or "collision safety worker failed to initialize")
        self._pending_seq: int | None = None
        self._pending_since_monotonic_sec: float | None = None
        self.safety_status = "unchecked"
        self.safety_fault = ""
        self.safety_message = ""

    @property
    def last_checked_seq(self) -> int:
        return int(self._last_checked_seq)

    def check(self, state: SafetyCheckState) -> None:
        seq = int(state.seq)
        self._raise_if_dead()
        now = time.monotonic()
        self._poll_responses()

        if seq != self._last_checked_seq and self._pending_seq is None:
            self._submit(state, now)

        self._enforce_pending_timeout(now, seq)

    def wait_until_checked(
        self,
        state: SafetyCheckState,
        timeout_sec: float | None = None,
    ) -> None:
        seq = int(state.seq)
        timeout = self.timeout_sec if timeout_sec is None else float(timeout_sec)
        deadline = time.monotonic() + timeout
        while self._last_checked_seq != seq:
            self.check(state)
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError(
                    "collision safety timeout: worker did not report initial "
                    f"state seq {seq} within {timeout:.3f}s"
                )
            time.sleep(min(0.001, remaining))

    def _submit(self, state: SafetyCheckState, now: float) -> None:
        seq = int(state.seq)
        try:
            self._request_q.put_nowait(state)
        except queue.Full as exc:
            self._set_fault(
                "fault",
                "worker_error",
                "collision safety worker request queue is unexpectedly full for "
                f"state seq {seq}",
            )
            raise RuntimeError(self.safety_message) from exc
        self._pending_seq = seq
        self._pending_since_monotonic_sec = now

    def _poll_responses(self) -> None:
        while True:
            try:
                result = self._response_q.get_nowait()
            except queue.Empty:
                return
            if result.kind != "check":
                continue
            result_seq = int(result.seq)
            if self._pending_seq == result_seq:
                self._pending_seq = None
                self._pending_since_monotonic_sec = None
            if not result.ok:
                self._set_fault(
                    "fault",
                    result.fault_kind or "worker_error",
                    result.error or f"collision safety worker failed for seq {result_seq}",
                )
                raise RuntimeError(self.safety_message)
            self._last_checked_seq = result_seq
            self.safety_status = "ok"
            self.safety_fault = ""
            self.safety_message = ""

    def _enforce_pending_timeout(self, now: float, seq: int) -> None:
        if self._pending_seq is None or self._pending_since_monotonic_sec is None:
            return
        if now - self._pending_since_monotonic_sec > self.timeout_sec:
            self._set_fault(
                "stale",
                "worker_stale",
                "collision safety timeout: worker did not report a check for "
                f"pending seq {self._pending_seq} within {self.timeout_sec:.3f}s "
                f"while monitoring state seq {seq}",
            )
            raise RuntimeError(self.safety_message)

    def close(self, timeout_sec: float = 1.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        try:
            self._request_q.put_nowait(None)
        except queue.Full:
            pass
        if self._process.is_alive():
            self._process.join(timeout=timeout_sec)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)

    def _raise_if_dead(self) -> None:
        if self._process.exitcode is not None:
            self._set_fault(
                "fault",
                "worker_error",
                f"collision safety worker exited: {self._process.exitcode}",
            )
            raise RuntimeError(self.safety_message)

    def _set_fault(self, status: str, fault: str, message: str) -> None:
        self.safety_status = status
        self.safety_fault = fault
        self.safety_message = message


def _safety_worker_main(
    *,
    settings: CollisionSafetySettings,
    initial_state: SafetyCheckState,
    request_q: mp.Queue,
    response_q: mp.Queue,
    stop_event: mp.Event,
) -> None:
    try:
        monitor = _SafetyChecker(settings=settings, initial_state=initial_state)
        response_q.put(_SafetyCheckResult(seq=-1, ok=True, kind="ready"))
    except BaseException as exc:
        response_q.put(
            _SafetyCheckResult(
                seq=-1,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                kind="ready",
            )
        )
        return

    while not stop_event.is_set():
        try:
            state = request_q.get(timeout=0.01)
        except queue.Empty:
            continue
        if state is None:
            return
        start = time.perf_counter()
        try:
            monitor.check(state)
        except _SafetyFault as exc:
            response_q.put(
                _SafetyCheckResult(
                    seq=int(state.seq),
                    ok=False,
                    error=str(exc),
                    fault_kind=exc.fault_kind,
                    elapsed_sec=time.perf_counter() - start,
                )
            )
            continue
        except (ValueError, FloatingPointError) as exc:
            response_q.put(
                _SafetyCheckResult(
                    seq=int(state.seq),
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    fault_kind="state_invalid",
                    elapsed_sec=time.perf_counter() - start,
                )
            )
            continue
        except BaseException as exc:
            response_q.put(
                _SafetyCheckResult(
                    seq=int(state.seq),
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    fault_kind="worker_error",
                    elapsed_sec=time.perf_counter() - start,
                )
            )
            continue
        response_q.put(
            _SafetyCheckResult(
                seq=int(state.seq),
                ok=True,
                elapsed_sec=time.perf_counter() - start,
            )
        )


@dataclass
class TiagoCollisionModel:
    """Reduced Pinocchio model plus collision geometry."""

    model: pin.Model
    geometry_model: pin.GeometryModel
    state_fixed_joint_names: tuple[str, ...]
    state_updated_joint_names: tuple[str, ...]
    # Distance from each collision geometry origin to its furthest local point.
    geometry_radii_m: np.ndarray
    controlled_joint_names: tuple[str, ...]
    monitored_geometry_parent_joint_names: tuple[str, ...]
    monitored_geometry_indices: tuple[int, ...]
    monitor_only_collision_pairs: bool = True

    def make_data(self) -> tuple[pin.Data, pin.GeometryData]:
        # GeometryData must be constructed after the final collision-pair list is
        # in place. Mutating geometry_model.collisionPairs afterwards can crash.
        return self.model.createData(), pin.GeometryData(self.geometry_model)


def build_tiago_collision_model(
    *,
    urdf_path: str | Path = DEFAULT_URDF_PATH,
    package_dirs: Sequence[str | Path] | None = None,
    state_fixed_joint_names: Iterable[str] = TIAGO_RUNTIME_STATE_FIXED_JOINT_NAMES,
    reference_positions: Mapping[str, float] | None = None,
    blacklist_path: str | Path | None = None,
    controlled_joint_names: Iterable[str] = RIGHT_ARM_JOINTS,
    monitor_only_collision_pairs: bool = True,
) -> TiagoCollisionModel:
    """Build the reduced full-body collision model used by runtime safety."""
    urdf = Path(urdf_path)
    package_dirs = tuple(Path(path) for path in (package_dirs or (urdf.parent,)))
    full_model = pin.buildModelFromUrdf(str(urdf))
    controlled_names = tuple(controlled_joint_names)
    missing_controlled = sorted(
        name for name in set(controlled_names) if full_model.getJointId(name) >= full_model.njoints
    )
    if missing_controlled:
        raise ValueError(
            "controlled joints not found in full URDF model: "
            f"{missing_controlled}"
        )
    full_geom = pin.buildGeomFromUrdf(
        full_model,
        str(urdf),
        pin.GeometryType.COLLISION,
        None,
        [str(path) for path in package_dirs],
    )

    reference_q = pin.neutral(full_model)
    if reference_positions:
        _write_named_positions(full_model, reference_q, reference_positions)

    state_fixed_names = tuple(state_fixed_joint_names)
    locked_ids = []
    missing_state_fixed = []
    for name in state_fixed_names:
        joint_id = full_model.getJointId(name)
        if joint_id >= full_model.njoints:
            missing_state_fixed.append(name)
        else:
            locked_ids.append(joint_id)
    if missing_state_fixed:
        raise ValueError(
            "state-fixed joints not found in URDF model: "
            f"{missing_state_fixed}"
        )

    model, geometry_model = pin.buildReducedModel(
        full_model,
        full_geom,
        locked_ids,
        reference_q,
    )
    geometry_model.addAllCollisionPairs()
    if blacklist_path is not None:
        apply_collision_blacklist(geometry_model, blacklist_path)

    state_updated = tuple(name for idx, name in enumerate(model.names) if idx != 0)
    monitored_parent_names = monitored_geometry_parent_joint_names(
        model,
        controlled_names,
    )
    monitored_indices = monitored_geometry_indices(
        model,
        geometry_model,
        monitored_parent_names,
    )
    if monitor_only_collision_pairs:
        _keep_collision_pairs_touching_geometry(geometry_model, monitored_indices)
    radii = np.asarray(
        [_geometry_local_radius(obj) for obj in geometry_model.geometryObjects],
        dtype=np.float64,
    )
    return TiagoCollisionModel(
        model=model,
        geometry_model=geometry_model,
        state_fixed_joint_names=state_fixed_names,
        state_updated_joint_names=state_updated,
        geometry_radii_m=radii,
        controlled_joint_names=controlled_names,
        monitored_geometry_parent_joint_names=monitored_parent_names,
        monitored_geometry_indices=monitored_indices,
        monitor_only_collision_pairs=bool(monitor_only_collision_pairs),
    )


def monitored_geometry_parent_joint_names(
    model: pin.Model,
    controlled_joint_names: Iterable[str],
) -> tuple[str, ...]:
    """Return model joints whose geometry is attached below controlled joints."""
    controlled = tuple(str(name) for name in controlled_joint_names)
    controlled_id_by_name = {name: model.getJointId(name) for name in controlled}
    missing = sorted(name for name, joint_id in controlled_id_by_name.items() if joint_id >= model.njoints)
    if missing:
        raise ValueError(f"controlled joints not found in URDF model: {missing}")
    controlled_ids = set(controlled_id_by_name.values())

    names = []
    for idx, name in enumerate(model.names):
        if idx == 0:
            continue
        if _joint_is_in_controlled_subtree(model, idx, controlled_ids):
            names.append(str(name))
    if not names:
        raise ValueError(f"no monitored geometry parent joints under {controlled}")
    return tuple(names)


def _joint_is_in_controlled_subtree(
    model: pin.Model,
    joint_id: int,
    controlled_joint_ids: set[int],
) -> bool:
    current = int(joint_id)
    while current > 0:
        if current in controlled_joint_ids:
            return True
        current = int(model.parents[current])
    return False


def monitored_geometry_indices(
    model: pin.Model,
    geometry_model: pin.GeometryModel,
    parent_joint_names: Iterable[str],
) -> tuple[int, ...]:
    parent_names = {str(name) for name in parent_joint_names}
    indices = tuple(
        idx
        for idx, obj in enumerate(geometry_model.geometryObjects)
        if _joint_name(model, obj.parentJoint) in parent_names
    )
    if not indices:
        raise ValueError(
            "no collision geometry attached to monitored parent joints: "
            f"{sorted(parent_names)}"
        )
    return indices


def _keep_collision_pairs_touching_geometry(
    geometry_model: pin.GeometryModel,
    geometry_indices: Iterable[int],
) -> None:
    monitored = {int(idx) for idx in geometry_indices}
    to_keep = [
        pair
        for pair in geometry_model.collisionPairs
        if int(pair.first) in monitored or int(pair.second) in monitored
    ]
    geometry_model.removeAllCollisionPairs()
    for pair in to_keep:
        geometry_model.addCollisionPair(pair)


def validate_state_completeness(model: pin.Model, state: NamedJointState) -> None:
    missing_position = []
    missing_velocity = []
    for idx, name in enumerate(model.names):
        if idx == 0:
            continue
        if name not in state.position:
            missing_position.append(name)
        if name not in state.velocity:
            missing_velocity.append(name)
    messages = []
    if missing_position:
        messages.append(f"missing positions for {missing_position}")
    if missing_velocity:
        messages.append(f"missing velocities for {missing_velocity}")
    if messages:
        raise ValueError("; ".join(messages))


def state_to_qv(model: pin.Model, state: NamedJointState) -> tuple[np.ndarray, np.ndarray]:
    """Map named joint positions/velocities into Pinocchio q/v arrays."""
    validate_state_completeness(model, state)
    q = pin.neutral(model)
    v = np.zeros(model.nv, dtype=np.float64)
    _write_named_positions(model, q, state.position)
    for idx, name in enumerate(model.names):
        if idx == 0:
            continue
        joint = model.joints[idx]
        velocity = float(state.velocity[name])
        if joint.nv != 1:
            raise ValueError(f"unsupported joint velocity dimension for {name}: {joint.nv}")
        v[joint.idx_v] = velocity
    return q, v


def check_joint_limits(
    model: pin.Model,
    state: NamedJointState,
    *,
    position_margin: float = 0.0,
    velocity_scale: float = 1.0,
    joint_names: Iterable[str] | None = None,
) -> list[JointLimitViolation]:
    """Return URDF position/velocity limit violations for selected joints."""
    if position_margin < 0.0:
        raise ValueError("position_margin must be non-negative")
    if velocity_scale <= 0.0:
        raise ValueError("velocity_scale must be positive")
    q, v = state_to_qv(model, state)
    checked_joints = None if joint_names is None else {str(name) for name in joint_names}
    if checked_joints is not None:
        missing = sorted(name for name in checked_joints if model.getJointId(name) >= model.njoints)
        if missing:
            raise ValueError(f"joint limit check names not found in URDF model: {missing}")
    violations = []
    for idx, name in enumerate(model.names):
        if idx == 0:
            continue
        if checked_joints is not None and name not in checked_joints:
            continue
        joint = model.joints[idx]
        if joint.nq != 1 or joint.nv != 1:
            raise ValueError(f"unsupported joint dimensions for limit check: {name}")
        q_idx = joint.idx_q
        v_idx = joint.idx_v
        lower = float(model.lowerPositionLimit[q_idx])
        upper = float(model.upperPositionLimit[q_idx])
        position = float(q[q_idx])
        if np.isfinite(lower) and position < lower + position_margin:
            violations.append(
                JointLimitViolation(
                    joint=name,
                    kind="position_lower",
                    value=position,
                    lower=lower,
                    upper=upper,
                    margin=float(position_margin),
                )
            )
        if np.isfinite(upper) and position > upper - position_margin:
            violations.append(
                JointLimitViolation(
                    joint=name,
                    kind="position_upper",
                    value=position,
                    lower=lower,
                    upper=upper,
                    margin=float(position_margin),
                )
            )
        velocity_limit = float(model.velocityLimit[v_idx])
        velocity = float(v[v_idx])
        if np.isfinite(velocity_limit) and velocity_limit > 0.0:
            scaled_limit = velocity_limit * velocity_scale
            if abs(velocity) > scaled_limit:
                violations.append(
                    JointLimitViolation(
                        joint=name,
                        kind="velocity",
                        value=velocity,
                        limit=velocity_limit,
                        scale=float(velocity_scale),
                    )
                )
    return violations


def compute_collision_margin_violation(
    collision_model: TiagoCollisionModel,
    q: np.ndarray,
    *,
    margin_m: float,
    stop_at_first: bool = True,
) -> CollisionMarginViolation | None:
    """Return one pair whose real clearance is at or below ``margin_m``.

    Coal implements safety-margin checks in the collision path via
    ``CollisionRequest.security_margin``. That avoids computing every exact
    pair distance on normal controller ticks.
    """
    violations = compute_collision_margin_violations(
        collision_model,
        q,
        margin_m=margin_m,
        stop_at_first=stop_at_first,
    )
    return violations[0] if violations else None


def compute_collision_margin_violations(
    collision_model: TiagoCollisionModel,
    q: np.ndarray,
    *,
    margin_m: float,
    stop_at_first: bool = False,
) -> list[CollisionMarginViolation]:
    """Return all reported pairs whose clearance is at or below ``margin_m``."""
    if margin_m < 0.0:
        raise ValueError("margin_m must be non-negative")

    data, geometry_data = collision_model.make_data()
    margin = float(margin_m)
    for request in geometry_data.collisionRequests:
        request.security_margin = margin
        request.break_distance = margin
        request.enable_contact = False

    pin.computeCollisions(
        collision_model.model,
        data,
        collision_model.geometry_model,
        geometry_data,
        np.asarray(q, dtype=np.float64),
        bool(stop_at_first),
    )
    violations = []
    for idx, result in enumerate(geometry_data.collisionResults):
        if result.isCollision():
            pair = collision_model.geometry_model.collisionPairs[idx]
            object_a = collision_model.geometry_model.geometryObjects[pair.first]
            object_b = collision_model.geometry_model.geometryObjects[pair.second]
            violations.append(
                CollisionMarginViolation(
                    index=idx,
                    geometry_a=object_a.name,
                    geometry_b=object_b.name,
                    link_a=_frame_name(collision_model.model, object_a.parentFrame),
                    link_b=_frame_name(collision_model.model, object_b.parentFrame),
                    margin_m=margin,
                    distance_lower_bound_m=float(result.distance_lower_bound),
                )
            )
    return violations


def compute_collision_body_speeds(
    collision_model: TiagoCollisionModel,
    q: np.ndarray,
    v: np.ndarray,
    *,
    geometry_indices: Iterable[int] | None = None,
) -> list[CollisionBodySpeedReport]:
    data = collision_model.model.createData()
    pin.forwardKinematics(
        collision_model.model,
        data,
        np.asarray(q, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
    )
    indices = (
        collision_model.monitored_geometry_indices
        if geometry_indices is None
        else tuple(int(idx) for idx in geometry_indices)
    )
    reports = []
    for idx in indices:
        obj = collision_model.geometry_model.geometryObjects[idx]
        # GeometryObject.placement is relative to parentJoint in this model.
        # This overload returns the collision geometry origin velocity.
        velocity = pin.getFrameVelocity(
            collision_model.model,
            data,
            obj.parentJoint,
            obj.placement,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        linear_speed = float(np.linalg.norm(velocity.linear))
        angular_speed = float(np.linalg.norm(velocity.angular))
        radius = float(collision_model.geometry_radii_m[idx])
        reports.append(
            CollisionBodySpeedReport(
                geometry=obj.name,
                link=_frame_name(collision_model.model, obj.parentFrame),
                parent_joint=_joint_name(collision_model.model, obj.parentJoint),
                linear_speed_m_s=linear_speed,
                angular_speed_rad_s=angular_speed,
                radius_m=radius,
                speed_bound_m_s=linear_speed + angular_speed * radius,
            )
        )
    return reports


def load_ignored_geometry_pairs(path: str | Path) -> set[tuple[str, str]]:
    source = Path(path)
    if not source.is_file():
        return set()
    data = json.loads(source.read_text(encoding="utf-8"))
    pairs = set()
    for item in data.get("ignored_geometry_pairs", []):
        pairs.add(_ordered_pair(str(item["geometry_a"]), str(item["geometry_b"])))
    return pairs


def apply_collision_blacklist(
    geometry_model: pin.GeometryModel,
    path: str | Path,
) -> int:
    ignored = load_ignored_geometry_pairs(path)
    if not ignored:
        return 0
    to_keep = []
    removed = 0
    for pair in geometry_model.collisionPairs:
        name_a = geometry_model.geometryObjects[pair.first].name
        name_b = geometry_model.geometryObjects[pair.second].name
        if _ordered_pair(name_a, name_b) in ignored:
            removed += 1
        else:
            to_keep.append(pair)
    geometry_model.removeAllCollisionPairs()
    for pair in to_keep:
        geometry_model.addCollisionPair(pair)
    return removed


def _write_named_positions(
    model: pin.Model,
    q: np.ndarray,
    positions: Mapping[str, float],
) -> None:
    for idx, name in enumerate(model.names):
        if idx == 0 or name not in positions:
            continue
        joint = model.joints[idx]
        value = float(positions[name])
        if joint.nq == 1:
            q[joint.idx_q] = value
        elif joint.nq == 2 and joint.nv == 1:
            q[joint.idx_q] = np.cos(value)
            q[joint.idx_q + 1] = np.sin(value)
        else:
            raise ValueError(f"unsupported joint configuration dimension for {name}: {joint.nq}")


def _geometry_local_radius(obj: pin.GeometryObject) -> float:
    geometry = obj.geometry
    type_name = type(geometry).__name__
    if type_name == "Box":
        return float(np.linalg.norm(np.asarray(geometry.halfSide, dtype=np.float64)))
    if type_name == "Cylinder":
        return float(np.hypot(float(geometry.radius), float(geometry.halfLength)))
    mesh_path = Path(str(getattr(obj, "meshPath", "")))
    if mesh_path.is_file():
        vertices = _read_stl_vertices(mesh_path)
        if vertices.size:
            scale = np.asarray(getattr(obj, "meshScale", np.ones(3)), dtype=np.float64)
            vertices = vertices * scale.reshape(1, 3)
            return float(np.max(np.linalg.norm(vertices, axis=1)))
    return 0.0


def _read_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) >= 84:
        triangle_count = int.from_bytes(data[80:84], "little", signed=False)
        expected = 84 + triangle_count * 50
        if expected == len(data):
            vertices = np.empty((triangle_count * 3, 3), dtype=np.float64)
            offset = 84
            out = 0
            for _ in range(triangle_count):
                offset += 12
                vertices[out : out + 3] = np.frombuffer(
                    data,
                    dtype="<f4",
                    count=9,
                    offset=offset,
                ).reshape(3, 3)
                out += 3
                offset += 36 + 2
            return vertices

    parsed = []
    for raw_line in data.decode("utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("vertex "):
            parsed.append([float(value) for value in line.split()[1:4]])
    return np.asarray(parsed, dtype=np.float64).reshape(-1, 3)


def _frame_name(model: pin.Model, frame_id: int) -> str:
    if 0 <= frame_id < len(model.frames):
        return str(model.frames[frame_id].name)
    return ""


def _joint_name(model: pin.Model, joint_id: int) -> str:
    if 0 <= joint_id < len(model.names):
        return str(model.names[joint_id])
    return ""


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)
