"""Collision-safety helpers for live TIAGo controller experiments.

The module is intentionally ROS-free. ROS-facing tools pass named joint-state
snapshots in, and this module owns the Pinocchio collision model, state mapping,
distance checks, and blacklist handling.

Current scope is collision distance and collision-body Cartesian speed. Joint
limit checks are not implemented here yet.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pinocchio as pin


REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_URDF_PATH = REPO_ROOT / "TiagoProURDF" / "tiago_pro.urdf"
_WORKSPACE_URDF_PATH = Path("/workspace/GATO/TiagoProURDF/tiago_pro.urdf")
DEFAULT_URDF_PATH = (
    _LOCAL_URDF_PATH if _LOCAL_URDF_PATH.is_file() else _WORKSPACE_URDF_PATH
)
DEFAULT_PACKAGE_DIRS = (DEFAULT_URDF_PATH.parent,)
DEFAULT_LOCKED_JOINTS = (
    "wheel_front_left_joint",
    "wheel_front_right_joint",
    "wheel_rear_left_joint",
    "wheel_rear_right_joint",
)
GRIPPER_JOINT_MARKER = "gripper_"


@dataclass(frozen=True)
class NamedJointState:
    """A ROS-independent named joint-state snapshot."""

    position: dict[str, float]
    velocity: dict[str, float]
    stamp_sec: float | None = None

    @classmethod
    def from_sequences(
        cls,
        names: Sequence[str],
        positions: Sequence[float],
        velocities: Sequence[float],
        *,
        stamp_sec: float | None = None,
    ) -> "NamedJointState":
        if len(names) != len(positions):
            raise ValueError(
                f"joint name/position length mismatch: {len(names)} != {len(positions)}"
            )
        if len(names) != len(velocities):
            raise ValueError(
                f"joint name/velocity length mismatch: {len(names)} != {len(velocities)}"
            )
        return cls(
            position={name: float(value) for name, value in zip(names, positions)},
            velocity={name: float(value) for name, value in zip(names, velocities)},
            stamp_sec=stamp_sec,
        )


@dataclass(frozen=True)
class CollisionPairReport:
    """Distance report for one Pinocchio geometry pair."""

    index: int
    geometry_a: str
    geometry_b: str
    link_a: str
    link_b: str
    parent_joint_a: str
    parent_joint_b: str
    distance_m: float
    in_collision: bool
    nearest_point_a: tuple[float, float, float] | None
    nearest_point_b: tuple[float, float, float] | None

    def to_json(self) -> dict[str, object]:
        return {
            "index": self.index,
            "geometry_a": self.geometry_a,
            "geometry_b": self.geometry_b,
            "link_a": self.link_a,
            "link_b": self.link_b,
            "parent_joint_a": self.parent_joint_a,
            "parent_joint_b": self.parent_joint_b,
            "distance_m": self.distance_m,
            "in_collision": self.in_collision,
            "nearest_point_a": list(self.nearest_point_a) if self.nearest_point_a else None,
            "nearest_point_b": list(self.nearest_point_b) if self.nearest_point_b else None,
        }


@dataclass(frozen=True)
class CollisionBodySpeedReport:
    """Conservative speed bound for one collision geometry object."""

    geometry: str
    link: str
    parent_joint: str
    linear_speed_m_s: float
    angular_speed_rad_s: float
    radius_m: float
    speed_bound_m_s: float


@dataclass
class TiagoCollisionModel:
    """Reduced Pinocchio model plus collision geometry."""

    model: pin.Model
    geometry_model: pin.GeometryModel
    locked_joint_names: tuple[str, ...]
    unlocked_joint_names: tuple[str, ...]
    geometry_radii_m: np.ndarray

    def make_data(self) -> tuple[pin.Data, pin.GeometryData]:
        # GeometryData must be constructed after the final collision-pair list is
        # in place. Mutating geometry_model.collisionPairs afterwards can crash.
        return self.model.createData(), pin.GeometryData(self.geometry_model)


def gripper_joint_names(model: pin.Model) -> tuple[str, ...]:
    return tuple(
        name
        for idx, name in enumerate(model.names)
        if idx != 0 and GRIPPER_JOINT_MARKER in name
    )


def default_locked_joint_names(*, lock_grippers: bool = False) -> tuple[str, ...]:
    names = list(DEFAULT_LOCKED_JOINTS)
    if lock_grippers:
        full_model = pin.buildModelFromUrdf(str(DEFAULT_URDF_PATH))
        names.extend(gripper_joint_names(full_model))
    return tuple(names)


def build_tiago_collision_model(
    *,
    urdf_path: str | Path = DEFAULT_URDF_PATH,
    package_dirs: Sequence[str | Path] | None = None,
    locked_joint_names: Iterable[str] = DEFAULT_LOCKED_JOINTS,
    reference_positions: Mapping[str, float] | None = None,
    blacklist_path: str | Path | None = None,
) -> TiagoCollisionModel:
    """Build the reduced full-body collision model used by SafetyMonitor."""
    urdf = Path(urdf_path)
    package_dirs = tuple(Path(path) for path in (package_dirs or (urdf.parent,)))
    full_model = pin.buildModelFromUrdf(str(urdf))
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

    locked_names = tuple(locked_joint_names)
    locked_ids = []
    missing_locked = []
    for name in locked_names:
        joint_id = full_model.getJointId(name)
        if joint_id >= full_model.njoints:
            missing_locked.append(name)
        else:
            locked_ids.append(joint_id)
    if missing_locked:
        raise ValueError(f"locked joints not found in URDF model: {missing_locked}")

    model, geometry_model = pin.buildReducedModel(
        full_model,
        full_geom,
        locked_ids,
        reference_q,
    )
    geometry_model.addAllCollisionPairs()
    if blacklist_path is not None:
        apply_collision_blacklist(geometry_model, blacklist_path)

    unlocked = tuple(name for idx, name in enumerate(model.names) if idx != 0)
    radii = np.asarray(
        [_geometry_radius_from_parent_frame(obj) for obj in geometry_model.geometryObjects],
        dtype=np.float64,
    )
    return TiagoCollisionModel(
        model=model,
        geometry_model=geometry_model,
        locked_joint_names=locked_names,
        unlocked_joint_names=unlocked,
        geometry_radii_m=radii,
    )


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


def compute_pair_distances(
    collision_model: TiagoCollisionModel,
    q: np.ndarray,
) -> list[CollisionPairReport]:
    data, geometry_data = collision_model.make_data()
    pin.computeDistances(
        collision_model.model,
        data,
        collision_model.geometry_model,
        geometry_data,
        np.asarray(q, dtype=np.float64),
    )
    reports = []
    for idx, pair in enumerate(collision_model.geometry_model.collisionPairs):
        object_a = collision_model.geometry_model.geometryObjects[pair.first]
        object_b = collision_model.geometry_model.geometryObjects[pair.second]
        result = geometry_data.distanceResults[idx]
        distance = float(result.min_distance)
        reports.append(
            CollisionPairReport(
                index=idx,
                geometry_a=object_a.name,
                geometry_b=object_b.name,
                link_a=_frame_name(collision_model.model, object_a.parentFrame),
                link_b=_frame_name(collision_model.model, object_b.parentFrame),
                parent_joint_a=_joint_name(collision_model.model, object_a.parentJoint),
                parent_joint_b=_joint_name(collision_model.model, object_b.parentJoint),
                distance_m=distance,
                in_collision=distance <= 0.0,
                nearest_point_a=_point_tuple(result.getNearestPoint1()),
                nearest_point_b=_point_tuple(result.getNearestPoint2()),
            )
        )
    return reports


def compute_collision_body_speeds(
    collision_model: TiagoCollisionModel,
    q: np.ndarray,
    v: np.ndarray,
) -> list[CollisionBodySpeedReport]:
    data, geometry_data = collision_model.make_data()
    pin.forwardKinematics(
        collision_model.model,
        data,
        np.asarray(q, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
    )
    pin.updateFramePlacements(collision_model.model, data)
    pin.updateGeometryPlacements(
        collision_model.model,
        data,
        collision_model.geometry_model,
        geometry_data,
        np.asarray(q, dtype=np.float64),
    )
    reports = []
    for idx, obj in enumerate(collision_model.geometry_model.geometryObjects):
        velocity = pin.getFrameVelocity(
            collision_model.model,
            data,
            obj.parentFrame,
            pin.ReferenceFrame.WORLD,
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


def collision_model_metadata(collision_model: TiagoCollisionModel) -> dict[str, object]:
    return {
        "locked_joints": list(collision_model.locked_joint_names),
        "unlocked_joints": list(collision_model.unlocked_joint_names),
        "geometry_count": len(collision_model.geometry_model.geometryObjects),
        "collision_pair_count": len(collision_model.geometry_model.collisionPairs),
    }


def geometry_objects_json(collision_model: TiagoCollisionModel) -> list[dict[str, object]]:
    objects = []
    for idx, obj in enumerate(collision_model.geometry_model.geometryObjects):
        objects.append(
            {
                "index": idx,
                "name": obj.name,
                "type": type(obj.geometry).__name__,
                "link": _frame_name(collision_model.model, obj.parentFrame),
                "parent_joint": _joint_name(collision_model.model, obj.parentJoint),
                "mesh_path": str(getattr(obj, "meshPath", "")),
                "mesh_scale": [float(value) for value in getattr(obj, "meshScale", [])],
                "radius_m": float(collision_model.geometry_radii_m[idx]),
            }
        )
    return objects


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


def _geometry_radius_from_parent_frame(obj: pin.GeometryObject) -> float:
    local_radius = _geometry_local_radius(obj)
    placement_offset = np.asarray(obj.placement.translation, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(placement_offset) + local_radius)


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


def _point_tuple(point: object) -> tuple[float, float, float] | None:
    try:
        arr = np.asarray(point, dtype=np.float64).reshape(3)
    except Exception:
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]))


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
