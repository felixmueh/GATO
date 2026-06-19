"""Offline collision-review helpers for TIAGo safety artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pinocchio as pin

from gato_tiago.safety_monitor import TiagoCollisionModel


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
            "nearest_point_a": (
                list(self.nearest_point_a) if self.nearest_point_a else None
            ),
            "nearest_point_b": (
                list(self.nearest_point_b) if self.nearest_point_b else None
            ),
        }


@dataclass(frozen=True)
class CollisionMinimumDistance:
    """Minimum distance result without per-pair debug report allocation."""

    index: int
    distance_m: float
    in_collision: bool


def compute_pair_distances(
    collision_model: TiagoCollisionModel,
    q: np.ndarray,
) -> list[CollisionPairReport]:
    _, geometry_data = _compute_all_pair_distances(collision_model, q)
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


def compute_pair_distance_values(
    collision_model: TiagoCollisionModel,
    q: np.ndarray,
) -> np.ndarray:
    """Return raw pair distances aligned with geometry_model.collisionPairs."""
    _, geometry_data = _compute_all_pair_distances(collision_model, q)
    return np.asarray(
        [
            float(geometry_data.distanceResults[idx].min_distance)
            for idx in range(len(collision_model.geometry_model.collisionPairs))
        ],
        dtype=np.float64,
    )


def compute_minimum_pair_distance(
    collision_model: TiagoCollisionModel,
    q: np.ndarray,
) -> CollisionMinimumDistance:
    """Return only the closest pair distance for offline analysis."""
    distances = compute_pair_distance_values(collision_model, q)
    if distances.size == 0:
        return CollisionMinimumDistance(
            index=-1,
            distance_m=float("inf"),
            in_collision=False,
        )
    index = int(np.argmin(distances))
    distance = float(distances[index])
    return CollisionMinimumDistance(
        index=index,
        distance_m=distance,
        in_collision=distance <= 0.0,
    )


def compute_pair_distance_reports_for_indices(
    collision_model: TiagoCollisionModel,
    q: np.ndarray,
    pair_indices: Iterable[int],
) -> list[CollisionPairReport]:
    """Compute exact distance reports only for selected collision-pair indices."""
    data, geometry_data = collision_model.make_data()
    pin.updateGeometryPlacements(
        collision_model.model,
        data,
        collision_model.geometry_model,
        geometry_data,
        np.asarray(q, dtype=np.float64),
    )
    reports = []
    for idx in pair_indices:
        pair_index = int(idx)
        pair = collision_model.geometry_model.collisionPairs[pair_index]
        object_a = collision_model.geometry_model.geometryObjects[pair.first]
        object_b = collision_model.geometry_model.geometryObjects[pair.second]
        result = pin.computeDistance(
            collision_model.geometry_model,
            geometry_data,
            pair_index,
        )
        distance = float(result.min_distance)
        reports.append(
            CollisionPairReport(
                index=pair_index,
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


def collision_model_metadata(collision_model: TiagoCollisionModel) -> dict[str, object]:
    return {
        "state_fixed_joints": list(collision_model.state_fixed_joint_names),
        "state_updated_joints": list(collision_model.state_updated_joint_names),
        "monitored_geometry_parent_joints": list(
            collision_model.monitored_geometry_parent_joint_names
        ),
        "monitored_geometry_indices": list(collision_model.monitored_geometry_indices),
        "monitor_only_collision_pairs": bool(collision_model.monitor_only_collision_pairs),
        "geometry_count": len(collision_model.geometry_model.geometryObjects),
        "collision_pair_count": len(collision_model.geometry_model.collisionPairs),
    }


def geometry_objects_json(collision_model: TiagoCollisionModel) -> list[dict[str, object]]:
    objects = []
    monitored = set(collision_model.monitored_geometry_indices)
    for idx, obj in enumerate(collision_model.geometry_model.geometryObjects):
        objects.append(
            {
                "index": idx,
                "name": obj.name,
                "type": type(obj.geometry).__name__,
                "link": _frame_name(collision_model.model, obj.parentFrame),
                "parent_joint": _joint_name(collision_model.model, obj.parentJoint),
                "monitored": idx in monitored,
                "mesh_path": str(getattr(obj, "meshPath", "")),
                "mesh_scale": [float(value) for value in getattr(obj, "meshScale", [])],
                "radius_m": float(collision_model.geometry_radii_m[idx]),
            }
        )
    return objects


def _compute_all_pair_distances(
    collision_model: TiagoCollisionModel,
    q: np.ndarray,
) -> tuple[pin.Data, pin.GeometryData]:
    data, geometry_data = collision_model.make_data()
    pin.computeDistances(
        collision_model.model,
        data,
        collision_model.geometry_model,
        geometry_data,
        np.asarray(q, dtype=np.float64),
    )
    return data, geometry_data


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
