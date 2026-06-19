#!/usr/bin/env python3
"""Replay a TIAGo full-joint-state history with robot mesh and EE trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TIAGO_SRC_DIR = REPO_ROOT / "tiago_src"
DEFAULT_LOCAL_URDF = REPO_ROOT / "TiagoProURDF" / "tiago_pro.urdf"
DEFAULT_WORKSPACE_URDF = Path("/workspace/GATO/TiagoProURDF/tiago_pro.urdf")
DEFAULT_COLLISION_URDF = (
    REPO_ROOT
    / "tiago_configs"
    / "collision_alpha_wrap_a3_o25_cap001"
    / "tiago_pro_collision_alpha_wrap_a3_o25_cap001.urdf"
)
DEFAULT_URDF = (
    DEFAULT_COLLISION_URDF
    if DEFAULT_COLLISION_URDF.is_file()
    else DEFAULT_LOCAL_URDF
    if DEFAULT_LOCAL_URDF.is_file()
    else DEFAULT_WORKSPACE_URDF
)
ARM_RIGHT_TOOL_FRAME = "arm_right_tool_link"
TORSO_FRAME = "torso_lift_link"
MAX_FACES_PER_MESH = 600
REFERENCE_DT = 0.008
TRAJECTORY_FRAME = "torso"
ROBOT_STYLE = "wireframe"
ROBOT_ALPHA = 1.0
WIRE_ALPHA = 1.0
EE_AXIS_LENGTH = 0.09
INTERACTIVE_TIMER_INTERVAL_MS = 1
OUTPUT_FPS = 20

if str(TIAGO_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TIAGO_SRC_DIR))

import review_collision_pairs as _collision_render_helpers  # noqa: E402
from review_collision_pairs import (  # noqa: E402
    _downsample_triangles,
    _local_triangles,
    _resolve_captured_path,
    _transform_triangles,
    _triangle_edge_segments,
)

_collision_render_helpers.np = np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "state_history",
        type=Path,
        help="full_joint_state_history.jsonl written by a ROS Tiago run.",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--geometry-type",
        choices=("visual", "collision"),
        default="collision",
        help="URDF geometry channel to render for the robot.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help=(
            "Directory used for auto-detecting trajectory overlays. Defaults to "
            "the state-history directory."
        ),
    )
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Disable reference/target and actual trajectory overlays.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write a GIF or PNG instead of opening a window.")
    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_args()
    if args.output is not None:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    import pinocchio as pin

    samples = _load_state_history(_resolve_captured_path(args.state_history))
    if not samples:
        raise SystemExit(f"no state samples in {args.state_history}")

    urdf = _resolve_captured_path(args.urdf)
    model = pin.buildModelFromUrdf(str(urdf))
    geometry_type = pin.GeometryType.VISUAL if args.geometry_type == "visual" else pin.GeometryType.COLLISION
    geometry_model = pin.buildGeomFromUrdf(
        model,
        str(urdf),
        geometry_type,
        None,
        [str(urdf.parent)],
    )
    geometry_data = pin.GeometryData(geometry_model)
    data = model.createData()
    local_triangles = [
        _downsample_triangles(_local_triangles(obj), MAX_FACES_PER_MESH)
        for obj in geometry_model.geometryObjects
    ]

    q_values = [_q_from_positions(model, sample["positions_by_name"]) for sample in samples]
    times = np.asarray([float(sample.get("stamp_sec", idx)) for idx, sample in enumerate(samples)], dtype=np.float64)
    times -= times[0]

    torso_to_world = _frame_placement(pin, model, data, q_values[0], TORSO_FRAME)
    overlays = _load_overlays(args, times, torso_to_world)
    initial_triangles = _world_triangles(pin, model, data, geometry_model, geometry_data, q_values[0], local_triangles)
    robot_points = np.concatenate([tri.reshape(-1, 3) for tri in initial_triangles if tri.size], axis=0)
    limit_points = [robot_points]
    for path in overlays.values():
        if path.points.size:
            limit_points.append(path.points)

    fig = plt.figure(figsize=(9.2, 7.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    collections = []
    for triangles in initial_triangles:
        if ROBOT_STYLE == "wireframe":
            collection = Line3DCollection(
                _triangle_edge_segments(triangles),
                colors=[to_rgba("#1A1A1A", WIRE_ALPHA)],
                linewidths=0.18,
                alpha=WIRE_ALPHA,
            )
        else:
            collection = Poly3DCollection(
                triangles,
                facecolors=to_rgba("#8C8C8C", ROBOT_ALPHA),
                edgecolors=to_rgba("#202020", min(1.0, ROBOT_ALPHA + 0.05)),
                linewidths=0.045,
                alpha=ROBOT_ALPHA,
            )
        ax.add_collection3d(collection)
        collections.append(collection)

    overlay_artists = _draw_overlays(ax, overlays)
    ee_axis_collection = Line3DCollection(
        _ee_axis_segments(
            pin,
            model,
            data,
            q_values[0],
            ARM_RIGHT_TOOL_FRAME,
            EE_AXIS_LENGTH,
        ),
        colors=["#C90016", "#2E8B57", "#003192"],
        linewidths=2.0,
    )
    ax.add_collection3d(ee_axis_collection)

    _set_axes(ax, np.concatenate(limit_points, axis=0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    _style_axes(ax)
    ax.view_init(elev=22, azim=-55)

    def update_at_time(replay_time: float):
        sample_idx = _sample_index_at(times, replay_time)
        q = q_values[sample_idx]
        world_triangles = _world_triangles(pin, model, data, geometry_model, geometry_data, q, local_triangles)
        for collection, triangles in zip(collections, world_triangles):
            if ROBOT_STYLE == "wireframe":
                collection.set_segments(_triangle_edge_segments(triangles))
            else:
                collection.set_verts(triangles)

        ee_axis_collection.set_segments(
            _ee_axis_segments(pin, model, data, q, ARM_RIGHT_TOOL_FRAME, EE_AXIS_LENGTH)
        )
        _update_overlay_points(overlay_artists, overlays, replay_time)
        mode = samples[sample_idx].get("controller_mode", "")
        ax.set_title(f"TIAGo state replay | t={times[sample_idx]:.2f}s | {mode}")
        return [*collections, *overlay_artists.values(), ee_axis_collection]

    update_at_time(0.0)
    fig.tight_layout()

    if args.output is None:
        clock = _RealtimeReplayClock(end_time=float(times[-1]))

        def update_realtime(_frame_number: int):
            replay_time = clock.next_time()
            artists = update_at_time(replay_time)
            if clock.done:
                realtime_anim.event_source.stop()
            return artists

        realtime_anim = animation.FuncAnimation(
            fig,
            update_realtime,
            interval=INTERACTIVE_TIMER_INTERVAL_MS,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()
    elif args.output.suffix.lower() == ".png":
        update_at_time(float(times[-1]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        print(f"wrote: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_times = _output_frame_times(float(times[-1]))
        anim = animation.FuncAnimation(
            fig,
            update_at_time,
            frames=output_times,
            interval=1000.0 / OUTPUT_FPS,
            blit=False,
        )
        anim.save(args.output, writer=animation.PillowWriter(fps=OUTPUT_FPS))
        print(f"wrote: {args.output}")
    plt.close(fig)
    return 0


def _load_state_history(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _q_from_positions(model, positions: dict[str, float | None]) -> np.ndarray:
    import pinocchio as pin

    q = pin.neutral(model)
    for idx, name in enumerate(model.names):
        if idx == 0 or name not in positions or positions[name] is None:
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
    return q


def _world_triangles(pin, model, data, geometry_model, geometry_data, q, local_triangles):
    pin.forwardKinematics(model, data, q)
    pin.updateGeometryPlacements(model, data, geometry_model, geometry_data, q)
    return [
        _transform_triangles(triangles, geometry_data.oMg[idx])
        for idx, triangles in enumerate(local_triangles)
    ]


def _frame_placement(pin, model, data, q, frame_name: str):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    frame_id = model.getFrameId(frame_name)
    if frame_id >= len(model.frames):
        raise ValueError(f"frame not found in URDF model: {frame_name}")
    return data.oMf[frame_id].copy()


def _ee_axis_segments(pin, model, data, q, frame_name: str, length: float) -> np.ndarray:
    placement = _frame_placement(pin, model, data, q, frame_name)
    origin = np.asarray(placement.translation, dtype=np.float64).reshape(3)
    rotation = np.asarray(placement.rotation, dtype=np.float64)
    return np.asarray(
        [
            [origin, origin + length * rotation[:, 0]],
            [origin, origin + length * rotation[:, 1]],
            [origin, origin + length * rotation[:, 2]],
        ],
        dtype=np.float64,
    )


class _OverlayPath:
    def __init__(self, points: np.ndarray, times: np.ndarray | None = None):
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.times = None if times is None else np.asarray(times, dtype=np.float64).reshape(-1)

    def index_at(self, t: float) -> int:
        if self.points.size == 0:
            return 0
        if self.times is None or self.times.size != self.points.shape[0]:
            if self.points.shape[0] == 1:
                return 0
            return int(np.clip(round(t), 0, self.points.shape[0] - 1))
        return int(np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, self.points.shape[0] - 1))


class _RealtimeReplayClock:
    def __init__(self, end_time: float):
        self.end_time = max(0.0, float(end_time))
        self.replay_time = 0.0
        self.last_render_time = time.perf_counter()
        self.done = False

    def next_time(self) -> float:
        now = time.perf_counter()
        self.replay_time += now - self.last_render_time
        self.last_render_time = now
        if self.replay_time >= self.end_time:
            self.replay_time = self.end_time
            self.done = True
        return self.replay_time


def _load_overlays(args: argparse.Namespace, state_times: np.ndarray, torso_to_world) -> dict[str, _OverlayPath]:
    if args.no_trajectories:
        return {}
    artifact_dir = _artifact_dir(args)
    reference_path = _existing(artifact_dir / "reference.csv")
    trajectory_path = _auto_trajectory_csv(artifact_dir)

    overlays: dict[str, _OverlayPath] = {}
    if reference_path is not None:
        reference = _load_xyz_csv(_resolve_captured_path(reference_path))
        overlays["reference"] = _OverlayPath(
            _trajectory_points_to_world(reference, TRAJECTORY_FRAME, torso_to_world),
            _reference_times(reference.shape[0], REFERENCE_DT),
        )
    if trajectory_path is not None:
        target, actual, trajectory_times = _load_trajectory_csv(_resolve_captured_path(trajectory_path))
        if target is not None and "reference" not in overlays:
            overlays["target"] = _OverlayPath(
                _trajectory_points_to_world(target, TRAJECTORY_FRAME, torso_to_world),
                trajectory_times,
            )
        if actual is not None:
            overlays["actual"] = _OverlayPath(
                _trajectory_points_to_world(actual, TRAJECTORY_FRAME, torso_to_world),
                trajectory_times,
            )
    return overlays


def _artifact_dir(args: argparse.Namespace) -> Path:
    if args.artifact_dir is not None:
        return _resolve_captured_path(args.artifact_dir, expect_dir=True)
    history_dir = _resolve_captured_path(args.state_history).parent
    return history_dir


def _existing(path: Path) -> Path | None:
    return path if path.is_file() else None


def _auto_trajectory_csv(artifact_dir: Path) -> Path | None:
    candidates = [
        artifact_dir / "batch_1_actual.csv",
        artifact_dir / "trajectory.csv",
        artifact_dir / "data" / "trajectory.csv",
    ]
    if artifact_dir.name == "data":
        candidates.append(artifact_dir / "trajectory.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(artifact_dir.glob("batch_*_actual.csv"))
    return matches[0] if matches else None


def _load_xyz_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] < 3:
        raise ValueError(f"expected at least 3 columns in {path}")
    return data[:, :3]


def _load_trajectory_csv(path: Path) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    name = path.name
    if name.startswith("batch_") and name.endswith("_actual.csv"):
        if data.shape[1] < 8:
            raise ValueError(f"expected fig8 batch actual format in {path}")
        return data[:, 2:5], data[:, 5:8], data[:, 0] - data[0, 0]
    if data.shape[1] >= 7:
        return data[:, 1:4], data[:, 4:7], data[:, 0] - data[0, 0]
    raise ValueError(f"unsupported trajectory CSV format: {path}")


def _reference_times(point_count: int, dt: float) -> np.ndarray | None:
    if point_count <= 0:
        return None
    if point_count == 1:
        return np.asarray([0.0], dtype=np.float64)
    return np.arange(point_count, dtype=np.float64) * float(dt)


def _sample_index_at(times: np.ndarray, replay_time: float) -> int:
    return int(np.clip(np.searchsorted(times, replay_time, side="right") - 1, 0, len(times) - 1))


def _output_frame_times(end_time: float) -> np.ndarray:
    if end_time <= 0.0:
        return np.asarray([0.0], dtype=np.float64)
    step = 1.0 / OUTPUT_FPS
    return np.arange(0.0, end_time + step, step, dtype=np.float64)


def _trajectory_points_to_world(points: np.ndarray, frame: str, torso_to_world) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if frame == "world":
        return points
    return points @ torso_to_world.rotation.T + torso_to_world.translation.reshape(1, 3)


def _draw_overlays(ax, overlays: dict[str, _OverlayPath]) -> dict[str, object]:
    styles = {
        "reference": {"color": "#747474", "linestyle": ":", "linewidth": 1.2, "label": "reference"},
        "target": {"color": "#C90016", "linestyle": ":", "linewidth": 1.2, "label": "target"},
        "actual": {"color": "#003192", "linestyle": "-", "linewidth": 1.7, "label": "actual"},
    }
    artists = {}
    for name, path in overlays.items():
        if path.points.size == 0:
            continue
        style = styles.get(name, {"color": "#111111", "linewidth": 1.2, "label": name})
        line = ax.plot(
            path.points[:, 0],
            path.points[:, 1],
            path.points[:, 2],
            **style,
        )[0]
        point = ax.plot(
            [path.points[0, 0]],
            [path.points[0, 1]],
            [path.points[0, 2]],
            "o",
            color=style["color"],
            markersize=5,
        )[0]
        artists[f"{name}_line"] = line
        artists[f"{name}_point"] = point
    if artists:
        ax.legend(loc="upper right")
    return artists


def _update_overlay_points(artists: dict[str, object], overlays: dict[str, _OverlayPath], t: float) -> None:
    for name, path in overlays.items():
        point = artists.get(f"{name}_point")
        if point is None or path.points.size == 0:
            continue
        idx = path.index_at(t)
        xyz = path.points[idx]
        point.set_data([xyz[0]], [xyz[1]])
        point.set_3d_properties([xyz[2]])


def _set_axes(ax, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        center = np.zeros(3)
        radius = 1.0
    else:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = max(0.25, 0.58 * float(np.max(maxs - mins)))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)


def _style_axes(ax) -> None:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.42)
        axis._axinfo["grid"]["linewidth"] = 0.55
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))


if __name__ == "__main__":
    raise SystemExit(main())
