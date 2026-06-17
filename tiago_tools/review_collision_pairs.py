#!/usr/bin/env python3
"""Interactively review Tiago collision pairs and write a blacklist."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TIAGO_SRC_DIR = REPO_ROOT / "tiago_src"


MAX_CONTEXT_FACES_PER_MESH = 220
MAX_HIGHLIGHT_FACES_PER_MESH = 1400
REPO_ROOT_MARKER = "GATO"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review captured SafetyMonitor collision pairs and persist "
            "ignore/monitor decisions immediately."
        )
    )
    parser.add_argument("review_data", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tiago_collision_blacklist.json"),
        help="Decision/blacklist JSON written after every pair decision.",
    )
    parser.add_argument(
        "--show-reviewed",
        action="store_true",
        help="Do not skip pairs already present in the output file.",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Review all monitored candidate pairs, including pairs that are clear.",
    )
    parser.add_argument(
        "--distance-below-m",
        type=float,
        default=None,
        help=(
            "Review captured pairs with distance below this threshold in metres. "
            "Already reviewed pairs in --output are skipped unless --show-reviewed is set."
        ),
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Disable the matplotlib pair viewer and use text-only review.",
    )
    args = parser.parse_args()
    if args.distance_below_m is not None and args.distance_below_m < 0.0:
        parser.error("--distance-below-m must be non-negative")
    return args


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _initial_decisions(review_data: dict, output_path: Path) -> dict:
    existing = _load_json(output_path)
    if existing:
        existing.setdefault("ignored_geometry_pairs", [])
        existing.setdefault("monitored_geometry_pairs", [])
        return existing
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": None,
        "review_data_path": str(review_data.get("_source_path", "")),
        "urdf_path": review_data.get("urdf_path"),
        "model": review_data.get("model", {}),
        "ignored_geometry_pairs": [],
        "monitored_geometry_pairs": [],
    }


def _reviewed_pairs(decisions: dict) -> set[tuple[str, str]]:
    reviewed = set()
    for key in ("ignored_geometry_pairs", "monitored_geometry_pairs"):
        for item in decisions.get(key, []):
            reviewed.add(_ordered_pair(str(item["geometry_a"]), str(item["geometry_b"])))
    return reviewed


def _forget_pair(decisions: dict, pair_key: tuple[str, str]) -> None:
    for key in ("ignored_geometry_pairs", "monitored_geometry_pairs"):
        decisions[key] = [
            item
            for item in decisions.get(key, [])
            if _ordered_pair(str(item["geometry_a"]), str(item["geometry_b"])) != pair_key
        ]


def _persist(path: Path, decisions: dict) -> None:
    decisions["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decisions, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_pair(pair: dict, offset: int, total: int) -> None:
    status = "COLLISION" if pair.get("in_collision") else "clear"
    print()
    print("=" * 78)
    print(f"[{offset}/{total}] pair index {pair['index']} ({status})")
    print(f"geometry_a:     {pair['geometry_a']}")
    print(f"link_a:         {pair.get('link_a', '')}")
    print(f"parent_joint_a: {pair.get('parent_joint_a', '')}")
    print(f"geometry_b:     {pair['geometry_b']}")
    print(f"link_b:         {pair.get('link_b', '')}")
    print(f"parent_joint_b: {pair.get('parent_joint_b', '')}")
    print(f"distance_m:     {float(pair['distance_m']):+.6f}")
    if pair.get("nearest_point_a") is not None:
        print(f"nearest_a:      {_fmt_vec(pair['nearest_point_a'])}")
    if pair.get("nearest_point_b") is not None:
        print(f"nearest_b:      {_fmt_vec(pair['nearest_point_b'])}")
    print("=" * 78)


def _fmt_vec(values) -> str:
    return "[" + ", ".join(f"{float(value):+.4f}" for value in values) + "]"


def _decision_entry(pair: dict, decision: str, reason: str) -> dict:
    return {
        "geometry_a": pair["geometry_a"],
        "geometry_b": pair["geometry_b"],
        "link_a": pair.get("link_a", ""),
        "link_b": pair.get("link_b", ""),
        "distance_m_at_review": float(pair["distance_m"]),
        "decision": decision,
        "reason": reason,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }


def _candidate_repo_roots() -> list[Path]:
    roots = []
    for raw in (Path.cwd(), REPO_ROOT):
        roots.append(raw)
        if ".worktree" in raw.parts:
            roots.append(Path(*raw.parts[: raw.parts.index(".worktree")]))

    unique = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _resolve_captured_path(value: str | Path, *, expect_dir: bool = False) -> Path:
    path = Path(value)
    if _path_matches(path, expect_dir=expect_dir):
        return path

    candidates = []
    if path.is_absolute() and REPO_ROOT_MARKER in path.parts:
        suffix = Path(*path.parts[path.parts.index(REPO_ROOT_MARKER) + 1 :])
        candidates.extend(root / suffix for root in _candidate_repo_roots())
    elif not path.is_absolute():
        candidates.extend(root / path for root in _candidate_repo_roots())

    for candidate in candidates:
        if _path_matches(candidate, expect_dir=expect_dir):
            return candidate
    return path


def _path_matches(path: Path, *, expect_dir: bool) -> bool:
    return path.is_dir() if expect_dir else path.is_file()


class _PairRenderer:
    def __init__(self, review_data: dict) -> None:
        global np

        import sys

        if str(TIAGO_SRC_DIR) not in sys.path:
            sys.path.insert(0, str(TIAGO_SRC_DIR))

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np
        import pinocchio as pin

        from gato_tiago.safety_monitor import build_tiago_collision_model

        self.plt = plt
        self.Poly3DCollection = Poly3DCollection
        self.pin = pin

        urdf_path = _resolve_captured_path(str(review_data["urdf_path"]))
        package_dirs = [
            _resolve_captured_path(path, expect_dir=True)
            for path in review_data.get("package_dirs", [urdf_path.parent])
        ]
        locked_joints = tuple(review_data.get("model", {}).get("locked_joints", []))
        reference_positions = review_data.get("state", {}).get("positions_by_name", {})
        self.collision_model = build_tiago_collision_model(
            urdf_path=urdf_path,
            package_dirs=package_dirs,
            locked_joint_names=locked_joints,
            reference_positions=reference_positions,
        )
        self.q = np.asarray(review_data["state"]["q"], dtype=np.float64)
        self._geometry_by_name = {
            obj.name: idx
            for idx, obj in enumerate(self.collision_model.geometry_model.geometryObjects)
        }
        self._world_triangles = self._compute_world_triangles()

        self.fig = self.plt.figure(figsize=(9, 7), dpi=120)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.plt.ion()
        self.fig.show()

    def show_pair(self, pair: dict) -> None:
        name_a = str(pair["geometry_a"])
        name_b = str(pair["geometry_b"])
        index_a = self._geometry_by_name.get(name_a)
        index_b = self._geometry_by_name.get(name_b)
        if index_a is None or index_b is None:
            print(f"image unavailable: geometry not found for {name_a} / {name_b}")
            return

        self.ax.clear()
        all_points = []
        for idx, triangles in enumerate(self._world_triangles):
            if idx in {index_a, index_b}:
                continue
            shown = _downsample_triangles(triangles, MAX_CONTEXT_FACES_PER_MESH)
            all_points.append(shown.reshape(-1, 3))
            self.ax.add_collection3d(
                self.Poly3DCollection(
                    shown,
                    facecolors="#C7C7C7",
                    edgecolors="none",
                    linewidths=0.0,
                    alpha=0.11,
                )
            )

        for idx, color in ((index_a, "#D62728"), (index_b, "#1F77B4")):
            shown = _downsample_triangles(self._world_triangles[idx], MAX_HIGHLIGHT_FACES_PER_MESH)
            all_points.append(shown.reshape(-1, 3))
            self.ax.add_collection3d(
                self.Poly3DCollection(
                    shown,
                    facecolors=color,
                    edgecolors="#222222",
                    linewidths=0.05,
                    alpha=0.82,
                )
            )

        nearest_a = pair.get("nearest_point_a")
        nearest_b = pair.get("nearest_point_b")
        if nearest_a is not None and nearest_b is not None:
            p_a = np.asarray(nearest_a, dtype=np.float64)
            p_b = np.asarray(nearest_b, dtype=np.float64)
            all_points.append(np.vstack([p_a, p_b]))
            self.ax.scatter(*p_a, color="#D62728", s=36)
            self.ax.scatter(*p_b, color="#1F77B4", s=36)
            self.ax.plot(
                [p_a[0], p_b[0]],
                [p_a[1], p_b[1]],
                [p_a[2], p_b[2]],
                color="#111111",
                linewidth=1.3,
            )

        self._set_axes(np.concatenate(all_points, axis=0), index_a, index_b)
        distance = float(pair["distance_m"])
        self.ax.set_title(
            f"{name_a}  <->  {name_b}\n"
            f"distance {distance:+.4f} m, pair index {pair['index']}"
        )
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def _compute_world_triangles(self) -> list[np.ndarray]:
        data = self.collision_model.model.createData()
        geometry_data = self.pin.GeometryData(self.collision_model.geometry_model)
        self.pin.forwardKinematics(self.collision_model.model, data, self.q)
        self.pin.updateGeometryPlacements(
            self.collision_model.model,
            data,
            self.collision_model.geometry_model,
            geometry_data,
            self.q,
        )

        triangles = []
        for idx, obj in enumerate(self.collision_model.geometry_model.geometryObjects):
            local = _local_triangles(obj)
            triangles.append(_transform_triangles(local, geometry_data.oMg[idx]))
        return triangles

    def _set_axes(self, points: np.ndarray, index_a: int, index_b: int) -> None:
        highlight = np.concatenate(
            [
                self._world_triangles[index_a].reshape(-1, 3),
                self._world_triangles[index_b].reshape(-1, 3),
            ],
            axis=0,
        )
        mins = highlight.min(axis=0)
        maxs = highlight.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = max(0.08, 0.8 * float(np.max(maxs - mins)))
        if np.isfinite(points).all():
            radius = max(radius, 0.2 * float(np.max(points.max(axis=0) - points.min(axis=0))))

        self.ax.set_xlim(center[0] - radius, center[0] + radius)
        self.ax.set_ylim(center[1] - radius, center[1] + radius)
        self.ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.view_init(elev=22, azim=-55)


def _make_renderer(review_data: dict, enabled: bool):
    if not enabled:
        return None
    try:
        return _PairRenderer(review_data)
    except Exception as exc:
        print(f"image viewer disabled: {exc}")
        return None


def _read_stl_triangles(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) >= 84:
        triangle_count = int.from_bytes(data[80:84], "little", signed=False)
        expected = 84 + triangle_count * 50
        if expected == len(data):
            triangles = np.empty((triangle_count, 3, 3), dtype=np.float64)
            offset = 84
            for idx in range(triangle_count):
                offset += 12
                triangles[idx] = np.frombuffer(
                    data,
                    dtype="<f4",
                    count=9,
                    offset=offset,
                ).reshape(3, 3)
                offset += 36 + 2
            return triangles

    vertices = []
    for raw_line in data.decode("utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("vertex "):
            vertices.append([float(value) for value in line.split()[1:4]])
    if len(vertices) % 3:
        raise ValueError(f"ASCII STL vertex count is not divisible by 3: {path}")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def _box_triangles(half_side: np.ndarray) -> np.ndarray:
    hx, hy, hz = np.asarray(half_side, dtype=np.float64)
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return vertices[faces]


def _cylinder_triangles(radius: float, half_length: float, segments: int = 32) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    circle = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    top_center = np.array([0.0, 0.0, half_length])
    bottom_center = np.array([0.0, 0.0, -half_length])
    triangles = []
    for idx in range(segments):
        next_idx = (idx + 1) % segments
        bottom_i = np.array([circle[idx, 0], circle[idx, 1], -half_length])
        bottom_j = np.array([circle[next_idx, 0], circle[next_idx, 1], -half_length])
        top_i = np.array([circle[idx, 0], circle[idx, 1], half_length])
        top_j = np.array([circle[next_idx, 0], circle[next_idx, 1], half_length])
        triangles.append([bottom_i, bottom_j, top_j])
        triangles.append([bottom_i, top_j, top_i])
        triangles.append([top_center, top_i, top_j])
        triangles.append([bottom_center, bottom_j, bottom_i])
    return np.asarray(triangles, dtype=np.float64)


def _local_triangles(geometry_object) -> np.ndarray:
    geometry = geometry_object.geometry
    type_name = type(geometry).__name__
    if type_name == "Box":
        return _box_triangles(np.asarray(geometry.halfSide, dtype=np.float64))
    if type_name == "Cylinder":
        return _cylinder_triangles(float(geometry.radius), float(geometry.halfLength))
    if type_name == "BVHModelOBBRSS":
        triangles = _read_stl_triangles(Path(str(geometry_object.meshPath)))
        scale = np.asarray(geometry_object.meshScale, dtype=np.float64)
        return triangles * scale.reshape(1, 1, 3)
    raise TypeError(f"unsupported collision geometry: {type_name}")


def _transform_triangles(triangles: np.ndarray, placement) -> np.ndarray:
    return triangles @ placement.rotation.T + placement.translation.reshape(1, 1, 3)


def _downsample_triangles(triangles: np.ndarray, max_faces: int) -> np.ndarray:
    if triangles.shape[0] <= max_faces:
        return triangles
    step = int(np.ceil(triangles.shape[0] / max_faces))
    return triangles[::step]


def main() -> int:
    args = parse_args()
    review_data = _load_json(args.review_data)
    review_data["_source_path"] = str(args.review_data)
    pairs = list(review_data.get("collision_pairs", []))
    if not pairs:
        raise SystemExit(f"no collision_pairs in {args.review_data}")
    if args.distance_below_m is not None:
        pairs = [
            pair
            for pair in pairs
            if float(pair.get("distance_m", float("inf"))) < args.distance_below_m
        ]
    elif not args.all_pairs:
        pairs = [pair for pair in pairs if pair.get("in_collision")]
    if not pairs:
        if args.distance_below_m is not None:
            mode = f"pairs below {args.distance_below_m:.6f} m"
        else:
            mode = "all candidate pairs" if args.all_pairs else "colliding pairs"
        raise SystemExit(f"no {mode} in {args.review_data}")

    decisions = _initial_decisions(review_data, args.output)
    reviewed = _reviewed_pairs(decisions)
    renderer = _make_renderer(review_data, enabled=not args.no_images)
    review_pairs = []
    for pair in pairs:
        pair_key = _ordered_pair(str(pair["geometry_a"]), str(pair["geometry_b"]))
        if pair_key in reviewed and not args.show_reviewed:
            continue
        review_pairs.append(pair)

    total = len(review_pairs)
    if total == 0:
        _persist(args.output, decisions)
        print("no unreviewed pairs matched the selected filter")
        print(f"saved {args.output}")
        return 0

    for offset, pair in enumerate(review_pairs, start=1):
        pair_key = _ordered_pair(str(pair["geometry_a"]), str(pair["geometry_b"]))

        while True:
            if renderer is not None:
                renderer.show_pair(pair)
            _print_pair(pair, offset, total)
            command = input("[i]gnore  [m]onitor  [p]rint  [v]iew  [q]uit > ").strip().lower()
            if command == "q":
                _persist(args.output, decisions)
                print(f"saved {args.output}")
                return 0
            if command == "p" or command == "" or command == "v":
                continue
            if command not in {"i", "m"}:
                print("unknown command")
                continue

            reason = input("reason (optional) > ").strip()
            _forget_pair(decisions, pair_key)
            if command == "i":
                decisions["ignored_geometry_pairs"].append(
                    _decision_entry(pair, "ignore", reason or "operator reviewed")
                )
                print("recorded: ignore")
            else:
                decisions["monitored_geometry_pairs"].append(
                    _decision_entry(pair, "monitor", reason or "operator reviewed")
                )
                print("recorded: monitor")
            reviewed.add(pair_key)
            _persist(args.output, decisions)
            break

    _persist(args.output, decisions)
    ignored = len(decisions.get("ignored_geometry_pairs", []))
    monitored = len(decisions.get("monitored_geometry_pairs", []))
    print(f"review complete: ignored={ignored} monitored={monitored}")
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
