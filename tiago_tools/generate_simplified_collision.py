#!/usr/bin/env python3
"""Generate the checked-in TIAGo simplified collision URDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_URDF = Path("/workspace/GATO/TiagoProURDF/tiago_pro.urdf")
if not DEFAULT_SOURCE_URDF.is_file():
    DEFAULT_SOURCE_URDF = REPO_ROOT / "TiagoProURDF" / "tiago_pro.urdf"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tiago_configs" / "collision_alpha_wrap_a3_o25_cap001"
DEFAULT_TOOL = Path("/tmp/alpha_wrap_tool_alpha_offset_capped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-urdf", type=Path, default=DEFAULT_SOURCE_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha-wrap-tool", type=Path, default=DEFAULT_TOOL)
    parser.add_argument("--relative-alpha", type=float, default=3.0)
    parser.add_argument("--relative-offset", type=float, default=25.0)
    parser.add_argument("--max-offset", type=float, default=0.01)
    parser.add_argument("--max-alpha", type=float, default=float("inf"))
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument("--containment-tolerance-m", type=float, default=1e-5)
    args = parser.parse_args()
    if not args.source_urdf.is_file():
        parser.error(f"--source-urdf not found: {args.source_urdf}")
    if not args.alpha_wrap_tool.is_file():
        parser.error(f"--alpha-wrap-tool not found: {args.alpha_wrap_tool}")
    if args.relative_alpha <= 0.0 or args.relative_offset <= 0.0:
        parser.error("relative alpha/offset must be positive")
    if args.max_offset <= 0.0:
        parser.error("--max-offset must be positive")
    if args.sample_count <= 0:
        parser.error("--sample-count must be positive")
    return args


def main() -> int:
    args = parse_args()
    tree = ET.parse(args.source_urdf)
    mesh_elements = [
        mesh
        for collision in tree.findall(".//collision")
        for mesh in collision.findall(".//mesh")
        if mesh.get("filename")
    ]
    source_by_name = {
        Path(mesh.get("filename")).name: _resolve_mesh(args.source_urdf, mesh.get("filename"))
        for mesh in mesh_elements
    }

    meshes_dir = args.output_dir / "meshes"
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    meshes_dir.mkdir(parents=True)

    results = []
    for name, source in sorted(source_by_name.items()):
        output = meshes_dir / name
        metadata = _run_alpha_wrap(args, source, output)
        _validate_contains(source, output, args.sample_count, args.containment_tolerance_m)
        results.append({"mesh": name, **metadata})

    for mesh in mesh_elements:
        mesh.set("filename", f"meshes/{Path(mesh.get('filename')).name}")

    artifact_name = args.output_dir.name
    urdf_path = args.output_dir / f"tiago_pro_collision_{artifact_name.removeprefix('collision_')}.urdf"
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    (args.output_dir / "alpha_wrap_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {urdf_path}")
    print(f"wrapped {len(results)} unique collision meshes")
    return 0


def _resolve_mesh(urdf: Path, filename: str) -> Path:
    raw = str(filename)
    if raw.startswith("package://"):
        basename = Path(raw).name
        matches = sorted(urdf.parent.rglob(basename))
        if not matches:
            matches = sorted(Path("/workspace/GATO/TiagoProURDF").rglob(basename))
        if matches:
            return matches[0]
    path = Path(raw)
    if not path.is_absolute():
        path = urdf.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"mesh not found for {filename!r}")
    return path


def _run_alpha_wrap(args: argparse.Namespace, source: Path, output: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(args.alpha_wrap_tool),
            str(source),
            str(output),
            f"{args.relative_alpha:g}",
            f"{args.relative_offset:g}",
            f"{args.max_offset:g}",
            f"{args.max_alpha:g}",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    metadata = {"source": str(source), "output": str(output)}
    for line in completed.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            metadata[parts[0]] = parts[1]
    return metadata


def _validate_contains(
    source: Path,
    wrapped: Path,
    sample_count: int,
    tolerance_m: float,
) -> None:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "containment validation requires trimesh; install it before generating "
            "a collision artifact"
        ) from exc

    source_mesh = trimesh.load_mesh(source, force="mesh")
    wrapped_mesh = trimesh.load_mesh(wrapped, force="mesh")
    if not wrapped_mesh.is_watertight:
        raise RuntimeError(f"wrapped mesh is not watertight: {wrapped}")
    samples = source_mesh.vertices
    if len(source_mesh.faces) and sample_count > 0:
        surface, _ = trimesh.sample.sample_surface(source_mesh, sample_count)
        samples = __import__("numpy").vstack([samples, surface])
    try:
        inside = wrapped_mesh.contains(samples)
    except BaseException as exc:
        raise RuntimeError(f"failed containment query for {wrapped}: {exc}") from exc
    if not bool(inside.all()):
        missed = int((~inside).sum())
        raise RuntimeError(
            f"{wrapped} does not contain {missed} sampled source points "
            f"within tolerance {tolerance_m:g} m"
        )


if __name__ == "__main__":
    raise SystemExit(main())
