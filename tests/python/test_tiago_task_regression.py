"""Opt-in task outcomes through the production TIAGo receding-horizon controller."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("tiago_task_worker.py")


@pytest.mark.cuda
@pytest.mark.task_regression
@pytest.mark.parametrize("case", ["short_reach", "long_reach", "figure8_tracking"])
def test_task_completes_with_physical_limits(case, tmp_path):
    if os.environ.get("GATO_RUN_TIAGO_TASK_TESTS") != "1":
        pytest.skip("set GATO_RUN_TIAGO_TASK_TESTS=1 to run TIAGo task regressions")

    artifact_root = Path(os.environ.get("GATO_TEST_ARTIFACT_DIR", str(tmp_path)))
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(tempfile.mkdtemp(prefix=f"{case}_", dir=artifact_root))
    command = [sys.executable, str(WORKER), "--case", case,
               "--artifact-dir", str(artifact_dir.resolve())]
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                timeout=600, check=False)
    except subprocess.TimeoutExpired as error:
        for name, output in (("stdout", error.stdout), ("stderr", error.stderr)):
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            (artifact_dir / f"{name}.txt").write_text(output or "", encoding="utf-8")
        pytest.fail(f"TIAGo task worker timed out; artifacts: {artifact_dir}")
    (artifact_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (artifact_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    summary_path = artifact_dir / "summary.json"
    assert result.returncode == 0, (
        f"TIAGo {case} failed (exit {result.returncode}); artifacts: {artifact_dir}\n"
        f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
    )
    assert summary_path.is_file(), f"Worker omitted task summary: {artifact_dir}"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["passed"], f"Failed task gates: {summary}; artifacts: {artifact_dir}"
