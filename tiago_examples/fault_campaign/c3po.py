#!/usr/bin/env python3
"""Host-side campaign orchestration for an explicitly selected running simulator.

Uses only named containers, observes before/after injection, and restores the
simulated arm between trials. It does not recreate containers or change DDS
files. Re-run with a new output directory; evidence is never overwritten.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo-host", type=Path, required=True)
parser.add_argument("--output-name", default="fault_campaign_20260921")
parser.add_argument("--gato-container", default="gato-tiago-container")
parser.add_argument("--pal-container", default="tiago-pro-12-dev-felix")
parser.add_argument("--horizon-fault-duration", type=float, default=0.0)
parser.add_argument("--cases", nargs="+", default=["baseline", "trajectory_nan", "trajectory_shape", "trajectory_dt_zero", "trajectory_expired", "trajectory_future", "trajectory_start_nan", "torque_spike", "planner_exception", "planner_stall", "planner_exit", "feedback_drop", "feedback_nan", "feedback_frozen", "feedback_stamp_backwards", "safety_exit", "controller_stall", "controller_exit", "publish_exception", "restore_exception", "baseline"])
args = parser.parse_args()
if Path(args.output_name).name != args.output_name:
    parser.error("--output-name must be a single directory name")
root = args.repo_host.resolve() / "example_artifacts" / args.output_name
root.mkdir(parents=True, exist_ok=False)
container_root = Path("/workspace/example_artifacts") / args.output_name
tool_dir = "/tmp/gato-fault-20260921"
for container in (args.gato_container, args.pal_container):
    subprocess.run(["docker", "cp", str(HERE) + "/.", f"{container}:{tool_dir}"], check=True)
# Record environment evidence without dumping secrets from container configs.
environment = {}
for container in (args.gato_container, args.pal_container):
    environment[container] = subprocess.check_output(["docker", "inspect", "--format", "{{.Config.Image}} {{.Image}} network={{.HostConfig.NetworkMode}}", container], text=True).strip()
(root / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")


def gato(command, output=None):
    shell = "source /opt/ros/humble/setup.bash; source tiago_tools/ros_simulation.sh; " + command
    return subprocess.run(["docker", "exec", "-w", "/workspace", "-e", "GATO_FAULT_REPO=/workspace", args.gato_container, "bash", "-c", shell], stdout=output, stderr=subprocess.STDOUT, timeout=115)


try:
    for index, case in enumerate(args.cases):
        name = f"{index:02d}_{case}"
        directory = root / name
        directory.mkdir()
        inside = container_root / name
        print(f"START {name}", flush=True)
        with (directory / "setup.log").open("w") as log:
            recovery = gato("/opt/gato-venv/bin/python tiago_tools/tiago_restore_default.py --duration 4 --timeout 12", log)
        if recovery.returncode:
            raise RuntimeError(f"Initial-state restoration failed before {name}; stopping campaign")
        stop_file = directory / "observer-stop"
        with (directory / "controllers-pal.jsonl").open("w") as controller_log:
            shell = f"source /opt/pal/alum/setup.bash; export ROS_DOMAIN_ID=1; python3 -u {tool_dir}/controllers.py --stop-file {shlex.quote(str(stop_file))}"
            observer = subprocess.Popen(["docker", "exec", args.pal_container, "bash", "-c", shell], stdout=controller_log, stderr=subprocess.PIPE)
            try:
                with (directory / "runner.log").open("w") as log:
                    result = gato(f"/opt/gato-venv/bin/python {tool_dir}/run.py trial --domain 1 --case {shlex.quote(case)} --output {shlex.quote(str(inside))} --controller-log {shlex.quote(str(inside / 'controllers-pal.jsonl'))} --horizon-fault-duration {args.horizon_fault_duration}", log)
            finally:
                stop_file.touch()
                try:
                    _, errors = observer.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    observer.terminate()
                    _, errors = observer.communicate(timeout=5)
                (directory / "controller-observer.log").write_bytes(errors or b"")
        print(f"DONE {name} exit={result.returncode}", flush=True)
        # This is external, assisted recovery after the observation window.
        with (directory / "recovery.log").open("w") as log:
            recovery = gato("/opt/gato-venv/bin/python tiago_tools/tiago_restore_default.py --duration 4 --timeout 12", log)
        (directory / "assisted-recovery.json").write_text(json.dumps({"wall": time.monotonic(), "exit": recovery.returncode}) + "\n")
        if recovery.returncode:
            raise RuntimeError(f"Assisted recovery failed after {name}; stopping campaign")
        if result.returncode:
            raise RuntimeError(f"Trial {name} was inconclusive or baseline failed; inspect evidence before continuing")
finally:
    gato(f"/opt/gato-venv/bin/python {tool_dir}/run.py report --output {shlex.quote(str(container_root))}")
