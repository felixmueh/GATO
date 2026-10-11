#!/usr/bin/env python3
"""Run one independently observed Gazebo fault trial, or summarize its evidence.

Use only an explicitly selected local simulator in addition to these ROS checks.
This runner never switches controllers or sends torque itself. Only the owned
experiment worker does so; emergency termination is reported as intervention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("GATO_FAULT_REPO", HERE.parents[1])).resolve()
sys.path[:0] = [str(HERE), str(REPO / "tiago_src"), str(REPO / "python")]
from hooks import CASES, event, install

# multiprocessing's spawn imports this module before invoking production child
# functions. The observer has no specification and thus never installs hooks.
if os.environ.get("GATO_FAULT_SPEC"):
    install(json.loads(os.environ["GATO_FAULT_SPEC"]))


def read_events(directory):
    rows = []
    for path in Path(directory).glob("events-*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Abrupt death can leave a partial final line. Preserve the
                # raw file; missing evidence must not be inferred as success.
                continue
    return sorted(rows, key=lambda row: row["wall"])


def read_controller_log(path):
    rows = []
    if path is None or not Path(path).exists():
        return rows
    for line in Path(path).read_text().splitlines():
        try:
            row = json.loads(line)
            if row.get("controllers") and all(c.get("name") and c.get("state") for c in row["controllers"]):
                rows.append(row)
        except json.JSONDecodeError:
            continue
    return rows


def summarize(directory):
    directory = Path(directory)
    meta = json.loads((directory / "trial.json").read_text())
    events = read_events(directory)
    states = []
    for line in (directory / "observer.jsonl").read_text().splitlines():
        try:
            states.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    injections = [e for e in events if e["kind"] == "injected"]
    armed = [e for e in events if e["kind"] == "effort_activated"]
    t0 = injections[0]["wall"] if injections else None
    if meta["case"] == "baseline" and armed:
        t0 = armed[0]["wall"] + meta.get("inject_after_wall_sec", 2.0)
    post_all = [s for s in states if t0 is not None and s["wall"] >= t0]
    post = [s for s in post_all if s["wall"] <= t0 + 2.0]
    before = [s for s in states if t0 is not None and s["wall"] < t0]
    start = before[-1] if before else (post[0] if post else None)
    requests = [e for e in events if e["kind"] == "restore_requested" and t0 is not None and e["wall"] >= t0]
    snapshots = read_controller_log(directory / "controllers-pal.jsonl") or meta.get("controller_snapshots", [])
    snapshots = [s for s in snapshots if s.get("controllers") and all(c.get("name") and c.get("state") for c in s["controllers"])]

    def restored(snapshot):
        active = {c["name"] for c in snapshot["controllers"] if c["state"] == "active"}
        return {"arm_right_controller", "arm_right_gravity_compensation_controller"} <= active and "gato_arm_effort_forward_runtime" not in active

    confirmed = [s for s in snapshots if t0 is not None and s["wall"] >= t0 and restored(s)]
    log = (directory / "worker.log").read_text(errors="replace")
    error_lines = [line.strip() for line in log.splitlines() if any(x in line for x in ("RuntimeError:", "ValueError:", "TimeoutError:", "clamping torque", "OverflowError:"))]
    publications = [e for e in events if e["kind"] == "effort_published" and t0 is not None and e["wall"] >= t0]
    consumed = [e for e in events if e["kind"] == "anomalous_horizon_consumed" and t0 is not None and e["wall"] >= t0]
    detected = [e for e in events if e["kind"] == "controller_status" and e.get("mode", "").startswith("ERROR") and t0 is not None and e["wall"] >= t0]
    gaps = np.diff([s["wall"] for s in post]) if len(post) > 1 else []
    displacement = max((float(np.max(np.abs(np.array(s["q"]) - start["q"]))) for s in post), default=None) if start else None
    max_velocity = max((max(abs(v) for v in s["qd"]) for s in post), default=None)
    final = snapshots[-1] if snapshots else None
    # This only describes the final observed half second, not hardware safety.
    tail = [s for s in post_all if s["wall"] >= post_all[-1]["wall"] - 0.5] if post_all else []
    settled = None
    if len(tail) >= 2 and tail[-1]["wall"] - tail[0]["wall"] >= 0.45 and max(gaps, default=0) < 0.1:
        settled = (all(max(abs(v) for v in s["qd"]) < 0.05 for s in tail)
                   and float(np.max(np.ptp(np.asarray([s["q"] for s in tail]), axis=0))) < 0.01)
    result = {
        "case": meta["case"], "description": CASES[meta["case"]],
        "armed": bool(armed), "fault_injected": bool(injections),
        "worker_exit": meta.get("worker_exit"), "observer_error": meta.get("observer_error"),
        "supervisor_intervened": meta.get("supervisor_intervened", False),
        "post_fault_samples": len(post), "max_observer_gap_sec": float(max(gaps)) if len(gaps) else None,
        "motion_window_sec": 2.0,
        "post_fault_observation_sec": post_all[-1]["wall"] - t0 if post_all else None,
        "horizon_fault_duration_sec": meta.get("horizon_fault_duration_sec", 0.0),
        "anomalous_horizon_consumed": bool(consumed),
        "controller_error_latency_sec": detected[0]["wall"] - t0 if detected else None,
        "restore_request_latency_sec": requests[0]["wall"] - t0 if requests else None,
        "restore_confirmed_latency_sec": confirmed[0]["wall"] - t0 if confirmed else None,
        "final_default_controllers_active": restored(final) if final else None,
        "final_controller_snapshot_age_sec": meta.get("finished_wall", time.monotonic()) - final["wall"] if final else None,
        "max_post_fault_joint_displacement_rad": displacement,
        "max_post_fault_joint_speed_rad_s": max_velocity,
        "max_published_effort_after_fault_nm": max((e["max_abs"] for e in publications), default=None),
        "last_nonzero_publication_after_fault_sec": max((e["wall"] - t0 for e in publications if e["max_abs"] > 1e-9), default=None),
        "settled_final_half_second": settled,
        "errors": list(dict.fromkeys(error_lines)),
        "evidence": {"events": "events-*.jsonl", "states": "observer.jsonl", "worker": "worker.log", "metadata": "trial.json"},
    }
    if not armed or not snapshots or meta.get("observer_error") or (meta["case"] != "baseline" and (not injections or len(post) < 2)):
        result["outcome"] = "inconclusive"
    elif meta["case"] == "baseline":
        result["outcome"] = "baseline_completed" if meta.get("worker_exit") == 0 and result["final_default_controllers_active"] else "baseline_failed"
    elif meta.get("supervisor_intervened"):
        result["outcome"] = "supervisor_intervention"
    elif confirmed:
        result["outcome"] = "default_control_observed"
    else:
        result["outcome"] = "default_control_not_observed"
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def report(root):
    root = Path(root)
    results = [summarize(p.parent) for p in sorted(root.glob("*/trial.json"))]
    lines = ["# Gazebo fault campaign", "", "Measured trials; software faults are injected at the interfaces described below. This is not physical-robot safety validation.", "", "| Trial | Outcome | Restore observed (s) | Max joint displacement (rad) | Max speed (rad/s) |", "| --- | --- | ---: | ---: | ---: |"]
    paths = sorted(root.glob("*/trial.json"))
    def fmt(value):
        return "not observed" if value is None else f"{value:.4f}"
    for result, path in zip(results, paths):
        name = path.parent.name
        lines.append(f"| [{name}]({name}/result.json) | {result['outcome']} | {fmt(result['restore_confirmed_latency_sec'])} | {fmt(result['max_post_fault_joint_displacement_rad'])} | {fmt(result['max_post_fault_joint_speed_rad_s'])} |")
    lines += ["", "## Interpretation", "", "Restoration latency is the first independent controller-manager observation after injection; it includes polling delay. It is not the exact transition time. Joint displacement is relative to the last raw observation before injection and includes intended motion. Compare with the matched baseline. A final pose need not equal the starting pose.", "", "Observer data is the unmodified ROS joint-state stream. This is independent of injected controller callback corruption, but it is not independent of Gazebo's state publisher. The trial uses a fixed wall-time observation window after worker exit. Process exits and external interventions remain explicit. Missing data is not evidence of zero motion.", "", "Feedback suppression is a callback injection, not a DDS packet-loss experiment. Effort publication exceptions are Python failures, not measurements of a NIC outage. Abrupt exits skip Python cleanup. Horizon injection duration is recorded per trial: zero means one submitted horizon; positive values mean a repeated injection window. Production clamping remains enabled. Finite future-dated plans may be superseded by the following healthy horizon. A successful run does not prove such metadata is validated.", "", "The full figure-eight is 24 seconds; these trials observe a short prefix, with faults injected after two wall-clock seconds of active effort control. Timing and control quality on a shared GPU can vary. Single trials are observations, not reliability estimates.", "", "## Per-case evidence", ""]
    for result, path in zip(results, paths):
        name = path.parent.name
        lines += [f"### {name}", "", result["description"], "", f"Armed: {result['armed']}; injected: {result['fault_injected']}; worker exit: {result['worker_exit']}; final default controllers: {result['final_default_controllers_active']}; settled: {result['settled_final_half_second']}; supervisor intervention: {result['supervisor_intervened']}.", "", f"[Worker log]({name}/worker.log), [raw observations]({name}/observer.jsonl), [metadata]({name}/trial.json).", ""]
        lines += [f"- `{line}`" for line in result["errors"]]
        lines.append("")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    (root / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results


def worker(args):
    import importlib.util
    path = REPO / "tiago_examples" / "tiago_fig8_tracking.py"
    spec = importlib.util.spec_from_file_location("fault_fig8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(path), "--plant", "tiago_right", "--ros-tiago", "--ros-clamp-torque",
                "--N", "64", "--dt", "0.008", "--ros-target-hz", "125",
                "--ros-reset-duration", "4", "--sim-time", "6", "--no-gif",
                "--output-root", str(args.output / "tracking")]
    # Plotting after shutdown adds no fault evidence and can obscure exit time.
    module.save_renderings = lambda *a, **k: None
    module.run(module.parse_args())


def observe(args):
    if os.environ.get("ROS_DOMAIN_ID") != str(args.domain) or os.environ.get("RMW_IMPLEMENTATION") != "rmw_cyclonedds_cpp":
        raise RuntimeError("ROS domain must match --domain and middleware must be Cyclone DDS")
    uri = os.environ.get("CYCLONEDDS_URI", "").removeprefix("file://")
    xml = ET.parse(uri).getroot()
    interfaces = [e.get("name") for e in xml.findall(".//NetworkInterface")]
    peers = [e.get("Address") for e in xml.findall(".//Peer")]
    if interfaces != ["lo"] or peers != ["localhost"] or xml.findtext(".//AllowMulticast") != "false":
        raise RuntimeError("Fault trials require the repository's loopback-only DDS configuration")
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from rosgraph_msgs.msg import Clock
    from controller_manager_msgs.srv import ListControllers
    from gato_tiago.ros_tiago import RIGHT_ARM_JOINTS

    if (args.output / "trial.json").exists():
        raise FileExistsError("Refusing to overwrite an existing trial")
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {"case": args.case, "description": CASES[args.case], "inject_after_wall_sec": args.inject_after,
            "horizon_fault_duration_sec": args.horizon_fault_duration,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "controller_snapshots": [], "supervisor_intervened": False,
            "ros_environment": {k: os.environ.get(k) for k in ("ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "CYCLONEDDS_URI")},
            "settled_velocity_threshold_rad_s": 0.05, "settled_dwell_sec": 0.5}
    meta["settled_position_span_rad"] = 0.01
    try:
        meta["source_head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        meta["source_status"] = subprocess.check_output(["git", "-C", str(REPO), "status", "--short"], text=True)
    except subprocess.CalledProcessError:
        meta["source_head"] = "unavailable"
    modules = list((REPO / "python/bsqp").glob("bsqpN64_tiago_right*.so"))
    meta["solver_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in modules}
    meta["harness_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE / "run.py", HERE / "hooks.py")}
    rclpy.init()
    node = rclpy.create_node("gato_fault_observer")
    state_file = (args.output / "observer.jsonl").open("w")
    clock_values = []
    last_states = []

    def state_callback(msg):
        indexes = {name: i for i, name in enumerate(msg.name)}
        if any(n not in indexes for n in RIGHT_ARM_JOINTS):
            return
        ids = [indexes[n] for n in RIGHT_ARM_JOINTS]
        if any(i >= len(msg.position) or i >= len(msg.velocity) for i in ids):
            return
        record = {"wall": time.monotonic(), "sim": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                  "q": [msg.position[i] for i in ids], "qd": [msg.velocity[i] for i in ids]}
        if not np.isfinite(record["q"] + record["qd"]).all():
            raise RuntimeError("Non-finite raw observer state")
        state_file.write(json.dumps(record) + "\n")
        state_file.flush()
        last_states.append(record)
        del last_states[:-30]

    node.create_subscription(JointState, "/joint_states", state_callback, qos_profile_sensor_data)
    node.create_subscription(Clock, "/clock", lambda m: clock_values.append(m.clock.sec + m.clock.nanosec * 1e-9), qos_profile_sensor_data)
    client = node.create_client(ListControllers, "/controller_manager/list_controllers")
    process = None
    future = None
    next_query = 0.0
    started = time.monotonic()
    exited_at = None
    try:
        while time.monotonic() - started < 90:
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.monotonic()
            if future is not None and future.done():
                response = future.result()
                if response is not None:
                    meta["controller_snapshots"].append({"wall": now, "controllers": [{"name": c.name, "state": c.state} for c in response.controller]})
                future = None
            if args.controller_log is not None:
                meta["controller_snapshots"] = read_controller_log(args.controller_log)
            if args.controller_log is None and future is None and now >= next_query and client.service_is_ready():
                future = client.call_async(ListControllers.Request())
                next_query = now + 0.2
            if process is None:
                publishers = node.get_publishers_info_by_topic("/clock")
                valid_controllers = meta["controller_snapshots"] and any(c["name"] == "arm_right_controller" for c in meta["controller_snapshots"][-1]["controllers"])
                if len(last_states) >= 10 and len(clock_values) >= 2 and clock_values[-1] > clock_values[0] and any(p.node_name == "gazebo" for p in publishers) and valid_controllers:
                    meta["preflight"] = {"wall": now, "clock": clock_values[-1], "samples": len(last_states), "gazebo_clock_publisher": True}
                    env = os.environ.copy()
                    env["GATO_FAULT_SPEC"] = json.dumps({"case": args.case, "directory": str(args.output), "inject_after": args.inject_after, "horizon_fault_duration": args.horizon_fault_duration})
                    log = (args.output / "worker.log").open("w")
                    process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--output", str(args.output)], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    log.close()
                    meta["worker_pid"] = process.pid
                elif now - started > 25:
                    raise RuntimeError("Dedicated simulator preflight failed: require advancing /clock, raw joint states, and controller query")
            else:
                if process.poll() is not None and exited_at is None:
                    exited_at = now
                    meta["worker_exit"] = process.returncode
                if exited_at is not None and now - exited_at >= 2.0:
                    break
        else:
            meta["supervisor_intervened"] = True
            meta["observer_error"] = "90-second wall deadline"
    except BaseException as exc:
        meta["observer_error"] = str(exc)
        traceback.print_exc()
    finally:
        if process is not None:
            if process.poll() is None:
                meta["supervisor_intervened"] = True
            # Kill only the owned experiment process group, including an orphan
            # controller/safety worker after an abrupt planner exit. The simulator
            # is reset externally before any subsequent motion trial.
            try:
                meta["group_cleanup_wall"] = time.monotonic()
                os.killpg(process.pid, signal.SIGKILL)
                meta["group_cleanup_sent"] = True
            except ProcessLookupError:
                meta["group_cleanup_sent"] = False
            process.wait(timeout=5)
            meta.setdefault("worker_exit", process.returncode)
        meta["finished_wall"] = time.monotonic()
        meta["clock_first"] = clock_values[0] if clock_values else None
        meta["clock_last"] = clock_values[-1] if clock_values else None
        state_file.close()
        node.destroy_node()
        rclpy.shutdown()
        (args.output / "trial.json").write_text(json.dumps(meta, indent=2) + "\n")
        (args.output / "worker.log").touch(exist_ok=True)
    result = summarize(args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["outcome"] not in {"inconclusive", "baseline_failed"} else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("list", "trial", "worker", "report"))
    parser.add_argument("--case", choices=tuple(CASES), default="baseline")
    parser.add_argument("--output", type=Path, default=Path("example_artifacts/fault_campaign"))
    parser.add_argument("--inject-after", type=float, default=2.0)
    parser.add_argument("--domain", type=int, choices=(1, 91), default=91,
                        help="Use 1 only for an explicitly selected existing local simulator")
    parser.add_argument("--controller-log", type=Path)
    parser.add_argument("--horizon-fault-duration", type=float, default=0.0,
                        help="0 injects one horizon; positive values corrupt submissions over that many seconds")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.inject_after <= 0 or not np.isfinite(args.inject_after):
        parser.error("--inject-after must be positive and finite")
    if args.horizon_fault_duration < 0 or not np.isfinite(args.horizon_fault_duration):
        parser.error("--horizon-fault-duration must be nonnegative and finite")
    if args.mode == "list":
        print(json.dumps(CASES, indent=2))
        return 0
    if args.mode == "worker":
        worker(args)
        return 0
    if args.mode == "report":
        report(args.output)
        return 0
    return observe(args)


if __name__ == "__main__":
    raise SystemExit(main())
