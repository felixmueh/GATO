#!/usr/bin/env python3
"""Passively record SEA feedback and its velocity estimate; never command motion."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20., help="Recording duration [s]")
    parser.add_argument("--consumer-hz", type=float, default=125.)
    parser.add_argument("--velocity-filter-tau", type=float, default=.01, help="Time constant [s]")
    parser.add_argument("--topic", default="/arm_right_sea_state_broadcaster_controller/sea_joint_states")
    parser.add_argument("--output", type=Path, help="New JSONL file; default is a UTC-stamped raw path")
    args = parser.parse_args()
    import numpy as np
    if not all(np.isfinite(x) and x > 0 for x in (args.duration, args.consumer_hz)):
        parser.error("duration and consumer-hz must be finite and positive")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tiago_src"))
    from gato_tiago.ros_tiago import SeaAbsoluteVelocityEstimator
    estimator = SeaAbsoluteVelocityEstimator(tau_sec=args.velocity_filter_tau)
    # Only message subscription/serialization: no arm client, publishers or controller RPCs.
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sea_state_msgs.msg import SeaState
    output = args.output or root / "_expr_results/physical-arm/passive-sea" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    latest = None
    receipts, source_stamps, estimates = [], [], []
    consumed = 0
    def receive(msg):
        nonlocal latest
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        received = time.monotonic()
        receipts.append(received)
        source_stamps.append(stamp)
        latest = msg, received, stamp, len(receipts)
    rclpy.init()
    node = rclpy.create_node("gato_passive_sea_observer", start_parameter_services=False,
                             enable_rosout=False)
    subscription = node.create_subscription(SeaState, args.topic, receive,
        QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    started = time.monotonic()
    next_tick = started
    try:
        with output.open("x") as stream:
            while rclpy.ok() and time.monotonic() - started < args.duration:
                remaining = started + args.duration - time.monotonic()
                rclpy.spin_once(node, timeout_sec=max(0., min(next_tick - time.monotonic(), remaining)))
                now = time.monotonic()
                if now < next_tick:
                    continue
                next_tick = now + 1. / args.consumer_hz
                if latest is None or latest[3] == consumed:
                    continue
                msg, received, stamp, consumed = latest
                sample = estimator.update(msg, received)
                if sample is not None:
                    estimates.append((sample[0], sample[1]))
                stream.write(json.dumps(dict(
                    source_stamp_sec=stamp, received_monotonic_sec=received,
                    consumed_monotonic_sec=now, age_at_consumption_ms=(now - received) * 1000.,
                    name=list(msg.name), position=list(msg.position),
                    absolute_position=list(msg.absolute_position),
                    q=None if sample is None else sample[0].tolist(),
                    qd_estimated=None if sample is None else sample[1].tolist())) + "\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    elapsed = time.monotonic() - started
    def intervals(values):
        gaps = np.diff(values) * 1000.
        return None if not len(gaps) else dict(median_ms=float(np.median(gaps)),
            p95_ms=float(np.percentile(gaps, 95)), max_ms=float(np.max(gaps)))
    summary = dict(topic=args.topic, output=str(output), duration_s=elapsed,
        requested_consumer_hz=args.consumer_hz,
        callbacks=len(receipts), callbacks_hz=len(receipts) / elapsed,
        estimates=len(estimates), estimates_hz=len(estimates) / elapsed,
        receive_intervals=intervals(receipts), source_intervals=intervals(source_stamps),
        velocity_filter_tau_s=args.velocity_filter_tau,
        scope="Passive SEA-only diagnostic; no commands, body-state/safety checks or controller execution. JSONL contains consumed samples; callback/interval summaries cover all received callbacks. Observer delivery rates do not qualify production delivery. Noise statistics require an already stationary arm.")
    if estimates:
        q, qd = (np.array(values) for values in zip(*estimates))
        summary.update(position_std_rad=np.std(q, axis=0).tolist(),
            estimated_velocity_rms_rad_s=np.sqrt(np.mean(qd ** 2, axis=0)).tolist(),
            estimated_speed_l1_below_point1_share=float(np.mean(np.sum(np.abs(qd), axis=1) <= .1)))
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if estimates else 1


if __name__ == "__main__":
    raise SystemExit(main())
