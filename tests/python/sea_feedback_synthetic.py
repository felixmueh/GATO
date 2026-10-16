"""Reproducible CPU-only SEA estimator noise/rate examples; no ROS transport.

Run from the checkout root:
  PYTHONPATH=tiago_src:python python tests/python/sea_feedback_synthetic.py --output DIR

A 125 Hz caller takes only the last message that arrived since its previous tick.
The generated 1000 Hz stream therefore supplies at most 125 callbacks per second.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from gato_tiago.ros_tiago import RIGHT_ARM_JOINTS, SeaAbsoluteVelocityEstimator


SEED = 20261016
DURATION_SEC = 8.0
CONSUMER_HZ = 125.0


def delivery_schedule(source_hz, *, jitter=False, duration_sec=DURATION_SEC, seed=SEED):
    rng = np.random.default_rng(seed)
    source_times = [0.]
    while True:
        dt = (rng.uniform(.65, 1.35) if jitter else 1.) / source_hz
        next_time = round(source_times[-1] + dt, 9)
        if next_time > duration_sec:
            break
        source_times.append(next_time)
    source_times = np.round(np.asarray(source_times), 9)
    arrivals = source_times.copy()
    keep = np.ones(len(source_times), dtype=bool)
    if jitter:
        arrivals += rng.uniform(0., .004, len(arrivals))
        # Modest source losses plus two short bursts; no retries/outcome selection.
        keep = rng.random(len(arrivals)) >= .08
        keep &= ~(((source_times >= 2.) & (source_times < 2.035))
                  | ((source_times >= 5.) & (source_times < 5.035)))
        arrivals[0], keep[0] = 0., True
    event_indices = np.flatnonzero(keep)
    event_indices = event_indices[np.argsort(arrivals[event_indices], kind="stable")]
    ticks = np.arange(int(duration_sec * CONSUMER_HZ) + 1) / CONSUMER_HZ
    selected = np.full(len(ticks), -1, dtype=int)
    previous_end = 0
    for tick_index, tick in enumerate(ticks):
        end = int(np.searchsorted(arrivals[event_indices], tick, side="right"))
        if end > previous_end:
            selected[tick_index] = event_indices[end - 1]
        previous_end = end
    return source_times, ticks, selected


def true_state(t, motion):
    if motion == "stationary":
        return np.zeros(7), np.zeros(7)
    phase = np.arange(7) * .2
    amplitude = np.linspace(.10, .15, 7)
    omega = 2 * np.pi * .4
    return amplitude * np.sin(omega * t + phase), amplitude * omega * np.cos(omega * t + phase)


def sea_message(t, q):
    stamp = 100. + t
    sec = int(stamp)
    return NS(name=list(RIGHT_ARM_JOINTS), absolute_position=np.asarray(q).tolist(), position=[],
              header=NS(stamp=NS(sec=sec, nanosec=round((stamp - sec) * 1e9))))


def run_case(source_hz, noise_std_rad, motion, *, jitter=False, seed=SEED):
    source_times, ticks, selected = delivery_schedule(source_hz, jitter=jitter, seed=seed)
    rng = np.random.default_rng(seed + 1)
    true_q = np.asarray([true_state(t, motion)[0] for t in source_times])
    positions = true_q + rng.normal(0., noise_std_rad, true_q.shape)
    estimator = SeaAbsoluteVelocityEstimator(tau_sec=.01, stale_timeout_sec=.1)
    baseline = SeaAbsoluteVelocityEstimator(tau_sec=0., stale_timeout_sec=.1)
    rows, delivered_times, delivered_receipts = [], [], []
    unavailable = 0
    for tick, index in zip(ticks, selected):
        if index < 0:
            continue  # Polling a held sample does not feed a new estimator observation.
        t = source_times[index]
        msg = sea_message(t, positions[index])
        result = estimator.update(msg, 1. + tick)
        raw = baseline.update(msg, 1. + tick)
        delivered_times.append(float(t))
        delivered_receipts.append(float(tick))
        if result is None:
            unavailable += 1
            continue
        q, qd, _ = result
        expected_q, expected_qd = true_state(t, motion)
        rows.append(dict(t=float(t), received=float(tick), q=q, qd=qd,
                         true_q=expected_q, true_qd=expected_qd, raw_qd=raw[1],
                         observed_q=positions[index].copy()))
    q_error = np.asarray([row["q"] - row["true_q"] for row in rows])
    qd = np.asarray([row["qd"] for row in rows])
    qd_error = np.asarray([row["qd"] - row["true_qd"] for row in rows])
    raw_error = np.asarray([row["raw_qd"] - row["true_qd"] for row in rows])
    startup = np.asarray([row["received"] - rows[0]["received"] <= .1 for row in rows])
    delivered_times = np.asarray(delivered_times)
    delivered_receipts = np.asarray(delivered_receipts)
    metrics = dict(
        motion=motion, seed=seed, source_hz=source_hz, consumer_hz=CONSUMER_HZ,
        delivery="interval_jitter_delay_and_losses" if jitter else "regular",
        noise_std_rad=noise_std_rad, duration_sec=DURATION_SEC, filter_tau_sec=.01,
        generated_samples=len(source_times), delivered_callbacks=len(delivered_times),
        valid_estimates=len(rows), estimator_unavailable_callbacks=unavailable,
        delivered_callback_rate_hz=(len(delivered_receipts) - 1)
            / float(delivered_receipts[-1] - delivered_receipts[0]),
        estimate_rate_hz=(len(rows) - 1) / (rows[-1]["received"] - rows[0]["received"]),
        max_delivered_source_gap_sec=float(np.max(np.diff(delivered_times))),
        max_callback_gap_sec=float(np.max(np.diff(delivered_receipts))),
        q_matches_input_exactly=all(np.array_equal(row["q"], row["observed_q"]) for row in rows),
        q_raw_error_rms_rad=float(np.sqrt(np.mean(q_error ** 2))),
        qd_error_rms_rad_s=float(np.sqrt(np.mean(qd_error ** 2))),
        qd_rms_rad_s=float(np.sqrt(np.mean(qd ** 2))),
        raw_derivative_error_rms_rad_s=float(np.sqrt(np.mean(raw_error ** 2))),
        qd_error_to_raw_derivative_ratio=float(np.sqrt(np.mean(qd_error ** 2) / np.mean(raw_error ** 2))),
        first_estimate_max_abs_rad_s=float(np.max(np.abs(qd[0]))),
        startup_100ms_max_abs_rad_s=float(np.max(np.abs(qd[startup]))),
        all_estimates_max_abs_rad_s=float(np.max(np.abs(qd))),
        max_speed_l1_rad_s=float(np.max(np.sum(np.abs(qd), axis=1))),
        stationary_reset_speed_l1_le_0p1_share=(float(np.mean(np.sum(np.abs(qd), axis=1) <= .1))
                                              if motion == "stationary" else None),
    )
    return metrics, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for motion in ("stationary", "smooth"):
        for std in (.0001, .001):
            for rate in (1000., 125., 50.):
                for jitter in (False, True):
                    metrics, _ = run_case(rate, std, motion, jitter=jitter)
                    cases.append(metrics)
    result = dict(
        description="Synthetic estimator callbacks only; no ROS, transport, closed-loop or robot qualification.",
        consumer="125Hz latest-arrival sampling; messages overwritten between ticks are not differentiated",
        noise="Independent seeded Gaussian position noise per joint and generated sample",
        filtering="q remains raw; default velocity tau=.01 seconds unchanged",
        smooth_motion="q_j(t)=A_j*sin(2*pi*.4*t+.2*j), A_j linearly .10 to .15 rad, j=0..6",
        evaluation="All valid estimates retained including startup; no burn-in or retries",
        reset_share="Descriptive speed-only share, not full reset-arrival/task/physical acceptance",
        scenarios=cases,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("motion noise_mrad source_hz delivery actual_hz q_rms_rad qd_err raw_err ratio reset_share")
    for row in cases:
        print(f'{row["motion"]:10s} {row["noise_std_rad"] * 1000:4.1f} {row["source_hz"]:6.0f} '
              f'{row["delivery"]:32s} {row["delivered_callback_rate_hz"]:6.1f} '
              f'{row["q_raw_error_rms_rad"]:.6f} {row["qd_error_rms_rad_s"]:.5f} '
              f'{row["raw_derivative_error_rms_rad_s"]:.5f} '
              f'{row["qd_error_to_raw_derivative_ratio"]:.3f} '
              f'{row["stationary_reset_speed_l1_le_0p1_share"]}')
    print(f"Saved {len(cases)} fixed-seed scenarios to {args.output / 'metrics.json'}")


if __name__ == "__main__":
    main()
