# Gazebo fault campaign

Runs the real GATO figure-eight planner and TIAGo controller with experiment-only
fault adapters. No fault hooks are imported by ordinary experiments. The campaign
is for simulation; it is not a hardware safety certification.

## Run

On the simulator host, with the PAL simulator and GATO containers already running,
copy this directory to the host and run:

```bash
python3 c3po.py --repo-host /home/felix/GATO --output-name fault_campaign_new
```

Use a new output name each time. The defaults select `gato-tiago-container` and
`tiago-pro-12-dev-felix`; override the corresponding command-line options if needed.
The host checkout must be mounted as `/workspace` in GATO and readable at its
host path inside PAL (the university container mounts `/home/felix` read-only).
Docker access and the built `bsqpN64_tiago_right` CUDA extension are required.

This command moves the **simulated** right arm. It observes each trial, then
performs separately logged assisted restoration before continuing. Do not run it
alongside another arm-control experiment. It does not recreate the simulator or
stop unrelated processes. Process-death injections target only the campaign's
own worker/controller/safety processes.

The runner requires Cyclone DDS, the expected domain, a loopback-only XML with
only `localhost` as a peer, advancing `/clock` from Gazebo, joint feedback, and
named controller observations. Domain 91 is the standalone runner default; the
host orchestrator explicitly selects the existing domain-1 simulator.

Select a subset with `--cases baseline planner_stall feedback_drop baseline`.
`--horizon-fault-duration 0.1` corrupts submissions for a 100 ms window instead
of one outgoing horizon. Keep these protocols separate: a one-shot corrupted
plan may be superseded in the latest-plan queue before consumption. The current
harness records anomalous-horizon consumption and controller error statuses.
Use `python run.py list` to list the implemented faults. Each trial tracks a
six-second prefix of the existing 24-second figure-eight with N=64, dt=0.008,
125 Hz command execution, clamping enabled, and the normal collision monitor.
Faults are injected two wall-clock seconds after confirmed effort activation.
The arm is restored over four seconds before each trial.

## Evidence

Results are written to `example_artifacts/<output-name>/` in the host checkout:

- `REPORT.md` and `results.json`: summaries of recorded trials.
- Per trial: `result.json`, `trial.json`, raw `observer.jsonl`,
  `controllers-pal.jsonl`, per-process event traces, and worker/setup/recovery logs.
- Normal tracking output is retained when the experiment reaches its output
  phase. An exception can prevent those files; independent observations survive.

Controller states are observed **inside PAL** using its installed message
definitions. On the measured c3po setup, PAL's `ControllerState` contains fields
absent in stock Humble; decoding its service from GATO returned empty names and
states. Empty/unavailable observations must be classified as inconclusive.

Restoration means the position and gravity-compensation controllers are observed
active and the GATO effort controller inactive. It does not mean a return to the
starting pose. Reported motion covers the first two wall-clock seconds after
injection, or the equivalent point in the baseline. The observer continues after
worker exit, and final settling uses a 0.05 rad/s threshold over at least 0.45 s
of the final half-second window, together with a position span below 0.01 rad.
Missing observations are not zero motion. The original v1/v2 trial reports used
only reported velocity for their settling field; the combined impact report
therefore does not use that field to establish a smooth stop. The stricter
position criterion was added after observing handover discontinuities.

Compare repeated baselines and repeats of each fault before drawing reliability
conclusions. One observed recovery does not guarantee bounded recovery latency.
Reference/model faults, real driver failures, host outages, and physical-network
packet loss are outside this initial implemented campaign.

## Local software checks

With NumPy, Pinocchio, and pytest installed:

```bash
python -m pytest -q tests/python/test_tiago_fault_campaign.py
```

These execute production validation and the controller loop with a fake ROS
client. They characterize command rejection and cleanup calls, not Gazebo motion.
Tests deliberately preserve existing failure behavior, including the fact that
publication failure during zero-effort cleanup can prevent controller restoration.
