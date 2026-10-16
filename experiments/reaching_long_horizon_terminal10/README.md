# N256 reaching with tenfold terminal weight on RTX 5060

This is a new, simulation-only cohort with **30 cases**: the original starts and saved first baseline trap states of the 15 trapped goals, interleaved by goal in the frozen input order. It asks whether a 7.65 s prediction window and terminal position weight 800 help the flat-penalty controller escape these traps. The two start types are reported separately. Trap starts retain all seven saved velocities; they begin a fresh cold solver session at local time zero. They are not continuations of the original solver's internal state.

The controller replans every **10 ms** with **N256**, **30 ms knots**, **SQP cap 1000**, **PCG cap 120**, and **30 s maximum simulated episode time**. The first control is held across the original flat-worker RK4 prefix of **3+3+3+1 ms**. Mode A leaves native solve and Python wall time unpaid. Natural zero-PCG stopping remains unchanged and is a practical heuristic; there is no KKT gate. The flat joint-position penalty stays in mode 1 for the whole episode. There are no IK candidates, posture references, trap triggers, route seeds, orientation costs or final approach.

The other frozen costs are q=4, qd=.05, u=.00001, q_lim=.02125, vel_lim=1.275, ctrl_lim=.02, mu=10 and rho=.01. The compiled penalty bounds use a .20 rad joint inset, .80 speed fraction and .85 torque fraction. Arrival is position error <=50 mm and joint-speed L1 <=1 rad/s at a feedback boundary, without orientation or dwell. Raw sampled URDF position, speed and torque compliance is recorded separately, including the earliest strict breach category/joint/time. Historical physical-pass tolerances remain separate from strict breach detection.

Every case uses a new process/session, repeated **full actual 14-state** and zero-control seed, and one initial dual/rho reset followed by retention and the existing fractional warm shift. Goal references are the saved original six-component end-effector references retiled over 256 knots. No saved goal posture is loaded. No solver arithmetic or stopping changes are included. The only native source addition is the previously authorized conditional PCG shared-memory launch opt-in, cherry-picked as `6d037790cfdbeefa6a13733a7ea7da4eb83c969f`; normal-fitting launches are unchanged.

One newly parameterized **8 s canonical** runs before both start types. It must complete without errors, arrive under the rough gate, pass physical checks, and meet the saved geometry/motion checks. Any canonical failure holds all 30 scientific cases. A shared case setup failure also holds subsequent GPU work through a private build receipt. Independent execution failures and scientific negatives are retained; there is no automatic retry, budget reduction or tuning.

## Launch

Root's handoff supplies the exact reviewed published commit and pull/start command. Inside the existing RTX 5060 CUDA container, the absolute-path form is:

```bash
git fetch https://github.com/felixmueh/GATO.git refs/heads/experiment/reaching-long-horizon-terminal10-20261016 && git worktree add --detach /workspace/GATO-reaching-long-horizon-terminal10-run FETCH_HEAD && python3 /workspace/GATO-reaching-long-horizon-terminal10-run/experiments/reaching_long_horizon_terminal10/launch.py
```

Use a fresh destination directory. The launcher uses `/opt/gato-venv/bin/python3` when present, otherwise the current Python. It builds one private `bsqpN256_tiago_right` module for CUDA `120-real` with existing joint-cost mode support and the explicit bounds above. It uses the copied reviewed runner/collector, one explicit plan, and sequential GPU work. No remote command, Gazebo, robot, package installation, power-mode or clock change is involved.

All source/build/cache/temp/raw artifacts stay below the selected checkout. Results are under `_expr_results/reaching-long-horizon-terminal10-20261016/`, with a separate archive under `_expr_results/reaching-long-horizon-terminal10-20261016-backup/results.tar.gz`. Dense states, velocities, applied torques, timestamps, feedback metrics, native typed work, initial seeds, execution errors and invalid plans are retained. Routine returned horizons are omitted. Progress and log paths print per case. Native and wall durations are in each case's metrics and receipts. The interleaved first two cases give an early runtime comparison for one original/trap pair.

Runtime at N256/SQP1000 is **not yet qualified**. The full budget permits 90,000 feedback solves across the 30 cases, each with up to 1000 SQP iterations; natural early stopping can greatly reduce this. The prior N256 one-SQP timing around 1.958 ms is not a duration estimate or qualification for this controller. Allow several GB for dense records and work arrays, plus the separate archive. There is no automatic truncation for the deadline. Ctrl-C preserves the interrupted attempt and collects/archives saved raw data after workers return; the GPU is released when the processes exit.

To resume, invoke the same worktree's launcher. Completed cases are reused, failed attempts remain recorded, and an explicit `--retry-failed` is needed to request new attempts. A shared setup hold requires review before dependent work can resume. Saved-data collection alone is available with `--collect-only`.

## CPU preparation checks

```bash
/opt/gato-venv/bin/python experiments/reaching_long_horizon_terminal10/check_worker_cpu.py
python3 experiments/reaching_long_horizon_terminal10/launch.py --dry-run
```

These checks do not import a native solver, build CUDA code or run GPU kernels. Eight bounded contracts cover all 30 saved-state/reference inputs, N256 packed shapes, nonzero-velocity cold seeds, fractional warm shift and 10 ms RK4 subdivision, arrival/timeout/failure retention, typed native work, strict first breaches, and the canonical gate. The plan has one build, one shared canonical and 30 scientific cases. `prepare_inputs.py` is the development freeze step, not an extra launch experiment; it does not modify the saved numerical task inputs.
