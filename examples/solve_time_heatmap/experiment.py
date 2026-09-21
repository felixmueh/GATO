"""Shared cell worker for the solve-time matrix.

The solver, initialization, control policy and recorder are shared by both clocks.
Only the update interval, reference indexing and run budget differ.
"""
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback

from settings import (CONTROL_DT, DEFAULT_INIT_SOLVES, INTEGRATION_DT,
                      REFERENCE_DT, cost_rule, prediction_step)

ROOT = Path(__file__).resolve().parents[2]


def clean(value):
    """Produce strict JSON, preserving failed numeric observations as null."""
    import numpy as np
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class NumericInstability(RuntimeError):
    """A nonfinite plan or simulated state, retained as an experiment outcome."""


def run_instrumented(solver, model, x_start, reference, record, *,
                     protocol='wall-time', wall_time=10., sim_time=10.,
                     max_solve_ms=1000., initialization='single', max_init_solves=DEFAULT_INIT_SOLVES,
                     horizon_time=None):
    """Execute MPC with an excluded initialization phase and shared measurement path.

    Wall-time mode advances simulation by 10 ms per update, repeats one reference
    cycle, and checks a soft wall budget between complete updates. Reference mode
    preserves the historical capped clock and finite reference stopping rule.
    Zero disables the measured-solve cutoff. A slow sample is always retained.
    Initialization either performs one historical solve, or repeats the unchanged
    initial problem until every batch member stops natively within a solve limit.
    """
    import numpy as np
    import pinocchio as pin
    from protocol import rk4, periodic_reference, advance_plan

    if protocol not in ('wall-time', 'reference'):
        raise ValueError(f'Unknown protocol: {protocol}')
    if initialization not in ('single', 'native-stop'):
        raise ValueError(f'Unknown initialization: {initialization}')
    if not isinstance(max_init_solves, int) or max_init_solves < 1:
        raise ValueError('Initialization solve limit must be a positive integer')
    if (not np.isfinite([wall_time, sim_time, max_solve_ms]).all()
            or wall_time <= 0 or sim_time <= 0 or max_solve_ms < 0):
        raise ValueError('Budgets must be positive and solve cutoff nonnegative')
    dt, sim_dt = CONTROL_DT, INTEGRATION_DT
    n, batch = solver.N, solver.batch_size
    prediction_dt = prediction_step(n, horizon_time)
    if horizon_time is not None and (protocol != 'wall-time' or
                                    not np.isfinite(horizon_time) or horizon_time < dt):
        raise ValueError('Fixed prediction duration requires wall-time mode and at least 10 ms')
    nq, nv = model.nq, model.nv
    nx, nu = nq + nv, nv
    reference = np.asarray(reference).reshape(-1, 6)
    if (len(reference) < n and protocol == 'reference') or not len(reference):
        raise ValueError('Reference is too short')

    def reference_at(offset):
        if horizon_time is not None:
            return periodic_reference(reference, offset * dt + np.arange(n) * prediction_dt).reshape(-1)
        knots = ((offset + np.arange(n)) % len(reference) if protocol == 'wall-time'
                 else np.arange(offset, offset + n))
        return reference[knots].reshape(-1)

    data = model.createData()
    model.gravity.linear = np.array([0., 0., -9.81])
    x_curr = x_start.copy()
    q, dq = x_curr[:nq], x_curr[nq:nx]
    x_curr_batch = np.tile(x_curr, (batch, 1))
    ee_g = reference_at(0)
    ee_g_batch = np.tile(ee_g, (batch, 1))
    xu = np.zeros(n * (nx + nu) - nu)
    for i in range(n):
        xu[i * (nx + nu):i * (nx + nu) + nx] = x_start
    xu_batch = np.tile(xu, (batch, 1))
    solver.set_f_ext_B(np.zeros((batch, 6), dtype=np.float32))
    solver.reset_dual()
    record.begin_initialization(initialization, max_init_solves)
    initialization_status = 'incomplete'
    try:
        for _ in range(1 if initialization == 'single' else max_init_solves):
            solver.reset_rho()
            xu_batch, warmup_us = solver.solve(x_curr_batch, ee_g_batch, xu_batch)
            xu_best = xu_batch[0, :]
            record(solver, xu_best, x_curr, ee_g, warmup_us, 0., 0, None, None, True,
                   finite_batch=np.isfinite(xu_batch).all(axis=1))
            if not np.isfinite(xu_batch).all():
                raise NumericInstability('Nonfinite warm-up plan')
            if not np.isfinite(warmup_us) or warmup_us <= 0:
                raise RuntimeError('Invalid native initialization timing')
            if initialization == 'single':
                initialization_status = 'single_solve'
                break
            if record.warmup['all_native_stop']:
                initialization_status = 'ready'
                break
    finally:
        record.finish_initialization(initialization_status)
    if initialization_status == 'incomplete':
        return 0., 'initialization_incomplete'

    update_index = 0
    total_sim_time = accumulated_time = 0.
    previous_solve_seconds = dt
    record.begin_measurement()
    try:
        while (record.measured_elapsed() < wall_time if protocol == 'wall-time'
               else total_sim_time < sim_time):
            if horizon_time is not None:
                q, dq = advance_plan(model, data, q, dq, xu_best, n, prediction_dt, dt, sim_dt)
                if not np.isfinite(q).all() or not np.isfinite(dq).all():
                    record.failure_state = dict(timestamp=total_sim_time + dt,
                                                state=np.concatenate([q, dq]))
                    raise NumericInstability('Nonfinite simulated state')
            else:
                # Preserve the historical integer substeps and residual clock.
                timestep = dt if protocol == 'wall-time' else min(previous_solve_seconds, dt)
                nsteps = int(timestep / sim_dt)
                if protocol == 'reference' and timestep % sim_dt > 1e-5:
                    accumulated_time += timestep % sim_dt
                extra_step = accumulated_time >= sim_dt
                if extra_step:
                    accumulated_time -= sim_dt
                for step_index in range(nsteps + int(extra_step)):
                    offset = int(step_index / (dt / sim_dt))
                    u_idx = nx + (nx + nu) * min(offset, n - 1)
                    q, dq = rk4(model, data, q, dq, xu_best[u_idx:u_idx + nu], sim_dt)
                    total_sim_time += sim_dt
                    if not np.isfinite(q).all() or not np.isfinite(dq).all():
                        record.failure_state = dict(timestamp=total_sim_time,
                                                    state=np.concatenate([q, dq]))
                        raise NumericInstability('Nonfinite simulated state')
            update_index += 1
            if protocol == 'wall-time':
                total_sim_time = update_index * dt
                reference_offset = update_index
            else:
                reference_offset = int(total_sim_time / dt)
                # Keep the original extra factor of six for historical parity.
                if reference_offset >= len(reference) - 6 * n:
                    return total_sim_time, 'reference_exhausted'
            x_curr = np.concatenate([q, dq])
            x_curr_batch = np.tile(x_curr, (batch, 1))
            ee_g = reference_at(reference_offset)
            ee_g_batch[:, :] = ee_g
            xu_batch[:, :nx] = x_curr
            solver.reset_rho()
            xu_batch_new, solve_time_us = solver.solve(x_curr_batch, ee_g_batch, xu_batch)
            finite_batch = np.isfinite(xu_batch_new).all(axis=1)
            xu_best = xu_batch_new[0, :]
            xu_batch[:, :] = xu_best
            pin.forwardKinematics(model, data, q)
            ee_pos = data.oMi[6].translation.copy()
            distance = np.linalg.norm(ee_pos - (ee_g[:3] if horizon_time is not None else ee_g[6:9]))
            record(solver, xu_best, x_curr, ee_g, solve_time_us, total_sim_time,
                   reference_offset, ee_pos, distance, False, finite_batch=finite_batch)
            if not finite_batch.all() or not np.isfinite(distance):
                raise NumericInstability('Nonfinite plan or tracking distance')
            if not np.isfinite(solve_time_us) or solve_time_us <= 0:
                raise RuntimeError('Invalid native solve timing')
            previous_solve_seconds = solve_time_us / 1e6
            if max_solve_ms > 0 and solve_time_us / 1000. > max_solve_ms:
                return total_sim_time, 'slow_solve'
        return total_sim_time, 'completed'
    finally:
        record.finish_measurement()


class Recorder:
    """Record initialization separately from measured per-update outcomes."""

    def __init__(self, output, *, batch_size, max_sqp_iters, max_solve_ms,
                 wall_time, protocol, meta, save_plans=False):
        self.output, self.batch_size = output, batch_size
        self.max_sqp_iters, self.max_solve_ms = max_sqp_iters, max_solve_ms
        self.wall_time, self.protocol, self.meta = wall_time, protocol, meta
        self.save_plans = save_plans
        self.rows, self.plans, self.states, self.ee = [], [], [], []
        self.warmup = self.failure_state = None
        self.initialization_rows = []
        self.initialization_mode = self.initialization_status = None
        self.max_init_solves = None
        self.initialization_started = self.initialization_finished = None
        self.measurement_started = self.measurement_finished = None
        self.started = self.last_checkpoint = self.last_progress = time.monotonic()
        self.stream = (output / 'solves.jsonl').open('w', buffering=1)

    def begin_initialization(self, mode, max_solves):
        self.initialization_mode = mode
        self.max_init_solves = max_solves
        self.initialization_status = 'running'
        self.initialization_started = time.monotonic()
        self.initialization_finished = None

    def finish_initialization(self, status):
        self.initialization_finished = time.monotonic()
        self.initialization_status = status
        self.save_initialization()

    def initialization_summary(self):
        end = (self.initialization_finished if self.initialization_finished is not None
               else time.monotonic())
        return dict(mode=self.initialization_mode, status=self.initialization_status,
            call_count=len(self.initialization_rows), max_solves=self.max_init_solves,
            total_native_ms=sum(row['native_ms'] for row in self.initialization_rows),
            wall_time_s=(end - self.initialization_started
                         if self.initialization_started is not None else 0.),
            all_native_stop=(self.initialization_rows[-1]['all_native_stop']
                             if self.initialization_rows else False))

    def save_initialization(self):
        save_json(self.output / 'initialization.json',
                  self.initialization_summary() | dict(calls=self.initialization_rows))

    def begin_measurement(self):
        self.measurement_started = time.monotonic()
        self.measurement_finished = None

    def finish_measurement(self):
        self.measurement_finished = time.monotonic()

    def measured_elapsed(self):
        if self.measurement_started is None:
            return 0.
        end = self.measurement_finished if self.measurement_finished is not None else time.monotonic()
        return end - self.measurement_started

    def __call__(self, solver, plan, x0, reference, native_us, timestamp,
                 reference_offset, ee_pos, tracking, warmup, *, finite_batch):
        import numpy as np
        stats = solver.get_stats()
        iterations_batch = np.asarray(stats['sqp_iters'], dtype=int).reshape(-1)
        flags_batch = np.asarray(stats['kkt_converged'], dtype=bool).reshape(-1)
        if len(iterations_batch) != self.batch_size or len(flags_batch) != self.batch_size:
            raise RuntimeError('Native statistics do not cover every batch member')
        cap_batch = (iterations_batch >= self.max_sqp_iters) & ~flags_batch
        row = dict(index=-1 if warmup else len(self.rows), warmup=warmup,
            timestamp_s=timestamp, reference_offset=reference_offset,
            native_ms=native_us / 1000.,
            measured_wall_elapsed_s=None if warmup else self.measured_elapsed(),
            slow_solve=bool(not warmup and self.max_solve_ms > 0 and native_us / 1000. > self.max_solve_ms),
            sqp_iters=int(iterations_batch[0]), native_converged=bool(flags_batch[0]),
            sqp_iters_batch=iterations_batch.tolist(), native_converged_batch=flags_batch.tolist(),
            iteration_cap_hit_batch=cap_batch.tolist(), all_native_stop=bool(flags_batch.all()),
            any_cap=bool(cap_batch.any()), finite_plan_batch=np.asarray(finite_batch, dtype=bool).tolist(),
            iteration_cap_hit=bool(cap_batch[0]), tracking_error_m=tracking,
            finite_plan=bool(np.isfinite(plan).all()),
            initial_merit=float(np.asarray(stats.get('initial_merit', [np.nan])).reshape(-1)[0]),
            final_merit=float(np.asarray(stats.get('final_merit', [np.nan])).reshape(-1)[0]))
        if warmup:
            self.warmup = row
            self.initialization_rows.append(row)
            self.save_initialization()
            save_json(self.output / 'warmup.json', row)
            if self.save_plans:
                np.savez_compressed(self.output / 'warmup.npz', plan=plan, x0=x0, reference=reference)
        else:
            self.stream.write(json.dumps(clean(row), allow_nan=False) + '\n')
            self.rows.append(row)
            if self.save_plans:
                self.plans.append(plan.copy())
            self.states.append(x0.copy())
            self.ee.append(ee_pos.copy())
        now = time.monotonic()
        if warmup or now - self.last_checkpoint >= 30.:
            self.save('running', timestamp)
            self.last_checkpoint = now
        if warmup or now - self.last_progress >= 20.:
            print(json.dumps(clean(dict(event='progress', N=solver.N, batch_size=solver.batch_size,
                samples=len(self.rows), sim_time_s=timestamp, latest=row))), flush=True)
            self.last_progress = now

    def save(self, status, end_time=None, error=None):
        import numpy as np
        rows = self.rows
        times = np.asarray([row['native_ms'] for row in rows])
        tracking = np.asarray([row['tracking_error_m'] for row in rows])
        iterations = np.asarray([row['sqp_iters'] for row in rows])
        def mean(values):
            return float(np.mean(values)) if len(values) else None
        result = self.meta | dict(status=status, samples=len(rows),
            simulation_end_s=(end_time if end_time is not None else
                self.failure_state['timestamp'] if self.failure_state is not None else
                rows[-1]['timestamp_s'] if rows else 0.),
            wall_time_s=time.monotonic() - self.started, warmup=self.warmup,
            initialization=self.initialization_summary(),
            measured_wall_elapsed_s=self.measured_elapsed(),
            wall_time_budget_s=self.wall_time if self.protocol == 'wall-time' else None,
            budget_overrun_s=max(0., self.measured_elapsed() - self.wall_time) if self.protocol == 'wall-time' else None,
            slow_solve_count=sum(row['slow_solve'] for row in rows),
            cutoff_exceeded=any(row['slow_solve'] for row in rows),
            native_converged_count=sum(row['native_converged'] for row in rows),
            iteration_cap_count=sum(row['iteration_cap_hit'] for row in rows),
            all_native_stop_count=sum(row['all_native_stop'] for row in rows),
            any_cap_count=sum(row['any_cap'] for row in rows),
            batch_member_native_stop_count=sum(sum(row['native_converged_batch']) for row in rows),
            batch_member_cap_count=sum(sum(row['iteration_cap_hit_batch']) for row in rows),
            avg_native_ms=mean(times),
            solver_frequency_hz=1000. / mean(times) if len(times) and mean(times) > 0 else None,
            p95_native_ms=float(np.percentile(times, 95)) if len(times) else None,
            avg_sqp_iters=mean(iterations), max_sqp_iters_observed=int(iterations.max()) if len(iterations) else None,
            tracking_mean_m=mean(tracking), tracking_rms_m=float(np.sqrt(np.mean(tracking**2))) if len(tracking) else None,
            tracking_max_m=float(tracking.max()) if len(tracking) else None,
            error=error, failure_state=self.failure_state)
        save_json(self.output / 'results.json', result)
        arrays = dict(states=np.asarray(self.states), ee_actual=np.asarray(self.ee),
            timestamps=[row['timestamp_s'] for row in rows], solve_times=times,
            goal_distances=tracking, sqp_iters=iterations,
            native_converged=[row['native_converged'] for row in rows],
            sqp_iters_batch=[row['sqp_iters_batch'] for row in rows],
            native_converged_batch=[row['native_converged_batch'] for row in rows],
            iteration_cap_hit_batch=[row['iteration_cap_hit_batch'] for row in rows],
            finite_plan_batch=[row['finite_plan_batch'] for row in rows],
            reference_offsets=[row['reference_offset'] for row in rows],
            measured_wall_elapsed_s=[row['measured_wall_elapsed_s'] for row in rows],
            slow_solve=[row['slow_solve'] for row in rows])
        if self.save_plans:
            arrays['plans'] = np.asarray(self.plans)
        temporary = self.output / 'samples.tmp.npz'
        np.savez_compressed(temporary, **arrays)
        temporary.replace(self.output / 'samples.npz')
        return result


def run_cell(args):
    """Worker called by run.py in a fresh process for each matrix cell."""
    import numpy as np
    import pinocchio as pin
    sys.path.insert(0, str(ROOT / 'python'))
    import bsqp
    bsqp.__path__ = [str(args.build_dir.resolve() / 'modules'), *bsqp.__path__]
    from bsqp.interface import BSQP
    from bsqp.common import figure8
    from bsqp.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS
    from protocol import solver_parameters, REFERENCE_COMMIT

    n, batch, repeat = args.cell
    output = args.output.resolve() / f'N{n}_B{batch}_R{repeat}'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.json').exists():
        raise RuntimeError(f'Cell already contains results: {output}')
    np.random.seed(args.seed + repeat)
    prediction_dt = prediction_step(n, args.horizon_time)
    params = solver_parameters(n, args.horizon_time) | {'max_sqp_iters': args.max_sqp_iters}
    urdf = ROOT / 'examples/indy7_description/indy7.urdf'
    meta = dict(N=n, batch_size=batch, repeat=repeat, gpu_name=args.gpu_name,
        protocol=args.protocol, sim_step_s=CONTROL_DT, sim_time_s=args.sim_time,
        horizon_time=args.horizon_time, prediction_dt_s=prediction_dt,
        prediction_duration_s=prediction_dt * (n - 1),
        initialization_mode=args.initialization, max_init_solves=args.max_init_solves,
        max_solve_ms=args.max_solve_ms, max_sqp_iters=args.max_sqp_iters,
        solver_params=params, reference_commit=REFERENCE_COMMIT,
        native_criterion='Native kkt_converged flag: zero PCG iterations; not a nonlinear KKT tolerance check',
        timing='Full-batch synchronized native host time; initialization excluded from all measured aggregates',
        frequency='1000 / mean native milliseconds; solve-only frequency, not end-to-end control frequency',
        tracking=('Per-solve joint-6 origin position error vs current-time reference; motion startup included; not time-weighted'
                  if args.horizon_time is not None else
                  'Per-solve joint-6 origin position error vs next reference knot; motion startup included; not time-weighted'),
        cost_rule=cost_rule(args.horizon_time),
        batch_policy='Batch 0 controls simulation; its unshifted plan is broadcast after every measured solve',
        cutoff_semantics='Stop after first measured native solve strictly above cutoff; zero disables; initialization excluded; slow sample retained',
        budget_semantics=('Soft wall budget starts after initialization and recording; checked between complete updates; final output excluded'
                          if args.protocol == 'wall-time' else 'Finite simulation-time budget with historical capped clock'))
    recorder = Recorder(output, batch_size=batch, max_sqp_iters=args.max_sqp_iters,
        max_solve_ms=args.max_solve_ms, wall_time=args.wall_time, protocol=args.protocol,
        meta=meta, save_plans=getattr(args, 'save_plans', False))
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    status, end_time, error = 'running', None, None
    try:
        solver = BSQP(str(urdf), batch, n, prediction_dt, **params)
        module = Path(solver.lib.__file__).resolve()
        if module.parent != (args.build_dir / 'modules').resolve():
            raise RuntimeError(f'Unexpected solver module: {module}')
        meta.update(module=str(module), module_sha256=hashlib.sha256(module.read_bytes()).hexdigest())
        start = np.r_[INDY7_START_CONFIGS['ready'], np.zeros(6)]
        reference_params = FIG8_DEFAULT_PARAMS | ({'cycles': 1} if args.protocol == 'wall-time' else {})
        reference = figure8(REFERENCE_DT, **reference_params)
        np.savez_compressed(output / 'inputs.npz', x_start=start, reference=reference)
        end_time, status = run_instrumented(solver, pin.buildModelFromUrdf(str(urdf)),
            start, reference, recorder, protocol=args.protocol, wall_time=args.wall_time,
            sim_time=args.sim_time, max_solve_ms=args.max_solve_ms,
            horizon_time=args.horizon_time, initialization=args.initialization,
            max_init_solves=args.max_init_solves)
    except NumericInstability as exc:
        status, error = 'unstable_nonfinite', str(exc)
    except KeyboardInterrupt as exc:
        status, error = 'interrupted', str(exc)
    except Exception:
        status, error = 'error', traceback.format_exc()
    finally:
        if recorder.measurement_started is not None and recorder.measurement_finished is None:
            recorder.finish_measurement()
        result = recorder.save(status, end_time, error)
        recorder.stream.close()
        signal.signal(signal.SIGTERM, previous_handler)
        print(json.dumps(clean(result), allow_nan=False), flush=True)
    return 1 if status in ('error', 'interrupted') else 0
