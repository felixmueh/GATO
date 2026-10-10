"""Lazy production factory and lightweight raw-return capture, no eager imports."""
import importlib
from pathlib import Path
import sys


class NativeCapture:
    def __init__(self, native): self.native = native; self.reply = None; self.inputs = None; self.sequence = 0
    def __getattr__(self, name): return getattr(self.native, name)
    def solve(self, xu, timestep, state, reference):
        self.reply = None; self.sequence += 1
        self.inputs = dict(XU=xu, timestep=timestep, state=state, reference=reference)
        self.reply = self.native.solve(xu, timestep, state, reference)
        return self.reply


class Events:
    """Collector callbacks happen after solve timing; durable raw capture happens here."""
    def __init__(self, recorder): self.recorder = recorder; self.capture = None; self.flushed = 0; self.native_calls = 0; self.observed_sequence = 0
    def bind(self, capture): self.capture = capture; self.flushed = 0; self.observed_sequence = 0
    def __call__(self, event, record):
        if self.capture is not None:
            self.native_calls += self.capture.sequence - self.observed_sequence
            self.observed_sequence = self.capture.sequence
        if event in ('solve_returned', 'solve_completed', 'solve_failed', 'execution_failed'):
            capture = self.capture
            if capture is not None and capture.sequence != self.flushed:
                metadata = {k: record[k] for k in ('episode_id', 'slot', 'solve_index', 'category', 'time_s',
                            'wall_solve_time_s', 'status', 'error_type', 'error') if k in record}
                if capture.reply is not None:
                    self.recorder('native_return', dict(native_reply=capture.reply,
                        native_argument_arrays_at_return=capture.inputs, wrapper_event=event, wrapper_metadata=metadata,
                        scope='native payload/argument arrays at return, serialized outside wrapper timer; precall wrapper bytes are in solve_started'))
                else:
                    self.recorder('native_call_failed', dict(native_reply=None,
                        native_reply_unavailable_reason='native solve raised before returning a payload',
                        native_argument_arrays_at_failure=capture.inputs, wrapper_event=event, wrapper_metadata=metadata,
                        scope='argument references at failure, serialized outside wrapper timer; precall wrapper bytes are in solve_started'))
                self.flushed = capture.sequence
        self.recorder(event, record)


class Owned:
    def __init__(self, controller, events):
        self.controller = controller
        self.capture = NativeCapture(controller.solver.solver)
        controller.solver.solver = self.capture
        events.bind(self.capture)
    def __getattr__(self, name): return getattr(self.controller, name)
    def close(self):
        self.capture.reply = None; self.capture.inputs = None; self.capture.native = None
        self.controller.solver.solver = None
        self.controller.solver = None
        self.controller = None


def production_builder(context, N):
    """Called only after release/readiness guards; one N per fresh worker process."""
    info = context['modules'][str(N)]; source = Path(info['source_root'])
    for name in sys.modules:
        if name.startswith('bsqp.bsqpN') and name != f'bsqp.bsqpN{N}_tiago_right':
            raise RuntimeError('Fresh one-horizon worker required')
    sys.path.insert(0, str(source / 'python')); sys.path.insert(0, str(source / 'tiago_src'))
    native = importlib.import_module(f'bsqp.bsqpN{N}_tiago_right')
    production = importlib.import_module('gato_tiago.tiago_mpc_controller')
    interface = importlib.import_module('bsqp.interface')
    if (Path(native.__file__).resolve() != Path(info['path']).resolve()
            or Path(production.__file__).resolve() != source / 'tiago_src/gato_tiago/tiago_mpc_controller.py'
            or Path(interface.__file__).resolve() != source / 'python/bsqp/interface.py'):
        raise RuntimeError('Imported native/controller/wrapper identity differs')
    import numpy as np
    import pinocchio as pin
    urdf = source / 'gato/dynamics/tiago_right/tiago_right_arm.urdf'
    def create(kind, params):
        model = pin.buildModelFromUrdf(str(urdf))
        owner = production.MPC_GATO(model, str(urdf), N=N, dt=.03, batch_size=1,
            constant_f_ext=np.zeros(6, dtype=np.float32), track_full_stats=True,
            plant_type='tiago_right', pendulum_config=None, solver_params=params, final_approach=False)
        if owner.solver.lib is not native or owner.solver_params != params:
            raise RuntimeError('Resolved native owner/preset differs')
        return owner
    return create


class Factory:
    def __init__(self, context, bank, events, builder=None):
        self.context = context; self.bank = bank; self.events = events
        self.builder = builder or production_builder; self.loaded_N = None; self.create = None
    def __call__(self, kind, cell):
        N = cell['N']
        if self.loaded_N is not None and self.loaded_N != N: raise RuntimeError('One N per worker')
        if self.create is None:
            self.create = self.builder(self.context, N); self.loaded_N = N
        params = dict(self.bank['presets'][kind], max_sqp_iters=cell['SQP'], max_pcg_iters=cell['PCG'])
        return Owned(self.create(kind, params), self.events)


def integrator(owner):
    source = importlib.import_module('gato_tiago.tiago_mpc_controller')
    return lambda q, v, u, dt: source.rk4(owner.model, owner.data, q, v, u, dt, owner.actual_f_ext)
