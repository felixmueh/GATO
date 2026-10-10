"""One fresh N/flag worker, public BSQP wrapper and reviewed native-return capture."""
import importlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from experiments.resource_profile_campaign.common import F


def native(context, N):
    info = context['selected_module']; source = Path(info['source_root'])
    name = f'bsqp.bsqpN{N}_tiago_right'
    if any(n.startswith('bsqp.bsqpN') for n in sys.modules): raise RuntimeError('Fresh one-N/flag process required')
    sys.path[:0] = [str(source / 'python'), str(source / 'tiago_src')]
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != Path(info['path']).resolve() or dict(module.experiment_build_flags) != info['flags']:
        raise RuntimeError('Imported module path/compiled flags differ')
    return module, source


class Factory:
    def __init__(self, context, bank, events): self.context=context; self.bank=bank; self.events=events; self.interface=None
    def __call__(self, B, SQP, PCG):
        if self.interface is None:
            started=time.perf_counter()
            self.module, self.source = native(self.context, 32)
            self.interface = importlib.import_module('bsqp.interface')
            if Path(self.interface.__file__).resolve() != self.source / 'python/bsqp/interface.py': raise RuntimeError('Actual public wrapper differs')
            self.events('native_wrapper_imported',dict(import_wall_s=time.perf_counter()-started,scope='native+public-wrapper import/identity-check host wall, before solver constructor'))
        params = dict(self.bank['preset'], max_sqp_iters=SQP, max_pcg_iters=PCG)
        started=time.perf_counter()
        solver = self.interface.BSQP(str(self.source/'gato/dynamics/tiago_right/tiago_right_arm.urdf'),B,32,.03,
            plant_type='tiago_right',**params)
        constructor_wall_s=time.perf_counter()-started
        self.events('native_owner_constructed',dict(constructor_wall_s=constructor_wall_s,B=B,N=32,effective_solver_params=params))
        if solver.lib is not self.module: raise RuntimeError('Resolved native differs')
        return F.Owned(SimpleNamespace(solver=solver,N=32,dt=.03,batch_size=B,solver_params=params),self.events)


def query(context, case):
    module, _ = native(context, case['N'])
    return module.experiment_resources(case['B'])
