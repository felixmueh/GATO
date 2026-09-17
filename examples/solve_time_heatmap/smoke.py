"""Small public-binding liveness check, also suitable for compute-sanitizer."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build-dir', type=Path, default=Path('build/solve-time-heatmap-felix-devel'))
p.add_argument('--knots', type=int, default=8)
p.add_argument('--batch', type=int, default=1)
args = p.parse_args()
name = f'bsqpN{args.knots}_indy7'
path = next((args.build_dir / 'modules').glob(name + '*.so'))
spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
solver = getattr(module, f'BSQP_{args.batch}_float')(
    .01, 1, .001, 200, 1e-4, 1., 10., 2., .01, 2e-6, 50., 0., 0., .01, 0., 0., .01)
x = np.tile(np.array([-1.096711, -.09903229, .83125766, -.10907673, .49704404,
                     .01499449, 0, 0, 0, 0, 0, 0], dtype=np.float32), (args.batch, 1))
xu = np.zeros((args.batch, args.knots * 18 - 6), dtype=np.float32)
for k in range(args.knots):
    xu[:, k*18:k*18+12] = x
reference = np.tile(np.array([-.3535534, .3535534, .8, 0, 0, 0], dtype=np.float32),
                    (args.batch, args.knots))
solver.set_f_ext_batch(np.zeros((args.batch, 6), dtype=np.float32))
solver.reset_dual()
for _ in range(3):
    result = solver.solve(xu, .01, x, reference)
    assert np.isfinite(result['XU']).all(), 'Nonfinite solution'
    assert np.isfinite(result['final_merit']).all(), 'Nonfinite merit'
    assert result['sqp_time_us'] > 0, 'Invalid solve time'
    assert np.all(result['sqp_iters'] == 1), 'Unexpected SQP count'
    print(f'N={args.knots} B={args.batch}: {result["sqp_time_us"]} us; '
          f'PCG {np.asarray(result["pcg_iters"]).reshape(-1).tolist()}')
    xu = result['XU'].copy()
    next_state = solver.sim_forward(x[0], xu[0, 12:18], .001)
    assert np.isfinite(next_state).all(), 'Nonfinite batch simulation'
