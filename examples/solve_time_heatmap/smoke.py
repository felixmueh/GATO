"""Small public-binding liveness check, also suitable for compute-sanitizer."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

from plants import DEFAULT_PLANT, PLANTS, plant_inputs
from protocol import solver_parameters
from settings import REFERENCE_DT

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--plant', choices=PLANTS, default=DEFAULT_PLANT)
p.add_argument('--build-dir', type=Path,
               help='Default: build/solve-time-heatmap-<plant>')
p.add_argument('--knots', type=int, default=8)
p.add_argument('--batch', type=int, default=1)
args = p.parse_args()
if args.build_dir is None:
    args.build_dir = ROOT / f'build/solve-time-heatmap-{args.plant}'
name = f'bsqpN{args.knots}_{args.plant}'
paths = list((args.build_dir / 'modules').glob(name + '*.so'))
if len(paths) != 1:
    p.error(f'Expected one {name} extension in {args.build_dir / "modules"}; found {len(paths)}')
spec = importlib.util.spec_from_file_location(name, paths[0])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
params = solver_parameters(args.knots, plant=args.plant)
# One SQP iteration establishes binding liveness, not solver convergence.
params['max_sqp_iters'] = 1
parameter_order = ('max_sqp_iters', 'kkt_tol', 'max_pcg_iters', 'pcg_tol',
                   'solve_ratio', 'mu', 'q_cost', 'qd_cost', 'u_cost', 'N_cost',
                   'ee_orient_cost', 'ee_orient_N_cost', 'q_lim_cost',
                   'vel_lim_cost', 'ctrl_lim_cost', 'rho')
solver = getattr(module, f'BSQP_{args.batch}_float')(
    REFERENCE_DT, *(params.get(key, 0.) for key in parameter_order))
_, start, references = plant_inputs(args.plant)
nx, nu = len(start), len(start) // 2
stride = nx + nu
x = np.tile(np.asarray(start, dtype=np.float32), (args.batch, 1))
xu = np.zeros((args.batch, args.knots * stride - nu), dtype=np.float32)
for k in range(args.knots):
    xu[:, k * stride:k * stride + nx] = x
reference = np.tile(np.asarray(references, dtype=np.float32).reshape(-1, 6)[0],
                    (args.batch, args.knots))
solver.set_f_ext_batch(np.zeros((args.batch, 6), dtype=np.float32))
solver.reset_dual()
for _ in range(3):
    result = solver.solve(xu, REFERENCE_DT, x, reference)
    assert np.isfinite(result['XU']).all(), 'Nonfinite solution'
    assert np.isfinite(result['final_merit']).all(), 'Nonfinite merit'
    assert result['sqp_time_us'] > 0, 'Invalid solve time'
    assert np.all(result['sqp_iters'] == 1), 'Unexpected SQP count'
    print(f'{args.plant} N={args.knots} B={args.batch}: {result["sqp_time_us"]} us; '
          f'PCG {np.asarray(result["pcg_iters"]).reshape(-1).tolist()}')
    xu = result['XU'].copy()
    next_state = solver.sim_forward(x[0], xu[0, nx:nx + nu], .001)
    assert np.isfinite(next_state).all(), 'Nonfinite batch simulation'
