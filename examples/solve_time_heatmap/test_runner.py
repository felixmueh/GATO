"""Regression checks for protecting measurements before GPU work starts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import types

import numpy as np
from unittest.mock import patch

import run


class RunnerTests(unittest.TestCase):
    def args(self, output):
        return run.parser().parse_args(['--gpu-name', 'Test GPU', '--output', str(output)])

    def test_backend_identity_and_separate_default_builds(self):
        args = self.args(Path('/unused'))
        self.assertEqual(args.backend, 'PCG')
        self.assertEqual(run.default_build_dir(args).name, 'solve-time-heatmap-tiago_right')
        run.validate_backend_identity('PCG', args.backend)
        args.backend = 'QDLDL'
        self.assertEqual(run.default_build_dir(args).name, 'solve-time-heatmap-tiago_right-qdldl')
        self.assertEqual(run.configuration(args)['linsys_solver'], 'QDLDL')
        command = run.cell_command(args, 32, 1, 0)
        self.assertEqual(command[command.index('--backend')+1], 'QDLDL')
        with self.assertRaisesRegex(ValueError, 'Backend identity'):
            run.validate_backend_identity('PCG', args.backend)

    def test_qdldl_scope_rejected_before_runtime_setup(self):
        with patch.object(run, 'match_torch_libraries') as setup:
            for options in ([], ['--knots', '32', '--batches', '32'],
                            ['--plant', 'indy7', '--knots', '32', '--batches', '1']):
                with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        run.main(['--gpu-name', 'Test GPU', '--backend', 'QDLDL', *options])
                    self.assertEqual(error.exception.code, 2)
            setup.assert_not_called()

    def test_worker_retains_partial_measurement_on_direct_rejection(self):
        import experiment
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(Path(folder) / 'raw')
            args.backend, args.cell = 'QDLDL', [32, 1, 0]
            args.build_dir = Path(folder) / 'build'
            modules = args.build_dir / 'modules'
            modules.mkdir(parents=True)
            module = modules / 'mock.so'
            module.write_bytes(b'CPU-test-double')
            solver = types.SimpleNamespace(N=32, batch_size=1, linsys_solver='QDLDL',
                lib=types.SimpleNamespace(__file__=str(module)),
                get_stats=lambda: dict(sqp_iters=[1], kkt_converged=[False],
                    direct_solve_work=np.ones((1, 1), dtype=int)))
            package = types.ModuleType('bsqp')
            package.__path__ = []
            interface = types.ModuleType('bsqp.interface')
            interface.BSQP = lambda *a, **kw: solver
            def reject_after_sample(solver, model, start, reference, recorder, **kwargs):
                values = (solver, np.zeros(20), start, reference, 1000., .01,
                          1, np.zeros(3), .02)
                recorder(*values, True, finite_batch=[True])
                recorder.begin_measurement()
                recorder(*values, False, finite_batch=[True])
                raise RuntimeError('QDLDL full Schur residual rejected')
            with patch.dict(sys.modules, {'bsqp': package, 'bsqp.interface': interface}), \
                 patch.object(experiment, 'plant_inputs', return_value=(Path('unused'), np.zeros(14), np.zeros(192))), \
                 patch('pinocchio.buildModelFromUrdf', return_value=object()), \
                 patch.object(experiment, 'run_instrumented', side_effect=reject_after_sample), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(experiment.run_cell(args), 1)
            output = args.output / 'N32_B1_R0'
            receipt = json.loads((output / 'results.json').read_text())
            self.assertEqual(receipt['status'], 'error')
            self.assertEqual(receipt['samples'], 1)
            self.assertEqual(receipt['linsys_solver'], 'QDLDL')
            self.assertIn('RuntimeError: QDLDL full Schur residual rejected', receipt['error'])
            self.assertEqual(len((output / 'solves.jsonl').read_text().splitlines()), 1)
            self.assertEqual(len(np.load(output / 'samples.npz')['timestamps']), 1)

    def test_new_run_refuses_unrelated_existing_cell_data(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            receipt = output / 'N8_B1_R0'
            receipt.mkdir()
            original = '{"status":"completed","samples":99}'
            (receipt / 'results.json').write_text(original)
            with patch.object(run, 'source_paths', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'must be empty'):
                    run.prepare_manifest(self.args(output))
            self.assertEqual((receipt / 'results.json').read_text(), original)
            self.assertFalse((output / 'results.json').exists())

    def test_nonfinite_budgets_rejected_before_runtime_setup(self):
        with patch.object(run, 'match_torch_libraries') as setup:
            for flag in ('--wall-time', '--sim-time', '--max-solve-ms', '--timeout'):
                for value in ('nan', 'inf'):
                    with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            run.main(['--gpu-name', 'Test GPU', flag, value])
                        self.assertEqual(error.exception.code, 2)
            setup.assert_not_called()

    def test_resume_archives_unrecorded_partial_before_restarting(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            cell = output / 'N8_B1_R0'
            cell.mkdir()
            (cell / 'results.json').write_text(json.dumps({'status': 'running'}))
            (cell / 'solves.jsonl').write_text('original partial measurement\n')
            def finish(command, **kwargs):
                self.assertFalse(cell.exists())
                cell.mkdir()
                (cell / 'results.json').write_text(json.dumps(
                    dict(N=8, batch_size=1, repeat=0, status='completed', samples=2)))
                return type('Process', (), {'returncode': 0})()
            with patch.object(run.subprocess, 'run', side_effect=finish):
                result = run.run_cell_process(self.args(output), 8, 1, 0)
            self.assertEqual(result['samples'], 2)
            archived = list((output / 'interrupted').glob('*/solves.jsonl'))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_text(), 'original partial measurement\n')

    def test_updates_config_and_worker_command_and_default(self):
        args = self.args(Path('/unused'))
        self.assertIsNone(args.updates)
        self.assertIsNone(run.configuration(args)['updates'])
        self.assertNotIn('--updates', run.cell_command(args, 8, 1, 0))
        args.updates = 300
        self.assertEqual(run.configuration(args)['updates'], 300)
        command = run.cell_command(args, 512, 1, 0)
        self.assertEqual(command[command.index('--updates')+1], '300')
        self.assertIn(512, run.KNOTS)

    def test_missing_nvidia_smi_retains_cuda_probe_identity(self):
        identity = dict(device='Orin', gpu_uuid='GPU-test', multiprocessors=16,
                        cuda_driver_version=12060)
        with patch.object(run.subprocess, 'check_output', side_effect=FileNotFoundError('nvidia-smi')):
            snapshot = run.gpu_snapshot(resource=identity)
        for key, value in identity.items():
            self.assertEqual(snapshot[key], value)
        self.assertIsNone(snapshot['nvidia_smi'])
        self.assertIn('nvidia-smi', snapshot['telemetry_error'])

    def test_failed_nvidia_smi_is_optional_and_cuda_selection_wins(self):
        with patch.object(run.subprocess, 'check_output', side_effect=subprocess.CalledProcessError(1, 'nvidia-smi')):
            self.assertIsNotNone(run.gpu_snapshot()['telemetry_error'])
        with patch.object(run.subprocess, 'check_output', return_value='name,uuid\nPhysical GPU,GPU-first\n'):
            snapshot = run.gpu_snapshot(dict(device='Selected GPU', gpu_uuid='GPU-selected', multiprocessors=8))
        self.assertEqual(snapshot['gpu_uuid'], 'GPU-selected')
        self.assertEqual(snapshot['device'], 'Selected GPU')

    def test_invalid_updates_rejected_before_runtime_setup(self):
        with patch.object(run, 'match_torch_libraries') as setup:
            for options in (['--updates', '0'], ['--updates', '-1'], ['--updates', '1.5'],
                            ['--plant', 'indy7', '--protocol', 'reference', '--updates', '300']):
                with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        run.main(['--gpu-name', 'Test GPU', *options])
                    self.assertEqual(error.exception.code, 2)
            setup.assert_not_called()

    def test_timeout_keeps_recorded_fixed_update_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            args = self.args(output)
            args.updates, args.timeout = 300, 1.
            cell = output/'N8_B1_R0'
            def timeout(command, **kwargs):
                cell.mkdir()
                (cell/'results.json').write_text(json.dumps(dict(
                    N=8, batch_size=1, repeat=0, status='running', samples=1,
                    measured_updates=1, measured_update_budget=300)))
                (cell/'solves.jsonl').write_text('retained partial\n')
                raise subprocess.TimeoutExpired(command, kwargs['timeout'])
            with patch.object(run.subprocess, 'run', side_effect=timeout):
                result = run.run_cell_process(args, 8, 1, 0)
            self.assertEqual(result['status'], 'error')
            self.assertEqual(result['samples'], 1)
            self.assertEqual(result['measured_update_budget'], 300)
            self.assertEqual((cell/'solves.jsonl').read_text(), 'retained partial\n')


if __name__ == '__main__':
    unittest.main()
