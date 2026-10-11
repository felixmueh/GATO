"""BB adapter boundaries use a tiny episode double; no native/GPU calls."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from experiments.adapters import bb
from experiments.recording import iter_events


class BBAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.source)], check=True)
        subprocess.run(['git', '-C', str(self.source), '-c', 'user.name=Test', '-c',
                        'user.email=test@example.invalid', 'commit', '--allow-empty', '-qm', 'fixture'], check=True)
        self.module = self.source / 'native.so'
        self.module.write_bytes(b'fixture: never load this module')
        self.pool = self.root / 'pool'
        self.pool.mkdir()
        for name in ('pool.npz', 'pairs.npz'):
            (self.pool / name).write_bytes(b'fixture: not a real task pool')
        self.protocol = self.root / 'bb_protocol.json'
        self.protocol.write_text(json.dumps(dict(timing_mode='B', offline_timing='solve_time', task_ids=[3626])))
        self.case = dict(id='pair-03626-B1', parameters=dict(pair_id=3626, batch=1))
        self.output = self.root / 'result'

    def run_case(self, worker, **overrides):
        return bb.run_case(self.source, self.module, self.pool, self.protocol,
                           self.case, self.output, worker=worker, **overrides)

    def test_ordinary_negative_is_completed_with_independent_physical_gate(self):
        calls = []
        class Worker:
            def run_episode(_, args, pair, batch, dt, *, recorder, final_checkpoint_only):
                calls.append((args, pair, batch, dt, final_checkpoint_only))
                recorder('solve_returned', dict(optimized=np.array([np.nan, np.inf], dtype=np.float32)))
                return dict(complete=True, task_success=False, physical=dict(physical_valid=True), outcome='timeout')
        self.assertEqual(self.run_case(Worker()), 0)
        args, pair, batch, dt, final_only = calls[0]
        self.assertEqual((pair, batch, dt, final_only), (3626, 1, .03, True))
        self.assertEqual((args.feedback, args.sim_dt, args.seconds, args.seed), (.03, .003, 8., 20260925))
        self.assertEqual((args.qd_cost, args.vel_lim_cost, args.prefix_guard), (.1, .1, False))
        metrics = json.loads((self.output / 'metrics.json').read_text())
        self.assertFalse(metrics['task_success'])
        self.assertTrue(metrics['physical_pass'])
        self.assertIsNone(metrics['solver_numerical_pass'])
        events = list(iter_events(self.output))
        array = events[0]['record']['optimized']
        self.assertEqual(array.dtype, np.float32)
        self.assertTrue(np.isnan(array[0]) and np.isinf(array[1]))

    def test_episode_exception_preserves_partial_raw_and_execution_failure(self):
        class Worker:
            def run_episode(_, args, pair, batch, dt, *, recorder, final_checkpoint_only):
                recorder('solve_input', dict(candidates=np.arange(4, dtype=np.float32)))
                raise RuntimeError('injected episode execution error')
        self.assertEqual(self.run_case(Worker()), 1)
        self.assertEqual([e['event'] for e in iter_events(self.output)], ['solve_input', 'execution_failed'])
        self.assertIn('injected episode', json.loads((self.output / 'metrics.json').read_text())['execution_error'])

    def test_failed_preflight_returns_failure_without_changing_task_gate(self):
        class Worker:
            def run_episode(_, *args, **kwargs):
                return dict(complete=True, task_success=True, physical=dict(physical_valid=False), outcome='reached')
        preflight = self.root / 'preflight.json'
        preflight.write_text('{}')
        self.case['parameters']['pair_id'] = -1
        self.assertEqual(self.run_case(Worker(), preflight_input=preflight), 1)
        metrics = json.loads((self.output / 'metrics.json').read_text())
        self.assertTrue(metrics['task_success'])
        self.assertFalse(metrics['physical_pass'])
        self.assertIsNone(metrics['execution_error'])

    def test_wrong_roster_and_old_attempt_are_rejected_before_episode(self):
        class Worker:
            def run_episode(_, *args, **kwargs):
                raise AssertionError('worker must not execute')
        self.case['parameters']['pair_id'] = 123
        self.assertEqual(self.run_case(Worker()), 1)
        old = (self.output / 'metrics.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_case(Worker())
        self.assertEqual((self.output / 'metrics.json').read_bytes(), old)


if __name__ == '__main__':
    unittest.main()
