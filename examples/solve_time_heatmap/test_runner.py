"""Regression checks for protecting measurements before GPU work starts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run


class RunnerTests(unittest.TestCase):
    def args(self, output):
        return run.parser().parse_args(['--gpu-name', 'Test GPU', '--output', str(output)])

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


if __name__ == '__main__':
    unittest.main()
