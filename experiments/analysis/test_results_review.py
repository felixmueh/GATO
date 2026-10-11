"""CPU checks for scientific denominator and pairing decisions."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from results_review import outcome, task_success, compatible

class ReviewChecks(unittest.TestCase):
    def test_known_crash_only_changes_reported_outcome(self):
        raw=dict(execution_error='nonfinite executed plant state; partial trajectory retained',task_success=None,physical_pass=None)
        report=outcome(raw)
        self.assertFalse(report['reported_task_success'])
        self.assertFalse(report['reported_physical_pass'])
        self.assertIsNone(raw['task_success'])
        self.assertIsNone(outcome(dict(execution_error='IO failure'))['reported_task_success'])

    def test_missing_task_field_and_explicit_unavailable_differ(self):
        self.assertTrue(task_success(dict(complete=True,outcome='reached')))
        self.assertIsNone(task_success(dict(complete=True,outcome='reached',task_success=None)))
        self.assertFalse(task_success(dict(complete=True,outcome='timeout')))

    def test_pairing_checks_settings_and_task_beyond_binary(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b';a.mkdir();b.mkdir()
            fields=('N','knot_dt','feedback_seconds','simulation_step','simulation_timeout','endpoint','initialization','proposal','dual_policy','rho_policy','physical_limits','success')
            m=dict.fromkeys(fields,1);m.update(binary_sha256='same',solver_params=dict(max_sqp_iters=5,pcg=120),args={})
            for p in (a,b):
                (p/'manifest.json').write_text(json.dumps(m));np.savez(p/'inputs.npz',initial_state=np.zeros(14),target_xyz=np.zeros(3),target_rotation=np.eye(3),reference=np.zeros(192))
            m['solver_params']['max_sqp_iters']=80;(b/'manifest.json').write_text(json.dumps(m))
            self.assertEqual(compatible(a,b),[])
            m['rho_policy']='changed';(b/'manifest.json').write_text(json.dumps(m))
            self.assertIn('rho_policy',compatible(a,b))
            np.savez(b/'inputs.npz',initial_state=np.ones(14),target_xyz=np.zeros(3),target_rotation=np.eye(3),reference=np.zeros(192))
            self.assertIn('task.initial_state',compatible(a,b))

if __name__=='__main__':unittest.main()
