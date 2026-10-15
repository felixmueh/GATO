"""CPU contracts for the three frozen Jetson continuation tracking cases."""
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest

import run
from adapters.task_parameters import worker_path

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/launch/jetson_tracking_c2_boundary_20261015.json'
SOURCE='c15303b2dd68529376487cf6a5465807b9205b38'
OLD='90bdb825d059e22b709e2e53a785d7d034aad1b0'
LARGE='f6cb96f1aa0c014ffc221b1b81adb2beebbb9564'
REF={128:'0578e468030e730895373795d7f57496b1b86979d5f3951c185a526908443595',
     256:'2abc6812834c398762c83b70b4761969e49ef09df7b558003d9713c58fca489f',
     512:'90a43e1a5be2949a8a831235918e6b4528b62f747dc48ca1d371c3609e4a3201'}


class TrackingContinuationPlan(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads(PLAN.read_text())

    def test_three_cases_one_build_and_one_canonical_without_waiver(self):
        self.assertEqual(self.plan['scientific_episodes'],3)
        self.assertEqual(self.plan['canonical_episodes'],1)
        keys,groups=set(),set()
        for n,e in zip((128,256,512),self.plan['experiments']):
            self.assertEqual(e['source_commit'],SOURCE)
            self.assertEqual((e['timing_mode'],e['backend'],e['machines'],e['repeats']),('B','internal',['jetson'],1))
            self.assertEqual(len(e['cases']),1)
            self.assertEqual(e['cases'][0],dict(id='figure8-6s',name='6-second figure-eight',parameters=dict(N=n,dt_s=.03,sqp_max_iter=5,pcg_max_iter=120)))
            pin=OLD if n==128 else LARGE
            self.assertEqual(e['collector_source_commit'],pin)
            self.assertEqual(e['command'][-2:],['--collector-pin',pin])
            self.assertEqual(e['preflight']['command'][-3:],['--collector-pin',LARGE,'--preflight'])
            self.assertNotIn('{case}',e['preflight']['command'])
            self.assertNotIn('allow_report_only_safety',e)
            self.assertNotIn(run.DANGEROUS_SAFETY_FLAG,e['command'])
            self.assertIn('-DKNOTS=32;128;256;512',e['build']['commands'][0])
            self.assertEqual(len(e['build']['artifacts']),5)
            for path,sha in e['build']['inputs'].items():
                self.assertEqual(hashlib.sha256((ROOT/'experiments'/path).read_bytes()).hexdigest(),sha)
            for path,sha in e['inputs'].items():
                self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),sha)
            keys.add(run.build_key(e,{'cuda_arch':'87'},'/python',{'python_abi':'same'}))
            groups.add(run.preflight_group(e))
        self.assertEqual(len(keys),1)
        self.assertEqual(len(groups),1)
        # These are all fields that distinguish shared preflight signatures;
        # per-case collector metadata must not accidentally split the canonical.
        fields=('source_commit','inputs','parameters','preflight','timing_mode','backend')
        self.assertTrue(all(tuple(e[k] for k in fields)==tuple(self.plan['experiments'][0][k] for k in fields)
                            for e in self.plan['experiments']))

    def test_original_resolved_protocols_references_and_shared_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolvers={}
            for pin in (OLD,LARGE):
                worker=worker_path(ROOT,Path(tmp),pin)
                resolvers[pin]=runpy.run_path(str(worker.with_name('protocol.py')))['resolve_protocol']
            inputs=ROOT/'experiments/inputs/t01'
            for e in self.plan['experiments']:
                case=e['cases'][0];n=case['parameters']['N']
                p=resolvers[e['collector_source_commit']](inputs,case,'internal')
                self.assertEqual((p['timing_mode'],p['offline_timing'],p['batch_size']),('B','solve_time',1))
                self.assertEqual((p['dt_s'],p['nominal_feedback_s'],p['sim_dt_s']),(.03,.03,.003))
                self.assertEqual((p['duration_s'],p['period_s'],p['provisional_RMS_gate_m'],p['provisional_maximum_gate_m']),(6.,6.,.025,.05))
                self.assertEqual(p['reference_sha256'],REF[n])
                self.assertFalse(p['reset_rho_each_solve']);self.assertTrue(p['adapt_rho'])
                self.assertFalse(p['safety_policy']['report_only_physical_stops'])
                self.assertFalse(p['safety_policy']['torque_clamping'])
                s=p['solver_params']
                self.assertEqual((s['max_sqp_iters'],s['max_pcg_iters'],s['pcg_tol'],s['kkt_tol']),(5,120,.001,.001))
                self.assertEqual((s['q_cost'],s['qd_cost'],s['u_cost'],s['N_cost']),(4.,.01,.0001,80.))
                self.assertEqual((s['q_lim_cost'],s['vel_lim_cost'],s['ctrl_lim_cost']),(.01,.001,.003))
                self.assertEqual((s['ee_orient_cost'],s['ee_orient_N_cost']),(0.,0.))
            canonical=json.loads((ROOT/'experiments/inputs/task_parameters/canonical_case.json').read_text())
            old=resolvers[OLD](inputs,canonical,'internal',preflight=True)
            large=resolvers[LARGE](inputs,canonical,'internal',preflight=True)
            self.assertEqual(old,large)
            self.assertEqual((large['N'],large['dt_s'],large['solver_params']['max_sqp_iters']),(32,.03,5))


if __name__=='__main__':unittest.main()
