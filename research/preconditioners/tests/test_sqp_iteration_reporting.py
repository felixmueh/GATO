"""CPU-only checks for native-interval accounting and zero-PCG classification."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from upstream_reach import summarize_sqp_iterations
import csv
import contextlib
import io
import json
import tempfile
from types import SimpleNamespace
from jetson_scaling import summarize

class NativeIterationReporting(unittest.TestCase):
    def fixture(self):
        records=[dict(native_solve_wall_us=1000,stats=dict(sqp_iters=[1],pcg_iters=[[7]])),
                 dict(native_solve_wall_us=500,stats=dict(sqp_iters=[1],pcg_iters=[[0]]))]
        solves=[dict(native_wall_us=1000,native_copy_us=10,candidate_copy_us=20,transfer_us=30,
                     compute_us=970,batch_size=1,max_sqp_iters=1,actual_sqp_iters=1,pcg_iters=7,zero_pcg_shortcut=False),
                dict(native_wall_us=500,native_copy_us=5,candidate_copy_us=15,transfer_us=20,
                     compute_us=480,batch_size=1,max_sqp_iters=1,actual_sqp_iters=1,pcg_iters=0,zero_pcg_shortcut=True)]
        return records,dict(native_sqp_solves=solves,transfer_wall_us=999999)
    def test_uses_native_copy_interval_and_retains_zero_shortcut(self):
        records,metrics=self.fixture();summary=summarize_sqp_iterations(records,metrics)
        self.assertAlmostEqual(summary['sqp_iteration_compute_mean_ms'],.725)
        self.assertEqual(summary['sqp_iteration_count'],2)
        self.assertEqual(summary['sqp_iteration_zero_pcg']['count'],1)
        self.assertEqual(summary['sqp_iteration_nonzero_pcg']['count'],1)
        self.assertAlmostEqual(summary['sqp_iteration_zero_pcg']['compute_mean_ms'],.480)
        self.assertEqual(records[0]['sqp_iteration']['compute_us'],970)
    def test_rejects_missing_or_mismatched_mapping(self):
        for defect in ('missing','wall','budget','count','pcg'):
            records,metrics=self.fixture()
            if defect=='missing':metrics['native_sqp_solves'].pop()
            elif defect=='wall':records[0]['native_solve_wall_us']+=1
            elif defect=='budget':metrics['native_sqp_solves'][0]['max_sqp_iters']=2
            elif defect=='count':records[0]['stats']['sqp_iters']=[2]
            elif defect=='pcg':records[0]['stats']['pcg_iters']=[[8]]
            with self.subTest(defect=defect),self.assertRaises(ValueError):summarize_sqp_iterations(records,metrics)
    def test_unmeasured_compute_is_not_fabricated(self):
        records,metrics=self.fixture()
        for solve in metrics['native_sqp_solves']:solve['compute_us']=None
        summary=summarize_sqp_iterations(records,metrics)
        self.assertIsNone(summary['sqp_iteration_compute_mean_ms'])
        self.assertAlmostEqual(summary['sqp_iteration_wall_mean_ms'],.750)

class ScheduledFailureReporting(unittest.TestCase):
    def test_retains_crashed_process_and_failed_horizon_without_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            jobs=[dict(knots=n,case_id='reach-seed2',method=m,repeat=0)
                  for n,m in ((128,'normal'),(128,'dummy'),(512,'normal'))]
            (root/'launch.json').write_text(json.dumps(dict(plan=dict(jobs=jobs))))
            (root/'status.json').write_text(json.dumps([dict(knots=128,returncode=1),dict(knots=512,returncode=1)]))
            passed=root/'N128/reach-seed2-normal-r0';passed.mkdir(parents=True)
            (passed/'summary.json').write_text(json.dumps(dict(sqp_iteration_compute_mean_ms=8.25)))
            (root/'N128/status.json').write_text(json.dumps([
                dict(case_id='reach-seed2',method='normal',repeat=0,returncode=0),
                dict(case_id='reach-seed2',method='dummy',repeat=0,returncode=-11)]))
            (root/'N512.stderr').write_text('shared setup failure retained here')
            output=root/'table.csv'
            with contextlib.redirect_stdout(io.StringIO()):summarize(SimpleNamespace(input=root,output=output))
            with output.open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),3)
            self.assertEqual(rows[0]['sqp_iteration_compute_mean_ms'],'8.25')
            self.assertEqual(rows[0]['error'],'')
            self.assertEqual(rows[1]['process_returncode'],'-11')
            for row in rows[1:]:
                self.assertEqual(row['summary_present'],'False')
                self.assertEqual(row['sqp_iteration_compute_mean_ms'],'')
                self.assertIn('No summary',row['error'])
            self.assertEqual(rows[2]['horizon_returncode'],'1')
            self.assertEqual(rows[2]['stderr_path'],str(root/'N512.stderr'))

if __name__=='__main__':unittest.main()
