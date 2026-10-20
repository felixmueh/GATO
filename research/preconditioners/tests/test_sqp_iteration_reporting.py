"""CPU-only checks for native-interval accounting and zero-PCG classification."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from upstream_reach import summarize_sqp_iterations

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

if __name__=='__main__':unittest.main()
