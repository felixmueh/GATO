"""Bounded CPU checks for fixed protocol and honest failed-cell aggregation."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import nb_matrix as driver
import nb_export

class MatrixContract(unittest.TestCase):
    def test_fixed_common_cells_and_smoke_are_original_cells(self):
        s=driver.spec();self.assertEqual(len(s['cells']),30)
        self.assertEqual([(c['N'],c['B']) for c in driver.ordered_cells(s)[:2]],[(32,1),(512,1)])
        t=driver.tool();args=SimpleNamespace(gpu='gtx1070',cache=Path('/tmp/cache'),raw=Path('/tmp/raw'))
        for c in s['cells']:
            a=driver.arguments(t,c,args)
            self.assertEqual((a.updates,a.repeats,a.max_sqp_iters,a.timeout,a.initialization,a.max_solve_ms),(300,2,1,180,'single',0))
            self.assertAlmostEqual(a.horizon_time/(c['N']-1),.03)

    def test_partial_timeout_samples_not_pooled_with_pass(self):
        s=driver.spec();c=s['cells'][0]
        m=dict(cells=s,gpu='gtx1070',source_commit='frozen',modules={},resources={'16':dict(kernels=[dict(fits=True)],multiprocessors=15)},results=[])
        with tempfile.TemporaryDirectory() as d:
            for repeat,status,count in [(0,'completed',300),(1,'error',9)]:
                folder=Path(d)/f'N16_B1_R{repeat}';folder.mkdir()
                sample=dict(native_ms=2.,sqp_iters=1,pcg_telemetry_complete=True,pcg_lane_sqp_slots=1,pcg_iters_sum=120,pcg_cap_count=1)
                (folder/'solves.jsonl').write_text(''.join(json.dumps(sample)+'\n' for _ in range(count)))
                m['results'].append(dict(N=16,batch_size=1,repeat=repeat,status=status,samples=count if repeat==0 else 0))
            rows,repeats=nb_export.summarize(m,d)
            self.assertEqual(rows[0]['status'],'incomplete');self.assertIsNone(rows[0]['mean_ms'])
            self.assertEqual(rows[0]['samples'],309)
            self.assertEqual(repeats[1]['retained_timing_samples'],9)

    def test_completed_stats_and_pcg_slots_include_zero_iterations(self):
        s=driver.spec();m=dict(cells=s,gpu='gtx1070',source_commit='frozen',modules={},resources={'16':dict(kernels=[dict(fits=True)],multiprocessors=15)},results=[])
        with tempfile.TemporaryDirectory() as d:
            for repeat in [0,1]:
                p=Path(d)/f'N16_B1_R{repeat}';p.mkdir()
                samples=[dict(native_ms=1.+repeat,sqp_iters=1,pcg_telemetry_complete=True,pcg_lane_sqp_slots=2,pcg_iters_sum=120,pcg_cap_count=1) for _ in range(300)]
                (p/'solves.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples))
                m['results'].append(dict(N=16,batch_size=1,repeat=repeat,status='completed',samples=300))
            rows,_=nb_export.summarize(m,d);r=rows[0]
            self.assertEqual(r['status'],'completed');self.assertEqual(r['samples'],600)
            self.assertEqual(r['mean_ms'],1.5);self.assertEqual(r['sd_ms'],.5)
            self.assertEqual(r['pcg_cap_hit_share'],.5);self.assertEqual(r['pcg_iters_mean'],60.)
            m['resources']['16']['kernels'][0]['fits']=False
            rows,_=nb_export.summarize(m,d);self.assertEqual(rows[0]['status'],'SMEM');self.assertIsNone(rows[0]['mean_ms'])

    def test_zero_checkpoint_preserves_two_streamed_timeout_samples(self):
        m=dict(cells=driver.spec(),gpu='gtx1070',source_commit='frozen',modules={},
               resources={'16':dict(kernels=[dict(fits=True)])},
               results=[dict(N=16,batch_size=1,repeat=0,status='error',samples=0)])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'N16_B1_R0';p.mkdir()
            (p/'solves.jsonl').write_text(json.dumps(dict(native_ms=3.,sqp_iters=1))+'\n'+json.dumps(dict(native_ms=4.,sqp_iters=1))+'\n')
            rows,_=nb_export.summarize(m,d)
            self.assertEqual(rows[0]['samples'],2)
            self.assertEqual(rows[0]['status'],'incomplete')
            self.assertIsNone(rows[0]['mean_ms'])

if __name__=='__main__':unittest.main()
