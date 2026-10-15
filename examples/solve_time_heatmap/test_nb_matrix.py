"""Bounded CPU checks for fixed protocol and honest failed-cell aggregation."""
import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch
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

    def test_qdldl_preserves_every_frozen_cell_argument(self):
        t=driver.tool()
        for cell in driver.spec()['cells']:
            args=SimpleNamespace(gpu='jetson',cache=Path('/tmp/qdldl-cache'),raw=Path('/tmp/qdldl-raw'),backend='QDLDL')
            qdldl=driver.arguments(t,cell,args)
            args.backend='PCG'
            pcg=driver.arguments(t,cell,args)
            before,after=vars(pcg).copy(),vars(qdldl).copy()
            self.assertEqual(before.pop('backend'),'PCG')
            self.assertEqual(after.pop('backend'),'QDLDL')
            self.assertEqual(before,after)
            with patch.object(t,'match_torch_libraries'),patch.object(t,'sweep') as sweep:
                self.assertEqual(t.main(cell['args']+['--gpu-name','Jetson','--backend','QDLDL']),0)
                sweep.assert_called_once()

    def test_retained_original_bytes_provenance_and_no_pcg_metrics(self):
        import qdldl_retained
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d);retained=raw/qdldl_retained.RELATIVE_ROOT;retained.mkdir(parents=True)
            config=dict(plant='tiago_right',linsys_solver='QDLDL',protocol='wall-time',knots=[32],batches=[1],
                repeats=2,max_sqp_iters=1,updates=300,initialization='single',max_solve_ms=0,timeout=180,
                seed=17092026,sim_dt=.001,sim_step=.01,save_plans=False,horizon_time=.93)
            original=dict(config=config,git_head='original18859',module_sha256={'old-module.so':'original1f66'},
                          resources={'32':dict(kernels=[dict(fits=True)])})
            (retained/'results.json').write_text(json.dumps(original))
            for repeat in (0,1):
                folder=retained/f'N32_B1_R{repeat}';folder.mkdir()
                (folder/'results.json').write_text(json.dumps(dict(N=32,batch_size=1,repeat=repeat,
                    status='completed',samples=300,linsys_solver='QDLDL',module='old-module.so',module_sha256='original1f66')))
                (folder/'samples.npz').write_bytes(b'unmodified-original-bytes')
                sample=dict(native_ms=2.78,sqp_iters=1,pcg_telemetry_complete=False)
                (folder/'solves.jsonl').write_text((json.dumps(sample)+'\n')*300)
            record=qdldl_retained.inspect(retained)
            manifest=dict(cells=driver.spec(),gpu='jetson',source_commit='new-driver',modules={'32':dict(sha256='new-module')},
                          resources={},results=[],retained_n32_b1=record,backend='QDLDL')
            rows,repeats=nb_export.summarize(manifest,raw)
            row=next(r for r in rows if (r['N'],r['B'])==(32,1))
            self.assertEqual((row['source_commit'],row['module_sha256']),('original18859','original1f66'))
            self.assertEqual((row['status'],row['samples']),('completed',600))
            self.assertAlmostEqual(row['mean_ms'],2.78)
            self.assertIsNone(row['pcg_cap_hit_share']);self.assertIsNone(row['pcg_iters_mean'])
            self.assertTrue(all(r['retained_original'] for r in repeats))
            (retained/'N32_B1_R0'/'solves.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                nb_export.summarize(manifest,raw)

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
