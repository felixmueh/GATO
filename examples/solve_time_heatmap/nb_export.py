"""CPU-only N x B matrix summaries; partial failures never become completed cells."""
import argparse
import csv
import json
import math
from pathlib import Path

FIELDS=['gpu','gpu_uuid','sm_count','N','B','knot_spacing_s','status','repeats','samples',
        'mean_ms','median_ms','p95_ms','sd_ms','mean_sqp_iters','pcg_cap_hit_share',
        'pcg_iters_mean','shmem_optin_applied','source_commit','module_sha256']

def observations(folder):
    """Use the flushed measured stream, including timeout partials; exclude initialization."""
    path=Path(folder)/'solves.jsonl'
    rows=[]
    if path.exists():
        with path.open() as f:
            for line in f:
                if not line.endswith('\n'):break
                row=json.loads(line)
                if not row.get('warmup'):rows.append(row)
    return rows

def finite(value):return isinstance(value,(int,float)) and math.isfinite(value)

def summarize(manifest,raw):
    import numpy as np
    resource=manifest['resources'].get('16',{})
    meta=manifest.get('gpu_metadata',{})
    if not isinstance(meta,dict):meta={}
    rows=[];repeats=[]
    for cell in manifest['cells']['cells']:
        n,b=cell['N'],cell['B']
        selected=[r for r in manifest['results'] if (r['N'],r['batch_size'])==(n,b)]
        probe=manifest['resources'].get(str(n),{})
        unsupported=bool(probe.get('kernels')) and not all(k['fits'] for k in probe['kernels'])
        complete=(len(selected)==2 and all(r.get('status')=='completed' and r.get('samples')==300 for r in selected))
        status=('SMEM' if unsupported else 'nonfinite' if any(r.get('status')=='unstable_nonfinite' for r in selected)
                else 'completed' if complete else 'incomplete')
        row=dict.fromkeys(FIELDS)
        row.update(gpu=manifest['gpu'],gpu_uuid=meta.get('gpu_uuid',resource.get('gpu_uuid',meta.get('uuid'))),
            sm_count=meta.get('multiprocessors',resource.get('multiprocessors')),N=n,B=b,
            knot_spacing_s=.03,status=status,repeats=len(selected),samples=sum(r.get('samples',0) for r in selected),
            shmem_optin_applied=probe.get('pcg_shared_memory_policy',{}).get('optin_applied'),
            source_commit=manifest['source_commit'],module_sha256=manifest['modules'].get(str(n),{}).get('sha256'))
        all_samples=[]
        retained_samples=0
        for r in selected:
            repeat=r['repeat'];samples=observations(Path(raw)/f'N{n}_B{b}_R{repeat}')
            retained_samples+=len(samples)
            good=[s for s in samples if finite(s.get('native_ms')) and s['native_ms']>0]
            repeats.append(dict(N=n,B=b,repeat=repeat,status=r.get('status'),samples=r.get('samples',0),
                retained_timing_samples=len(good),mean_partial_ms=float(np.mean([s['native_ms'] for s in good])) if good else None,
                error=r.get('error'),pcg_telemetry_complete=r.get('pcg_telemetry_complete')))
            all_samples.extend(good)
        row['samples']=retained_samples
        if complete and status=='completed':
            # Completed statistics require every expected sample; never silently fill a gap.
            if len(all_samples)!=600:row['status']='incomplete'
            else:
                times=np.asarray([s['native_ms'] for s in all_samples]);sqps=[s['sqp_iters'] for s in all_samples]
                row.update(mean_ms=float(times.mean()),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),
                           sd_ms=float(times.std(ddof=0)),mean_sqp_iters=float(np.mean(sqps)))
                if all(s.get('pcg_telemetry_complete') for s in all_samples):
                    slots=sum(s['pcg_lane_sqp_slots'] for s in all_samples)
                    if slots:
                        row['pcg_iters_mean']=sum(s['pcg_iters_sum'] for s in all_samples)/slots
                        row['pcg_cap_hit_share']=sum(s['pcg_cap_count'] for s in all_samples)/slots
        rows.append(row)
    return rows,repeats

def csv_write(path,rows,fields=None):
    with Path(path).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)

def export(raw,output,final=False):
    raw,output=Path(raw).resolve(),Path(output).resolve()
    if output.is_relative_to(raw):raise ValueError('Keep compact export outside raw evidence')
    m=json.loads((raw/'nb-results.json').read_text());output.mkdir(parents=True,exist_ok=True)
    rows,repeats=summarize(m,raw)
    csv_write(output/'cells.csv',rows,FIELDS)
    if repeats:csv_write(output/'repeat_status.csv',repeats)
    (output/'provenance.json').write_text(json.dumps(dict(source_commit=m['source_commit'],cells_sha256=m['cells_sha256'],
        raw=str(raw),gpu_metadata=m.get('gpu_metadata'),resources=m['resources'],modules=m['modules'],
        aggregation='Only cells with two completed300-update repeats and600retained timings have headline stats. Failed/partial samples stay raw and in repeat_status.csv.',
        sd='Population standard deviation over600individual native timings; no independence assumption.',
        pcg='Sum PCG cap hits / actual lane-SQP slots, including zero iterations; unavailable telemetry stays empty.'),indent=2)+'\n')
    if final:render(rows,m,raw,output)
    return rows

def render(rows,m,raw,output):
    import shutil
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    ns=m['cells']['knots'];bs=m['cells']['batches']
    for field,label,name in [('mean_ms','Mean full-batch native solve time (ms)','nb_solve_time'),
                             ('median_ms','Median full-batch native solve time (ms)','nb_median'),
                             ('mean_sqp_iters','Mean actual SQP iterations','nb_sqp')]:
        values=np.full((len(ns),len(bs)),np.nan)
        fig,ax=plt.subplots(figsize=(8,6))
        for r in rows:
            i,j=ns.index(r['N']),bs.index(r['B'])
            if r[field] is not None:values[i,j]=r[field]
        image=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap='viridis')
        for r in rows:
            i,j=ns.index(r['N']),bs.index(r['B'])
            text=f'{r[field]:.3g}' if r[field] is not None else r['status']
            ax.text(j,i,text,ha='center',va='center',fontsize=8,color='black',bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
        ax.set_xticks(range(len(bs)),bs);ax.set_yticks(range(len(ns)),ns);ax.set_xlabel('Batch B');ax.set_ylabel('Knots N')
        ax.set_title(m['gpu']+' · fixed300updates ×2repeats');fig.colorbar(image,ax=ax,label=label)
        fig.tight_layout();fig.savefig(output/(name+'.png'),dpi=130);fig.savefig(output/(name+'.pdf'));plt.close(fig)
    counts={s:sum(r['status']==s for r in rows) for s in ['completed','SMEM','nonfinite','incomplete']}
    report=f'''# TIAGo N × B matrix — {m['gpu']}

Cell outcomes: {counts}. A completed cell requires both planned repeats and all
600 measured updates. No failed/partial samples enter completed headline timing.
All original repeats, streamed partial timings, failures and resource rejections
remain in the raw folder and repeat_status.csv. No retries or solver tuning.

Protocol: N16/32/64/128/256/512 × B1/2/4/8/16, SQP1, PCG120, prediction spacing
30 ms, control interval10 ms, RK4max1 ms. Each of two repeats executes300 measured
updates after one excluded single-solve initialization. Nominal measured reference
timestamps0.01…3.00 s are the same; actual states and PCG work may vary. No measured
wall-budget stop or slow-solve cutoff; hard worker timeout180 s retains partials.

This follows the existing TIAGo tracking tool, not the prior task-parameter workers.
Running/limit weights are scaled by0.03/0.008=3.75; terminal position stays80.
The previous unshifted batch0 plan seeds every lane. Full-batch synchronized native
host solve timings exclude initialization and Python/integration/recording. A
reciprocal solve-time rate is not end-to-end control frequency or batch throughput.
Completed budget does not certify task success, physical safety or full KKT convergence.

Only resource-fitting native modules are built. SMEM is a configured/device
resource result, not a task failure. Conditional shared-memory opt-in and device
UUID/SM counts are retained with the exact module/source identities. Null resource
operands are never launched. The first N32/B1 and N512/B1 grid cells are reused
as readiness observations; no extra measurement attempts are added.

![Full-batch mean native solve times](nb_solve_time.png)

![Full-batch median native solve times](nb_median.png)

![Actual SQP work](nb_sqp.png)

cells.csv is the requested30-row schema. Mean/median/p95/SD use600individual
native timings only for completed cells; SD is population SD, not uncertainty
of independent repeated experiments. PCG cap share divides actual cap-hit count
by all recorded lane-SQP slots, including zero iterations. Missing PCG telemetry
stays empty. Failed-cell headline timing fields remain empty; partial samples are
not dropped from raw. These descriptive comparisons do not establish a causal GPU
effect, guaranteed stability, collision clearance or exact numerical determinism.

Source commit: {m['source_commit']}.
Common cells SHA256: {m['cells_sha256']}.
Raw: {raw}.
Exact probe, module hashes, GPU metadata and aggregation labels: provenance.json.
Archive raw and keep a separate backup before any cleanup. Immutable raw includes
source snapshots, source identity, cells.json, probes/build log and per-repeat
results/samples/solves streams. Export regeneration is CPU-only:
python3 {Path(__file__).resolve()} --raw {raw} --output {output}.
'''
    (output/'REPORT.md').write_text(report)
    import textwrap,re
    text_lines=[]
    for line in report.splitlines():
        if line.startswith('!['):continue
        text_lines.extend(textwrap.wrap(line.lstrip('# '),width=100) or [''])
    with PdfPages(output/'REPORT.pdf') as pdf:
        for start in range(0,len(text_lines),65):
            fig=plt.figure(figsize=(8.27,11.69));block=fig.text(.07,.96,'\n'.join(text_lines[start:start+65]),va='top',fontsize=8,fontfamily='monospace',linespacing=1.4)
            fig.canvas.draw();bounds=block.get_window_extent(fig.canvas.get_renderer());assert bounds.x1<fig.bbox.x1 and bounds.y0>.04*fig.bbox.height
            pdf.savefig(fig)
            if start==0:fig.savefig(output/'report_preview.png',dpi=100)
            plt.close(fig)
        for name in ['nb_solve_time','nb_median','nb_sqp']:
            fig,ax=plt.subplots(figsize=(8.27,8));ax.imshow(plt.imread(output/(name+'.png')));ax.axis('off');pdf.savefig(fig);plt.close(fig)
    shutil.copyfile(Path(__file__),output/'nb_export.py') if Path(__file__).resolve()!=output/'nb_export.py' else None

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.raw,a.output,final=True)
