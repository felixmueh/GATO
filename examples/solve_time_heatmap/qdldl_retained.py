"""Specific retained Jetson N32/B1 cohort; no new or synthesized measurements."""
import hashlib
import json
import math
from pathlib import Path

RELATIVE_ROOT = '_retained/n32-b1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def inspect(root):
    root=Path(root)
    manifest=json.loads((root/'results.json').read_text())
    config=manifest['config']
    expected=dict(plant='tiago_right',linsys_solver='QDLDL',protocol='wall-time',
        knots=[32],batches=[1],repeats=2,max_sqp_iters=1,updates=300,
        initialization='single',max_solve_ms=0,timeout=180,seed=17092026,
        sim_dt=.001,sim_step=.01,save_plans=False)
    if any(config.get(key)!=value for key,value in expected.items()):
        raise ValueError('Retained N32/B1 protocol differs from the frozen matrix')
    if not math.isclose(config['horizon_time'],.93,rel_tol=0,abs_tol=1e-12):
        raise ValueError('Retained N32/B1 knot timing differs')
    hashes={'results.json':sha(root/'results.json')}
    modules=set()
    for repeat in (0,1):
        folder=root/f'N32_B1_R{repeat}'
        receipt=json.loads((folder/'results.json').read_text())
        if (receipt['N'],receipt['batch_size'],receipt['repeat'],receipt['status'],receipt['samples'],receipt['linsys_solver']) != (32,1,repeat,'completed',300,'QDLDL'):
            raise ValueError('Retained N32/B1 must contain both completed300-update repeats')
        if manifest['module_sha256'].get(receipt['module']) != receipt['module_sha256']:
            raise ValueError('Retained module provenance differs between manifest and receipt')
        modules.add(receipt['module_sha256'])
        with (folder/'solves.jsonl').open() as stream:
            measured=[json.loads(line) for line in stream if line.strip()]
        if len(measured)!=300 or any(row.get('warmup') for row in measured):
            raise ValueError('Retained N32/B1 needs300 measured stream rows per repeat')
        for name in ('results.json','solves.jsonl','samples.npz'):
            path=folder/name;hashes[str(path.relative_to(root))]=sha(path)
    if len(modules)!=1:raise ValueError('Retained repeat module identities differ')
    return dict(relative_root=RELATIVE_ROOT,source_commit=manifest['git_head'],
                module_sha256=modules.pop(),files_sha256=hashes)

def validate_copy(raw,record):
    root=Path(raw)/record['relative_root']
    for name,expected in record['files_sha256'].items():
        if sha(root/name)!=expected:raise ValueError('Retained cohort artifact changed: '+name)
    return root
