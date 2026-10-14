"""CPU-only adapters to the reviewed recorder/collector; never import native here."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


# Dependencies supplied by the simple adapter; scheduling/request functions unchanged.
F = E = RAW = None
from recording import sha, write_json
DURATION_APPROVAL_SHA256 = 'aa96ce80280b1580670005ec42dacac41ff5e46ed140e59e805b41e7fb6888a2'
BUDGETS = ((1, 100), (5, 100), (10, 800))
STAGES = ('setup_kkt', 'schur_system', 'pcg', 'compute_dz', 'residual_count_D2H',
          'merit_candidates', 'line_search_and_stats_publication')


def cases(sustained_duration_s=120):
    if isinstance(sustained_duration_s,bool) or not isinstance(sustained_duration_s,(int,float)) or not math.isfinite(sustained_duration_s) or sustained_duration_s<=0:
        raise ValueError("Positive finite proposed sustained wall interval required")
    query = [dict(study='F04', arm='query', N=N, B=B, stage_timing=False)
             for N in (16, 32, 64, 128) for B in (1, 16)]
    timing = []
    for SQP, PCG in BUDGETS:
        for B in (1, 16):
            base = dict(study='F05', N=32, B=B, SQP=SQP, PCG=PCG)
            timing.extend(dict(base, arm='first_response', replicate=i, stage_timing=False) for i in range(3))
            timing.append(dict(base, arm='profile', stage_timing=True))
            timing.append(dict(base, arm='sustained', duration_s=sustained_duration_s, stage_timing=False))
    return query + timing


def stage_extension_cases():
    """Requested bounded GTX1070 N128 extension; historical38cases unchanged."""
    return [dict(study='F05',arm='profile',N=N,B=1,SQP=SQP,PCG=100,stage_timing=True)
            for N,SQP in ((128,5),(128,1),(32,5))]


def schedule(B, i):
    if type(i) is not int or i < 0 or B not in (1, 16): raise ValueError('Invalid schedule')
    return (i // 16 % 6, [i % 16]) if B == 1 else (i % 6, list(range(16)))


def stage_layout(cap):
    if type(cap) is not int or not 0 <= cap <= 10: raise ValueError('Profile cap must be0..10')
    return [('initial_reset_merit_snapshot', None)] + [(name, i) for i in range(cap) for name in STAGES] + [('final_reset_and_merit', None)]


def validate_profile(profile, cap):
    if profile['schema_version'] != 1 or profile['units'] != 'stream_elapsed_us' or profile['pcg_times_us_is_placeholder'] is not True:
        raise ValueError('Profile scope/placeholder differs')
    expected = stage_layout(cap); rows = profile['windows']
    if [(r['stage'], r['sqp_iteration']) for r in rows] != expected: raise ValueError('Profile event indices differ')
    for row in rows:
        if row['elapsed_us'] is None:
            if row['unavailable_reason'] != 'not_reached': raise ValueError('Skipped stage needs reason')
        elif not math.isfinite(row['elapsed_us']) or row['elapsed_us'] < 0 or row['unavailable_reason'] is not None:
            raise ValueError('Measured stage metadata differs')
    if rows[0]['elapsed_us'] is None or rows[-1]['elapsed_us'] is None: raise ValueError('Initial/final windows missing')


def bank(path, expected_sha):
    """Read exact F01c frozen bank; no construction/filtering of candidate seeds."""
    import numpy as np
    path = Path(path).resolve()
    if sha(path / 'SHA256SUMS.json') != expected_sha: raise ValueError('F01c bank manifest differs')
    manifest = json.loads((path / 'SHA256SUMS.json').read_text())
    for name, digest in manifest.items():
        target = (path / name).resolve()
        if not target.is_relative_to(path) or sha(target) != digest: raise ValueError('Frozen bank bytes/path differ')
    meta = json.loads((path / 'bank.json').read_text())
    with np.load(path / 'arrays.npz', allow_pickle=False) as saved: arrays = {k: saved[k].copy() for k in saved.files}
    if meta['schema'] != 'F01c-six-input-N32-v1' or len(meta['inputs']) != 6 or meta['approved_budgets'] != [list(x) for x in BUDGETS]:
        raise ValueError('Approved six-input/budget policy differs')
    shapes = dict(starts=(2,14), pools=(2,16,665), references=(6,192), warmup_state=(14,), warmup_pool=(16,665), warmup_reference=(192,))
    for key, shape in shapes.items():
        if arrays[key].shape != shape or arrays[key].dtype != np.dtype('<f4'): raise ValueError('Frozen array shape/dtype differs: ' + key)
    for entry in meta['inputs']:
        start = arrays['starts'][entry['start_index']]
        if not np.array_equal(arrays['pools'][entry['start_index'], :, :14], np.tile(start, (16,1))):
            raise ValueError('Candidate initial states differ')
    return meta, arrays


def request(meta, arrays, B, input_index=0, candidates=None, warmup=False):
    import numpy as np
    candidates = candidates if candidates is not None else ([0] if B == 1 else list(range(16)))
    if len(candidates) != B: raise ValueError('Wrong candidate count')
    entry = meta['inputs'][input_index]
    state = arrays['warmup_state'] if warmup else arrays['starts'][entry['start_index']]
    pool = arrays['warmup_pool'] if warmup else arrays['pools'][entry['start_index']]
    reference = arrays['warmup_reference'] if warmup else arrays['references'][input_index]
    return dict(input_id='unrelated-fraction.025-goal3' if warmup else entry['id'],
        input_index=None if warmup else input_index, candidate_ids=candidates,
        state=np.tile(state, (B,1)), reference=np.tile(reference, (B,1)), warm=pool[candidates].copy())


def bank_f01ab(path, expected_sha, case):
    """Use retained F01ab first cold reaching input, with its unrelatedwarmup."""
    import numpy as np
    path=Path(path).resolve()
    if sha(path/'SHA256SUMS.json')!=expected_sha:raise ValueError('F01ab manifest differs')
    for name,digest in json.loads((path/'SHA256SUMS.json').read_text()).items():
        target=(path/name).resolve()
        if not target.is_relative_to(path) or sha(target)!=digest:raise ValueError('F01ab input differs')
    bank=json.loads((path/'bank.json').read_text())
    if bank['schema']!='held-F01-input-bank-v1' or case['N'] not in bank['N'] or case['SQP'] not in bank['SQP'] or case['PCG'] not in bank['PCG'] or case['B']!=1:
        raise ValueError('F01ab profile dimensions/budgets differ')
    with np.load(path/'arrays.npz',allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
    episode=bank['variants']['full24']['episodes'][0]
    if episode['kind']!='reaching':raise ValueError('Expected first reaching input')
    start=arrays['starts'][episode['start_index']];goal=arrays['goals'][episode['goal_indices'][0]];N=case['N']
    # Exactly the source F01ab cold_seed primitive; no MPC episode/capture.
    pool=E.cold_seed(start,N)[None,:,:]
    warmup_pool=E.cold_seed(arrays['warmup_state'],N)
    meta=dict(preset=bank['presets']['reaching'],N=N,inputs=[dict(id='F01ab-cold-'+episode['id'],start_index=0)])
    payload=dict(starts=start[None,:],pools=pool,references=np.tile(goal,N)[None,:],
                 warmup_state=arrays['warmup_state'],warmup_pool=warmup_pool,warmup_reference=np.tile(arrays['warmup_goal'],N))
    return meta,payload
