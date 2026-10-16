#!/usr/bin/env python3
"""Freeze the existing30 saved-state tasks and explicit new protocol; CPU only."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COHORT = 'reaching-long-horizon-terminal10-20261016'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    prior = json.loads((ROOT/'experiments/reaching_sqp5_5060/inputs/TERMINAL10_FLAT.json').read_text())
    tasks = HERE/'inputs/tasks.json'
    rows = json.loads(tasks.read_text())
    if len(rows)!=30 or len({r['case_id'] for r in rows})!=30:
        raise ValueError('Expected exactly30 frozen saved-state inputs')
    for index in range(0,30,2):
        a,b = rows[index:index+2]
        if (a['pair_id']!=b['pair_id'] or a['start_type']!='original' or b['start_type']!='trap'):
            raise ValueError('Expected interleaved original/trap starts')
    canonical = HERE/'inputs/canonical.json'
    source_canonical = ROOT/'experiments/reaching_sqp5_5060/inputs/canonical.json'
    if canonical.exists() and sha(canonical)!=sha(source_canonical):
        raise ValueError('Existing canonical differs; preserve it')
    shutil.copyfile(source_canonical,canonical)
    protocol = {key:prior[key] for key in ('solver_params','limit_penalty','arrival_policy','arrival_orientation_gate',
                'orientation_required_for_arrival','reset_rho_each_solve','position_tolerance_m','tool_axis_tolerance_deg',
                'velocity_l1_tolerance_rad_s','goal_dwell_s','final_approach','batch','timing_mode','offline_timing',
                'knot_dt_s','sim_dt_s','timeout_s','joint_cost_mode')}
    protocol.update(id=COHORT,arm='FLAT_N256_TERMINAL10_SQP1000',N=256,feedback_s=.01,canonical_timeout_s=8.,
        tasks_sha256=sha(tasks),canonical_sha256=sha(canonical),prediction_duration_s=7.65,
        canonical_policy='Single new8s canonical: rough arrival, physical pass, finite/error-free complete execution and geometry/motion checks must allpass; failure holds both starttypes.',
        initialization='Repeat full actual14-state at all256knots, zero controls, reset dual/rho once thenretain; trapstate velocity is preserved.',
        no_goal_witness=True,no_posture_bias=True,solver_arithmetic_unchanged=True,
        shared_memory_optin_commit='6d037790cfdbeefa6a13733a7ea7da4eb83c969f',
        physical_checks='RawURDF q/v/u sampled at executed RK4 states/controlstarts; strict>0 firstbreach plus historical1e-6 tolerancephysicalpass.',
        recording='Dense executed q/v/u/t and typedwork, initialcoldseed/reference, feedbackwork/errors/limits; no routinehorizons; invalidoptimizedplanretained.')
    protocol['solver_params'] = dict(prior['solver_params'],max_sqp_iters=1000,N_cost=800.)
    names = [n for n in prior['source_sha256'] if not n.startswith('experiments/')]
    names += ['gato/bsqp/pcg_launch_resources.cuh','experiments/reaching_long_horizon_terminal10/run_case.py']
    protocol['source_sha256'] = {n:sha(ROOT/n) for n in names}
    (HERE/'inputs/protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    frozen={p.name:sha(p) for p in sorted((HERE/'inputs').iterdir()) if p.is_file() and p.name!='FREEZE_RECEIPT.json'}
    (HERE/'inputs/FREEZE_RECEIPT.json').write_text(json.dumps(dict(cohort=COHORT,files=frozen,scientific_cases=30,
        start_types=['original','trap'],new_canonicals=1,source='Saved raw base flat10ms state/goal data only; no goalwitness'),indent=2)+'\n')
    print(json.dumps(dict(status='frozen',files=frozen),indent=2))

if __name__=='__main__':
    main()
