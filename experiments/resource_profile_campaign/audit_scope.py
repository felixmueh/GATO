"""Static default-OFF and no-added-sync audit; this is not native qualification."""
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
BASE='9cbf257'
NATIVE=('gato/types.cuh','gato/bsqp/bsqp.cuh','python/bindings.cu')


def disabled(source):
    lines=[]; owned=[]
    for line in source.splitlines():
        stripped=line.lstrip()
        if stripped.startswith(('#if ', '#ifdef ', '#ifndef ')):
            ours='GATO_EXPERIMENT_STAGE_TIMING' in line or 'GATO_EXPERIMENT_RESOURCE_QUERY' in line
            owned.append(ours or (owned[-1] if owned else False))
            if not owned[-1]:lines.append(line)
        elif stripped.startswith('#endif'):
            active=owned.pop()
            if not active:lines.append(line)
        elif stripped.startswith(('#else', '#elif')):
            if not any(owned):lines.append(line)
        elif not any(owned):lines.append(line)
    if owned:raise ValueError('Unbalanced preprocessor scopes')
    return re.sub(r'\s+','', '\n'.join(lines))


def audit():
    checks={}
    for name in NATIVE:
        baseline=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT,text=True)
        current=(ROOT/name).read_text()
        checks[name+'_default_OFF_same']=disabled(current)==disabled(baseline)
        for sync in ('cudaStreamSynchronize(', 'cudaDeviceSynchronize(', 'cudaEventSynchronize('):
            checks[name+'_'+sync+'count_same']=current.count(sync)==baseline.count(sync)
    helper=(ROOT/'gato/experiment_stage_timing.cuh').read_text()
    checks['helper_no_sync_or_kernel_or_device_buffer']=not any(x in helper for x in ('Synchronize(', 'cudaMalloc(', 'cudaMemset(', 'cudaMemcpy(', '<<<','__syncthreads','atomic'))
    binding=(ROOT/'python/bindings.cu').read_text().split('// Queries compiled kernel pointers only.')[1].split('PYBIND11_MODULE')[0]
    checks['query_no_launch_sync_attribute_or_allocation']=not any(x in binding for x in ('<<<','Synchronize(', 'cudaFuncSetAttribute(', 'cudaMalloc(', 'PyBSQP<'))
    bsqp=(ROOT/'gato/bsqp/bsqp.cuh').read_text()
    checks['read_after_chrono_and_existing_bookkeeping']=bsqp.index('completed_windows()')>bsqp.index('auto sqp_end_time') and bsqp.index('completed_windows()')>bsqp.index('memset(h_sqp_iters_B_')
    checks['existing_PCG_zero_placeholder_preserved']='pcg_stats.solve_time_us = 0;' in bsqp
    if not all(checks.values()):raise AssertionError([k for k,v in checks.items() if not v])
    return dict(status='PASS_CPU_STATIC_ONLY',baseline=BASE,checks=checks,explicit_new_synchronizations=0,
                explicit_new_device_arrays=0,native_compile_and_timing='NOT_RUN_HELD')


if __name__=='__main__':print(json.dumps(audit(),indent=2))
