#!/usr/bin/env python3
"""Run original BA/BB canonical arms, holding only execution/setup failures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from recording import sha, write_json


def run(source, build, inputs, output, study):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    result = dict(study=study, policy='H100 execution-only readiness; task/physical negatives are descriptive',
        task_success=None, physical_pass=None, execution_error=None,
        execution_readiness_pass=False, arms=[], helper_sha256=sha(__file__))
    arms = [(1,'default'),(16,'default')] + ([(1,'nominal16x')] if study=='BA' else [])
    try:
        for batch, arm in arms:
            folder = output/(f'B{batch}' if arm=='default' else 'nominal16x-B1')
            command = [sys.executable,str(Path(__file__).parent/'adapters'/f'{study.lower()}.py'),
                '--source',str(source),'--build',str(build)]
            if study=='BB':
                command += ['--pool',str(inputs),'--protocol',str(Path(inputs)/'bb_protocol.json'),
                    '--preflight-input',str(Path(inputs)/'bb_preflight.json')]
            else:
                command += ['--inputs',str(inputs),'--preflight','--arm',arm]
            command += ['--batch',str(batch),'--output',str(folder)]
            code = subprocess.run(command).returncode
            path = folder/'metrics.json'
            metrics = json.loads(path.read_text())
            item = dict(batch=batch,arm=arm,returncode=code,metrics_path=str(path),
                metrics_sha256=sha(path),task_success=metrics.get('task_success'),
                physical_pass=metrics.get('physical_pass'),complete=metrics.get('complete'),
                finite=metrics.get('physical',{}).get('finite'),
                execution_error=metrics.get('execution_error'),
                failure=metrics.get('failure'),controller_failure=metrics.get('controller_failure'),
                selection_failure=metrics.get('selection_failure'))
            result['arms'].append(item)
            # Child exit1 also represents ordinary rough/physical negatives.
            # Retain those labels; never turn unknown/missing metrics into a pass.
            if (code not in (0,1) or 'execution_error' not in metrics
                or metrics['execution_error'] is not None or item['complete'] is not True
                or item['finite'] is not True
                or any(item[key] is not None for key in ('failure','controller_failure','selection_failure'))
                or type(item['task_success']) is not bool or type(item['physical_pass']) is not bool):
                raise RuntimeError(f'{study} {arm} B{batch} execution/setup/finite check failed; see {path}')
        result.update(execution_readiness_pass=True,
            task_success=all(a['task_success'] for a in result['arms']),
            physical_pass=all(a['physical_pass'] for a in result['arms']))
    except Exception as error:
        result['execution_error'] = f'{type(error).__name__}: {error}'
    write_json(output/'metrics.json',result)
    return int(not result['execution_readiness_pass'])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['source','build','inputs','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--study',choices=['BA','BB'],required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source,args.build,args.inputs,args.output,args.study))
