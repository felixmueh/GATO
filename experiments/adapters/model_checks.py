#!/usr/bin/env python3
"""Run the existing bounded model/pose/cost probe without changing its science."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recording import sha, write_json


def classification(summary, returncode):
    # The frozen worker exits one for a completed negative qualification too.
    completed = ((summary.get('status'), returncode) in (('PASS', 0), ('FAIL', 1))
                 and summary.get('native_calls') == 351)
    return dict(solver_numerical_pass=summary.get('status') == 'PASS' if completed else None,
        outcome='checks_completed' if completed else 'execution_failed',
        execution_error=None if completed else f'Probe did not complete its frozen 351 calls (exit {returncode})')


def run(source, build, output):
    source, build, output = (Path(p).resolve() for p in (source, build, output))
    output.mkdir(parents=True, exist_ok=True)
    metrics = dict(campaign_id='Q02', name='Model, orientation and cost consistency checks',
        research_question='RQ-model', timing_mode='A', execution_backend='internal',
        task_success=None, physical_pass=None, solver_numerical_pass=None,
        execution_error=None, outcome='execution_failed')
    try:
        modules = list((build / 'modules').glob('q02_probe*.so'))
        if len(modules) != 1:
            raise ValueError('Expected one exact Q02 probe extension')
        command = [sys.executable, '-B', str(source / 'experiments/campaign/q02/run.py'),
                   '--module', str(modules[0]), '--output', str(output / 'probe-raw')]
        write_json(output / 'provenance.json', dict(command=command, source_root=str(source),
            module_sha256=sha(modules[0]), adapter_sha256=sha(__file__),
            source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
            protocol_sha256=sha(source / 'experiments/campaign/q02/protocol.json'),
            scope='Original one-pass protocol and original tolerances, including previously observed negatives'))
        with (output / 'probe.log').open('w') as log:
            result = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=False)
        summary_path = output / 'probe-raw/summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if not (output / 'probe-raw/raw_manifest.json').is_file():
            raise RuntimeError('Probe did not finalize its raw manifest; retain partial evidence')
        metrics.update(classification(summary, result.returncode), original_summary=summary,
                       original_returncode=result.returncode)
    except BaseException:
        metrics['execution_error'] = traceback.format_exc()
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'SHA256SUMS.json', {str(p.relative_to(output)): sha(p) for p in output.rglob('*')
        if p.is_file() and p.name != 'SHA256SUMS.json'})
    return 1 if metrics['execution_error'] else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source, args.build, args.output))
