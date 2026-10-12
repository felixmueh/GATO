#!/usr/bin/env python3
"""Explicit front-square successor using the existing compact demo collector."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
PROPOSAL=Path(__file__).with_name('proposal_front.json')
RUNNER=Path('/workspace/GATO-experiment-runner-simple')
RUNNER_PIN='f29864689de97fabd149d5b824d55941811f8d85'
RUNTIME_PIN='fa6f6bede286c122cc41ac65f42a583da84290e0'
RUNTIME_SOURCE=RUNNER/'.experiment-work/gtx1070-zero-pcg-20261011T214000Z/cache/sources'/RUNTIME_PIN
BUILD=RUNNER/'.experiment-work/gtx1070-zero-pcg-20261011T214000Z/cache/builds/90c1111817d3792a4dbb1c2f7dec73d5fd5b650389cec47579c011c5efbdb7b4'
N32_MODULE=BUILD/'modules/bsqpN32_tiago_right.cpython-310-x86_64-linux-gnu.so'
N32_SHA256='ddd78006da6ea6aa841f283b9db810f4ab37785fc4572732cdfb3a50e26c3b5d'
HELPERS=('experiments/adapters/demo.py','experiments/adapters/task_recording.py','experiments/recording.py')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--preflight-output',type=Path);parser.add_argument('--preview',action='store_true')
    parser.add_argument('--proposal',type=Path,default=PROPOSAL,help='Explicit separately frozen scientific successor input')
    args=parser.parse_args()
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    helper_hashes={}
    for path in HELPERS:
        expected=subprocess.check_output(['git','-C',str(RUNNER),'show',RUNNER_PIN+':'+path])
        if (RUNNER/path).read_bytes()!=expected:raise ValueError('Reviewed recording helper changed: '+path)
        helper_hashes[path]=sha(RUNNER/path)
    if sha(N32_MODULE)!=N32_SHA256:raise ValueError('Native module differs from accepted v5 module')
    sys.path.insert(0,str(RUNNER/'experiments'))
    from adapters.demo import load_collector
    worker=load_collector(Path(__file__).with_name('run_demo.py'))
    proposal=args.proposal.resolve();worker.INPUT=proposal
    identity=worker.runtime_identity(RUNTIME_SOURCE,RUNTIME_PIN)
    provenance=dict(collector_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        collector_driver_sha256=sha(__file__),collector_source_sha256=sha(Path(__file__).with_name('run_demo.py')),
        scientific_input=str(proposal),scientific_input_sha256=sha(proposal),runner_pin=RUNNER_PIN,recording_helper_sha256=helper_hashes,
        runtime_identity=identity,module=dict(path=str(N32_MODULE),sha256=N32_SHA256),
        scope='One explicit canonical preflight or one square-goal modeB simulation; no Gazebo/robot, no retry or solver/controller changes')
    if args.preview:
        protocol=worker.resolve_preflight('internal') if args.preflight else worker.resolve('internal','square')
        print(json.dumps(dict(provenance=provenance,protocol=protocol),indent=2));return 0
    if os.environ.get('GATO_MACHINE')!='gtx1070':raise ValueError('Set GATO_MACHINE=gtx1070 for this explicit cohort')
    if args.output is None:raise ValueError('Supply a fresh explicit raw output path')
    if not args.preflight and args.preflight_output is None:raise ValueError('Supply the matching successful canonical attempt')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    launch=args.output.with_suffix('.launch.json')
    if args.output.exists() or launch.exists():raise ValueError('Fresh output/launch required; previous attempts retained')
    launch.write_text(json.dumps(provenance,indent=2)+'\n')
    sys.argv=[str(Path(__file__).with_name('run_demo.py')),'--backend','internal','--tasks','square',
        '--source',str(RUNTIME_SOURCE),'--module',str(N32_MODULE),'--output',str(args.output)]
    if args.preflight:sys.argv+=['--preflight']
    else:sys.argv+=['--preflight-output',str(args.preflight_output)]
    code=worker.main()
    return code


if __name__=='__main__':raise SystemExit(main())
