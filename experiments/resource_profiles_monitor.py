#!/usr/bin/env python3
"""Run one resource/timing plan with passive host monitoring outside timers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from recording import sha,write_json

ROOT=Path(__file__).resolve().parents[1]


def commands(plan,machine,python):
    runner=[python,str(ROOT/'experiments/run.py'),'--plan',str(Path(plan).resolve()),'--machine',machine]
    monitor=(['tegrastats','--interval','1000'] if machine=='jetson' else
        ['nvidia-smi','--query-gpu=timestamp,temperature.gpu,utilization.gpu,clocks.sm,clocks.mem,power.draw,memory.used',
         '--format=csv','--loop=1'])
    return runner,monitor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--machine',choices=('rtx5060','jetson'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    runner,monitor=commands(args.plan,args.machine,args.python)
    if args.dry_run:
        print(json.dumps(dict(runner=runner,monitor=monitor,output=str(args.output)),indent=2));return 0
    args.output.mkdir(parents=True,exist_ok=False)
    receipt=dict(machine=args.machine,plan_sha256=sha(args.plan),runner=runner,monitor=monitor,
        script_sha256=sha(__file__),started_unix_s=time.time(),status='started',
        scope='Passive 1Hz host temperature/clock/power samples outside worker timers; no thermal steady-state claim')
    write_json(args.output/'receipt.json',receipt)
    process=None;code=1
    try:
        with (args.output/'conditions.txt').open('w') as samples,(args.output/'monitor-stderr.txt').open('w') as errors:
            process=subprocess.Popen(monitor,stdout=samples,stderr=errors)
            # An immediately missing/failing monitor is a shared setup failure.
            try:monitor_code=process.wait(timeout=.2)
            except subprocess.TimeoutExpired:monitor_code=None
            if monitor_code is not None:raise RuntimeError('Passive monitor exited before plan launch')
            code=subprocess.call(runner,cwd=ROOT)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        receipt.update(ended_unix_s=time.time(),runner_exit_code=code,
            monitor_exit_code=None if process is None else process.returncode,status='completed' if code==0 else 'failed')
        write_json(args.output/'receipt.json',receipt)
    return code


if __name__=='__main__':raise SystemExit(main())
