#!/usr/bin/env python3
"""Run one build/benchmark in the shared quiet window, retaining its command and clocks."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cwd', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    a.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(command=command, cwd=str(a.cwd.resolve()),
                    start=datetime.now(timezone.utc).isoformat(), state='waiting_for_quiet_window')
    status = a.output / 'command.json'
    status.write_text(json.dumps(metadata, indent=2))
    locks = [open('/workspace/GATO-upstream-contributions/.gpu.lock', 'a'),
             open('/tmp/a2rlab-timing.lock', 'a')]
    deadline = time.monotonic() + 45
    for lock in locks:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    metadata.update(state='quiet_window_unavailable', returncode=75)
                    status.write_text(json.dumps(metadata, indent=2))
                    return 75
                time.sleep(.2)
    metadata.update(state='running', acquired=datetime.now(timezone.utc).isoformat())
    status.write_text(json.dumps(metadata, indent=2))
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', GATO_QUIET_WINDOW='1',
               PYTHONPATH=str(a.cwd.resolve() / 'python'))
    with (a.output / 'gpu-during.csv').open('w') as clocks, \
         (a.output / 'gpu-monitor.stderr').open('w') as monitor_error, \
         (a.output / 'stdout.log').open('w') as stdout, \
         (a.output / 'stderr.log').open('w') as stderr:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used',
                                    '--format=csv', '-lms', '100'], stdout=clocks, stderr=monitor_error)
        try:
            result = subprocess.run(command, cwd=a.cwd, env=env, stdout=stdout, stderr=stderr)
            metadata.update(state='complete' if result.returncode == 0 else 'failed',
                            returncode=result.returncode)
        finally:
            monitor.terminate()
            monitor.wait()
            metadata.update(end=datetime.now(timezone.utc).isoformat())
            status.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
