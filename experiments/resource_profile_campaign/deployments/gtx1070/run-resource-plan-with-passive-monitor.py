"""Run the pinned plan once and retain passive 1 Hz nvidia-smi samples."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.plan) != args.plan_sha256:
        raise ValueError('Pinned plan differs')
    args.output.mkdir(parents=True, exist_ok=False)
    runner = ['/opt/gato-venv/bin/python3', '-B', 'experiments/run.py',
              '--plan', str(args.plan.resolve()), '--machine', 'gtx1070']
    monitor = ['nvidia-smi',
               '--query-gpu=timestamp,temperature.gpu,utilization.gpu,clocks.sm,clocks.mem,power.draw,memory.used',
               '--format=csv', '--loop=1']
    receipt = dict(plan=dict(path=str(args.plan.resolve()), sha256=args.plan_sha256),
                   runner=runner, monitor=monitor, script_sha256=sha(__file__),
                   scope='Passive NVML queries once per second across the queue. '
                         'Separate CSV, outside worker timers; observer overhead is not estimated. '
                         'No clock/power setting changes, CUDA context, kernel or explicit synchronization added.',
                   started_unix_s=time.time(), status='running')
    receipt_path = args.output / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    code = None
    process = None
    try:
        with (args.output / 'gpu.csv').open('w') as samples, (args.output / 'monitor-stderr.txt').open('w') as errors:
            process = subprocess.Popen(monitor, stdout=samples, stderr=errors)
            try:
                code = subprocess.call(runner, cwd='/workspace/GATO-eval-resource-profiles')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    finally:
        receipt.update(ended_unix_s=time.time(), runner_exit_code=code,
                       monitor_exit_code=None if process is None else process.returncode,
                       status='completed' if code == 0 else 'failed_or_interrupted')
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return code if code is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())
