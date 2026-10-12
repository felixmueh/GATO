#!/usr/bin/env python3
"""Run the exact pinned force worker with a recording-only injected recorder."""
import argparse
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recording
from adapters.force_recording import ForceRecorder, FullForceRecorder


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--worker', choices=('run_task', 'run_native'), required=True)
    parser.add_argument('--full-io', action='store_true')
    args, worker_args = parser.parse_known_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / 'python'))
    worker = importlib.import_module('experiments.force_adapter.' + args.worker)
    expected = source / 'experiments' / 'force_adapter' / (args.worker + '.py')
    if Path(worker.__file__).resolve() != expected:
        raise ValueError('Force worker imported from a different source checkout')
    # The pinned worker already injects this recorder into run_case; no controller,
    # estimator, timing, native solver or task gate is replaced.
    recording.RawRecorder = FullForceRecorder if args.worker == 'run_native' or args.full_io else ForceRecorder
    sys.argv = [str(expected), *worker_args]
    return worker.main()


if __name__ == '__main__':
    raise SystemExit(main())
