#!/usr/bin/env python3
"""Run from the extracted prepared transfer directory: python3 start.py MACHINE."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

package = Path(__file__).resolve().parent
inventory = json.loads((package / 'inventory.json').read_text())
bundle = package / 'sources.bundle'
with bundle.open('rb') as stream:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b''):
        digest.update(chunk)
if digest.hexdigest() != inventory['package_sha256']['sources.bundle']:
    raise SystemExit('Source bundle hash mismatch')
checkout = package / 'GATO-experiment-runner'
if not checkout.exists():
    subprocess.run(['git', 'clone', '--branch', inventory['runner_bundle_branch'],
                    str(bundle), str(checkout)], check=True)
if subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip() != inventory['runner_commit']:
    raise SystemExit('Transfer checkout changed; keep its results and use a fresh prepared transfer directory.')
raise SystemExit(subprocess.call([sys.executable, str(checkout / 'experiments/machine_launch.py'),
                                 *sys.argv[1:], '--package', str(package)]))
