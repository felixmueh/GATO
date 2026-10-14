#!/usr/bin/env python3
"""Bind the unchanged example to its pinned Python files and exact N64 extension."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--module', type=Path, required=True)
p.add_argument('--receipt', type=Path, required=True)
p.add_argument('--verify-only', action='store_true')
p.add_argument('arguments', nargs=argparse.REMAINDER)
a = p.parse_args()
source, module = a.source.resolve(), a.module.resolve()
sys.path[:0] = [str(source/'python'), str(source/'tiago_src')]
import bsqp
bsqp.__path__ = [str(source/'python/bsqp')]
import gato_tiago
controller = importlib.import_module('gato_tiago.tiago_mpc_controller')
interface = importlib.import_module('bsqp.interface')
for loaded, expected in ((gato_tiago, source/'tiago_src/gato_tiago/__init__.py'),
                         (controller, source/'tiago_src/gato_tiago/tiago_mpc_controller.py'),
                         (interface, source/'python/bsqp/interface.py')):
    if Path(loaded.__file__).resolve() != expected:
        raise RuntimeError('Unexpected imported Python runtime: '+str(loaded.__file__))
name = 'bsqp.bsqpN64_tiago_right'
spec = importlib.util.spec_from_file_location(name, module)
native = importlib.util.module_from_spec(spec)
sys.modules[name] = native
spec.loader.exec_module(native)
if Path(native.__file__).resolve() != module or native.KNOT_POINTS != 64:
    raise RuntimeError('Unexpected loaded native module')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
receipt = dict(gato_tiago_path=gato_tiago.__file__, controller_path=controller.__file__,
    controller_sha256=sha(controller.__file__), bsqp_package_paths=list(bsqp.__path__),
    interface_path=interface.__file__, interface_sha256=sha(interface.__file__),
    native_module_path=native.__file__, native_module_sha256=sha(native.__file__),
    verified_before_simulation=True)
a.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
if not a.verify_only:
    arguments = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    sys.argv = [str(source/'tiago_examples/tiago_fig8_tracking.py'), *arguments]
    runpy.run_path(sys.argv[0], run_name='__main__')
