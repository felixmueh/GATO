"""Preload the selected build for unchanged task-test subprocesses only."""
import importlib.util
import os
from pathlib import Path
import sys
import traceback

if os.environ.get('GATO_TASK_NATIVE_MODULE'):
    try:
        source = Path(os.environ['GATO_TASK_SOURCE']).resolve()
        module = Path(os.environ['GATO_TASK_NATIVE_MODULE']).resolve()
        sys.path[:0] = [str(source / 'python'), str(source / 'tiago_src')]
        import bsqp
        if Path(bsqp.__file__).resolve() != source / 'python/bsqp/__init__.py':
            raise RuntimeError('Task bootstrap imported a different BSQP package')
        name = 'bsqp.bsqpN32_tiago_right'
        spec = importlib.util.spec_from_file_location(name, module)
        native = importlib.util.module_from_spec(spec)
        sys.modules[name] = native
        spec.loader.exec_module(native)
    except BaseException:
        # Python otherwise ignores sitecustomize exceptions and might load an
        # unrelated installed extension. A missing selected build is a failure.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(86)
