"""Select the reviewed private N64 build in existing simulator child workers."""
import importlib.util
import os
from pathlib import Path
import sys
import traceback

if os.environ.get('GATO_FAULT_NATIVE_MODULE'):
    try:
        source=Path(os.environ['GATO_FAULT_REPO']).resolve();module=Path(os.environ['GATO_FAULT_NATIVE_MODULE']).resolve()
        sys.path[:0]=[str(source/'python'),str(source/'tiago_src')]
        import bsqp
        if {Path(p).resolve() for p in bsqp.__path__}!={source/'python/bsqp'}:
            raise RuntimeError('Fault bootstrap loaded another source package')
        name='bsqp.bsqpN64_tiago_right'
        spec=importlib.util.spec_from_file_location(name,module);native=importlib.util.module_from_spec(spec)
        sys.modules[name]=native;spec.loader.exec_module(native)
    except BaseException:
        traceback.print_exc();sys.stderr.flush();os._exit(86)
