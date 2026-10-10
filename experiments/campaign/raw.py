"""Durable raw campaign events; no numerical arrays are filtered or coerced."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class RawRecorder:
    """Observer called only outside the solve timer; every event has its own NPZ.

    JSON metadata maps arrays to NPZ keys, including NaNs/infinities and integer
    flags in native telemetry. One event per file avoids object arrays/pickle.
    A completed write is flushed before the next solve starts. SIGKILL can leave
    only its currently in-flight event incomplete; earlier files stay readable.
    """
    def __init__(self, output):
        self.output = Path(output)
        (self.output / 'raw').mkdir(parents=True, exist_ok=True)
        self.events = []
        self.started = time.perf_counter()

    def __call__(self, kind, record):
        arrays = {}

        def pack(value, key):
            if isinstance(value, np.ndarray):
                arrays[key] = value
                return {'npz_key': key, 'shape': list(value.shape),
                        'dtype': str(value.dtype),
                        'nonfinite_count': int(np.count_nonzero(~np.isfinite(value)))
                        if np.issubdtype(value.dtype, np.number) else None}
            if isinstance(value, dict):
                return {str(k): pack(v, key + '__' + str(k)) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [pack(v, key + '__' + str(i)) for i, v in enumerate(value)]
            # Retain nonfinite scalar bytes too, while JSON uses explicit null.
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                arrays[key] = np.asarray(value)
                return {'npz_key': key, 'nonfinite_scalar': str(value)}
            return json_safe(value)

        index = len(self.events)
        metadata = pack(record, 'record')
        relative = f'raw/{index:07d}-{kind}.npz'
        target = self.output / relative
        with target.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        target.with_suffix('.npz.tmp').replace(target)
        event = {'event_index': index, 'event': kind,
                 'recorder_wall_elapsed_s': time.perf_counter() - self.started,
                 'raw_file': relative, 'sha256': sha(target), 'record': metadata}
        with (self.output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
            stream.flush()
        self.events.append(event)

    def manifest(self):
        # Runner logs/receipt continue changing until its worker has exited;
        # only hash artifacts owned and finalized by this worker here.
        files = list((self.output/'raw').glob('*.npz'))
        files += [self.output/name for name in ('events.jsonl', 'execution.npz',
                  'protocol.json', 'provenance.json', 'controller_stats.json', 'metrics.json',
                  'screening.json', 'readiness.json', 'build_receipt.json')
                  if (self.output/name).is_file()]
        files.sort()
        write_json(self.output / 'SHA256SUMS.json',
                   {str(p.relative_to(self.output)): sha(p) for p in files})
