"""Bounded raw event recording, using NumPy and append-only JSON metadata.

Completed chunks survive interruption. A hard kill may lose the current buffered
chunk (at most max_events, or max_bytes plus one individual event). Call close(),
manifest(), or use a context manager on handled failures to retain that chunk.
Old per-event NPZ recordings remain readable without rewriting their evidence.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


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
    def __init__(self, output, max_events=64, max_bytes=16 * 1024 * 1024):
        if max_events < 1 or max_bytes < 1:
            raise ValueError('Chunk limits must be positive')
        self.output = Path(output)
        (self.output / 'raw').mkdir(parents=True, exist_ok=True)
        if (self.output / 'events.jsonl').exists() or any((self.output / 'raw').glob('*.npz')):
            raise FileExistsError('Use a fresh attempt directory for raw recording')
        self.max_events, self.max_bytes = max_events, max_bytes
        self.started = time.perf_counter()
        self.count = self.chunk = self.buffer_bytes = 0
        self._events, self._arrays = [], {}
        # Fresh-attempt chunks are immutable after publication. Hash while
        # their bytes are still cached, avoiding a second cold read at the end.
        self._chunk_hashes = {}
        self._closed = False

    def __call__(self, kind, record):
        if self._closed:
            raise ValueError('Recorder is closed')
        arrays = {}

        def pack(value):
            scalar = isinstance(value, np.generic) or (
                isinstance(value, float) and not math.isfinite(value))
            if isinstance(value, np.ndarray) or scalar:
                array = np.array(value, copy=True)
                if array.dtype.hasobject:
                    raise TypeError('Object arrays cannot be recorded without pickle')
                key = f'e{self.count}_a{len(arrays)}'
                arrays[key] = array
                return {'npz_key': key, 'shape': list(array.shape),
                        'dtype': str(array.dtype), 'scalar': scalar}
            if isinstance(value, dict):
                return {str(k): pack(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [pack(v) for v in value]
            return value

        metadata = pack(record)
        # Validate before modifying the buffer, including unsupported values.
        metadata_bytes = len(json.dumps(metadata, allow_nan=False).encode())
        size = sum(a.nbytes for a in arrays.values()) + metadata_bytes
        if self._events and self.buffer_bytes + size > self.max_bytes:
            self.flush()
        event = {'event_index': self.count, 'event': kind,
                 'recorder_wall_elapsed_s': time.perf_counter() - self.started,
                 'raw_file': f'raw/chunk-{self.chunk:07d}.npz', 'record': metadata}
        self._events.append(event)
        self._arrays.update(arrays)
        self.buffer_bytes += size
        self.count += 1
        if len(self._events) >= self.max_events or self.buffer_bytes >= self.max_bytes:
            self.flush()

    def flush(self):
        if not self._events:
            return
        target = self.output / self._events[0]['raw_file']
        temporary = target.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, **self._arrays)
        temporary.replace(target)
        with (self.output / 'events.jsonl').open('a') as stream:
            for event in self._events:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
            stream.flush()
        self.chunk += 1
        self._events.clear()
        self._arrays.clear()
        self.buffer_bytes = 0
        self._chunk_hashes[str(target.relative_to(self.output))] = sha(target)

    def close(self):
        self.flush()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def manifest(self):
        self.flush()
        files = list((self.output / 'raw').glob('*.npz'))
        files += [self.output / name for name in (
            'events.jsonl', 'execution.npz', 'protocol.json', 'provenance.json',
            'controller_stats.json', 'metrics.json', 'screening.json',
            'readiness.json', 'build_receipt.json') if (self.output / name).is_file()]
        write_json(self.output / 'SHA256SUMS.json',
                   {str(p.relative_to(self.output)): self._chunk_hashes[str(p.relative_to(self.output))]
                    if str(p.relative_to(self.output)) in self._chunk_hashes else sha(p) for p in sorted(files)})


def iter_events(output):
    """Yield event envelopes with record arrays restored, for old or new layouts.

    Arrays are independent copies. An incomplete last JSON line from an abrupt
    interruption is ignored; malformed complete records still raise an error.
    """
    output = Path(output)
    raw_file, archive = None, None

    def unpack(value):
        if isinstance(value, dict):
            if 'npz_key' in value:
                array = archive[value['npz_key']].copy()
                return array[()] if value.get('scalar') or 'nonfinite_scalar' in value else array
            return {k: unpack(v) for k, v in value.items()}
        if isinstance(value, list):
            return [unpack(v) for v in value]
        return value

    try:
        with (output / 'events.jsonl').open() as stream:
            for line in stream:
                if not line.endswith('\n'):
                    break
                event = json.loads(line)
                if event['raw_file'] != raw_file:
                    if archive is not None:
                        archive.close()
                    raw_file = event['raw_file']
                    archive = np.load(output / raw_file, allow_pickle=False)
                yield dict(event, record=unpack(event['record']))
    finally:
        if archive is not None:
            archive.close()
