"""Raw observer: lossless NumPy arrays plus readable JSON metadata."""
from dataclasses import fields, is_dataclass
import json
from pathlib import Path

import numpy as np


def write_record(directory, record):
    """Caller supplies a unique per-tick directory. Refuse existing destinations."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            key = f"array_{len(arrays):04d}"
            arrays[key] = value
            return {"npz_key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
        if is_dataclass(value):
            return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, dict):
            return {str(key): encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        if callable(value):
            return {"callable": f"{value.__module__}.{value.__qualname__}",
                    "provenance": "caller must supply implementation/config source hashes"}
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return {"nonfinite_float": str(value)}
        return value

    metadata = encode(record)
    # Arrays preserve NaN/Inf without filtering or nonstandard JSON tokens.
    np.savez(directory / "arrays.npz", **arrays)
    (directory / "record.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
