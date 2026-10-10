"""Explicit Jetson target selection, with optional host device-tree detection."""
from pathlib import Path
import platform


def select_target(device=None):
    if device not in (None,'jetson'):raise ValueError('Dedicated launcher only supports --device jetson')
    model=None;error=None
    try:model=Path('/proc/device-tree/model').read_text(errors='replace').strip('\x00\n')
    except OSError as exception:error=f'{type(exception).__name__}: {exception}'
    if device is None and (model is None or 'Orin' not in model):
        raise RuntimeError('Host Orin detection unavailable; use --device jetson in the Jetson container. No attempt directory created.')
    return dict(target='jetson',selection='explicit_user_selection' if device else 'host_Orin_model_detection',
                host_architecture=platform.machine(),device_tree_model=model,device_tree_read_error=error,
                hardware_validation='Actual suite SM87 CUDA smoke remains required before preflight/matrix; target selection is not CUDA verification')
