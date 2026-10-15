"""Read-only metadata for the portable Jetson study; no device-setting changes."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import platform
import shutil
import subprocess


def gpu_snapshot():
    result=dict(utc=datetime.now(timezone.utc).isoformat(),machine=platform.machine(),platform=platform.platform())
    for name,path in [('device_model','/proc/device-tree/model'),('load_average','/proc/loadavg')]:
        p=Path(path)
        if p.exists():result[name]=p.read_text().strip('\x00\n')
    try:
        command=['nvidia-smi','--query-gpu=name,uuid,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv']
        snapshot=subprocess.run(command,capture_output=True,text=True,timeout=5)
        result.update(nvidia_smi=snapshot.stdout,returncode=snapshot.returncode,stderr=snapshot.stderr)
    except (OSError,subprocess.TimeoutExpired) as error:
        result.update(nvidia_smi=None,telemetry_error=str(error))
    # Jetson usually lacks nvidia-smi; one bounded observation, no clock changes.
    if result.get('nvidia_smi') is None or result.get('returncode',0):
        try:
            fallback=subprocess.run(['timeout','3','tegrastats','--interval','1000'],capture_output=True,text=True,timeout=5)
            result.update(tegrastats=fallback.stdout,tegrastats_returncode=fallback.returncode,tegrastats_stderr=fallback.stderr)
        except (OSError,subprocess.TimeoutExpired) as error:
            result.update(tegrastats=None,tegrastats_error=str(error))
    return result


def copy_host_conditions(source,output):
    if source is None:return dict(status='not supplied')
    source=Path(source).resolve()
    if not source.is_dir():raise ValueError('Host-condition evidence directory missing')
    destination=Path(output)/'host_conditions'
    shutil.copytree(source,destination)
    return dict(original_directory=str(source),files_sha256={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in destination.rglob('*') if p.is_file()},status='supplied query evidence retained unchanged')
