"""CUDA free/total snapshots and Linux host RSS; no claimed exact peak."""
import ctypes
import ctypes.util
import resource
import time


class CudaMemory:
    def __init__(self):
        library=ctypes.util.find_library('cudart') or 'libcudart.so'
        self.library=ctypes.CDLL(library)
        self.query=self.library.cudaMemGetInfo
        self.query.argtypes=(ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t))
        self.query.restype=ctypes.c_int

    def __call__(self,stage):
        free,total=ctypes.c_size_t(),ctypes.c_size_t();start=time.perf_counter()
        status=self.query(ctypes.byref(free),ctypes.byref(total))
        return dict(stage=stage,cuda_status=int(status),free_bytes=free.value if status==0 else None,
            total_bytes=total.value if status==0 else None,
            host_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            query_wall_s=time.perf_counter()-start,
            scope='CUDA context/device free-total snapshots; host RSS separate; Jetson shared system memory, no summing or exact allocation/peak claim')


def summarize(snapshots):
    valid=[s for s in snapshots if s.get('cuda_status')==0 and s.get('free_bytes') is not None]
    baseline=valid[0]['free_bytes'] if valid and valid[0]['stage']=='before_owner' else None
    return dict(snapshots=snapshots,
        maximum_observed_additional_device_bytes=None if baseline is None else max(0,baseline-min(s['free_bytes'] for s in valid)),
        complete=bool(valid) and len(valid)==len(snapshots),
        query_wall_s=sum(s.get('query_wall_s',0.) for s in snapshots),
        scope='difference from before-owner free memory at sampled boundaries; not per-allocation accounting or a continuous peak')
