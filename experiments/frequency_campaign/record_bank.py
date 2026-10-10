"""Durable raw recorder using campaign NPZ/JSONL contract; never a native import."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('f01_raw',ROOT/'experiments/campaign/raw.py')
raw=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw)


class Recorder:
    def __init__(self,output,protocol):
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        raw.write_json(self.output/'protocol.json',protocol)
        self.recorder=raw.RawRecorder(self.output)

    def __call__(self,event,record):
        self.recorder(event,record)

    def finish(self,result):
        raw.write_json(self.output/'metrics.json',result)
        self.recorder.manifest()
