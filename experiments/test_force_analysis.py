"""Saved force correspondence reuses E03 and keeps failed/foreign cohorts."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from recording import RawRecorder,sha,write_json
from analysis.force import load


def test_saved_pairs_failures_and_distinct_cohorts(tmp_path):
    attempts=[]
    for mode,bank,status in [('original','bankA','complete'),('adapted','bankA','complete'),('adapted','bankB','complete'),('original','bankA','failed')]:
        name=str(len(attempts));raw=tmp_path/name
        with RawRecorder(raw) as recorder:
            write_json(raw/'protocol.json',dict(case=dict(mode=mode,input_index=0,truth_index=1)))
            write_json(raw/'provenance.json',dict(machine='rtx5060',source_commit='a'*40,module_sha256='module',bank_manifest_sha256=bank))
            write_json(raw/'metrics.json',dict(parameters=dict(mode=mode,input_index=0,truth_index=1),evidence_kind='measurement',outcome=status,execution_error='retained' if status=='failed' else None))
            recorder('force_tick',dict(category='measured',status=status,selected_index=3,hypotheses=np.zeros((16,6)),predictions=np.zeros((16,14)),
                timings_s=dict(total=2. if mode=='original' else 1.),backend_measurements=[dict(native_solve_us=1000.,memory=None,memory_error='unavailable')]))
            recorder.manifest()
        attempts.append(dict(experiment='E03',attempt=name,status='execution_failed' if status=='failed' else 'completed',manifest_sha256=sha(raw/'SHA256SUMS.json')))
    index=tmp_path/'index.json';write_json(index,dict(attempts=attempts))
    result=load(index)
    assert len(result['attempts'])==4 and len(result['ticks'])==4
    assert len(result['pairs'])==1 and result['pairs'][0]['E01_correspondence']
    assert result['pairs'][0]['original_over_adapted_wall']==2.
    assert result['ticks'][0]['native_reciprocal_Hz']==1000.
    assert result['ticks'][0]['observed_additional_device_bytes'] is None
    assert result['ticks'][-1]['status']=='failed'
    (tmp_path/'0'/'metrics.json').write_text('{}')
    import pytest
    with pytest.raises(ValueError,match='Pinned raw'):load(index)
