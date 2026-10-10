"""Bounded CPU-only check of unchanged CMake artifact location retention."""
import importlib.util,json,tempfile
from pathlib import Path
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('preparation',root/'prepare_jetson.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as tmp:
    base=Path(tmp);actual=base/'campaign-source/test-artifacts/tiago-tasks';copied=base/'suite/raw'
    for name in ['short_reach','long_reach','figure8_tracking']:
        p=actual/name;p.mkdir(parents=True);(p/'summary.json').write_text(json.dumps({'passed':True,'case':name}))
    summaries=m.retain_suite(actual,copied)
    assert len(summaries)==3
    for p in summaries:assert p.read_bytes()==(actual/p.parent.name/'summary.json').read_bytes()
    (actual/'short_reach/summary.json').write_text(json.dumps({'passed':False}))
    try:m.retain_suite(actual,base/'negative/raw')
    except RuntimeError:pass
    else:raise AssertionError('negative suite accepted')
    try:m.retain_suite(base/'missing',base/'missing-copy')
    except RuntimeError:pass
    else:raise AssertionError('missing suite accepted')
print('CPU artifact copy/original retention, negative and missing checks passed; no native calls/builds.')
