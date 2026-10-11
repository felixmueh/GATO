"""One saved synthetic negative rendering; no solver, ROS or robot execution."""
import json
from pathlib import Path

import numpy as np

from analysis import demo_visual
from recording import write_json


def test_saved_negative_outcomes_skips_and_existing_animation(tmp_path):
    source=Path(__file__).resolve().parents[1]
    protocol=json.loads((source/'experiments/inputs/demo/proposal.json').read_text())
    protocol.update(backend='internal',task_id='square',kind='reaching',input_sha256='synthetic-input')
    raw=tmp_path/'raw';attempt=raw/'square';attempt.mkdir(parents=True)
    write_json(raw/'index.json',[dict(task_id='square',backend='internal',status='completed'),dict(task_id='figure8-24s',backend='internal',status='skipped',reason='explicituserselection')])
    write_json(attempt/'protocol.json',protocol)
    write_json(attempt/'metrics.json',dict(campaign_id='demo_visual_square',task_id='square',backend='internal',machine='synthetic',evidence_kind='synthetic_test',task_success=False,settling_success=False,physical_pass=True,execution_error=None))
    write_json(attempt/'provenance.json',dict(runtime_source_commit='synthetic',module=dict(sha256='synthetic')))
    q=np.array(protocol['initial_state'][:7]);write_json(attempt/'controller_stats.json',dict(timestamps=[.03,.06],joint_positions=[q,q+.001],goal_indices=[0,1]))
    output=tmp_path/'derived'
    assert demo_visual.analyze(raw,source,output)==False
    summary=json.loads((output/'summary.json').read_text())
    assert summary['attempts'][0]['task_success'] is False and summary['attempts'][0]['evidence_kind']=='synthetic_test'
    assert [r['status'] for r in summary['selection']]==['completed','skipped']
    assert (output/'attempt-001/path-and-errors.png').is_file()
    assert (output/'attempt-001/reach_target.gif').is_file()
    assert (output/'attempt-001/feedback_trace.csv').is_file()
