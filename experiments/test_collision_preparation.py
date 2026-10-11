"""CPU-only portable geometry, corpus and missing fault-evidence checks."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import collision_plan
from analysis.fault_envelope import analyze_trial,lines


def test_bundle_preserves_used_mesh_bytes_and_only_rewrites_uri(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'meshes').mkdir()
    mesh=source/'meshes/test.stl';mesh.write_bytes(b'original used mesh bytes')
    urdf=source/'robot.urdf';urdf.write_text('<robot name="test"><link name="a"><collision><geometry><mesh filename="package://meshes/test.stl" scale="1 2 3"/></geometry></collision></link></robot>')
    result=collision_plan.bundle(urdf,tmp_path/'portable')
    bundled=ET.parse(tmp_path/'portable/model.urdf').find('.//collision/geometry/mesh')
    assert bundled.attrib['scale']=='1 2 3'
    assert (tmp_path/'portable'/bundled.attrib['filename']).read_bytes()==mesh.read_bytes()
    assert result['original_sha256']==collision_plan.sha(urdf)


def test_frozen_monitor_corpus_is_bounded_current_torso_configuration():
    states=collision_plan.corpus()
    assert len(states)==8 and len({s['id'] for s in states})==8
    assert all(s['torso_lift_joint_m']==.1 and len(s['right_arm_q_rad'])==7 for s in states)
    assert {tuple(s['right_arm_qd_rad_s']) for s in states}=={(0.,)*7,(.05,)*7}


def test_missing_observations_do_not_become_zero_motion_or_safety_certificate(tmp_path):
    (tmp_path/'trial.json').write_text(json.dumps(dict(case='planner_exit',worker_exit=86)))
    result=analyze_trial(tmp_path)
    assert result['outcome']=='inconclusive' and result['observed_samples']==0
    assert result['max_observed_joint_displacement_rad'] is None
    assert result['state_age_s'] is result['clearance_m'] is result['certified_braking_distance_m'] is None


def test_malformed_middle_observation_is_visible_without_dropping_valid_rows(tmp_path):
    path=tmp_path/'observer.jsonl';path.write_text('{"wall":1}\n{bad json}\n{"wall":2}\n')
    errors=[]
    assert lines(path,errors)==[{'wall':1},{'wall':2}]
    assert errors[0]['line']==2 and errors[0]['file']==str(path) and not errors[0]['terminal_line']
