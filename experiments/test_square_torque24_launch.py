"""Bounded CPU contract for the single limits-only Gazebo successor; no native run."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from adapters import demo
import machine_launch
import run


def test_limits_only_live_patch_input_and_collector_preserve_guard_and_old_weights(tmp_path):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root/'experiments/launch/rtx5060_gazebo_square_torque24_limits_only.json').read_text())
    assert len(plan['experiments']) == 1
    square = plan['experiments'][0]
    assert square['cases'] == [dict(id='square', parameters=dict(backend='gazebo'))]
    machine_launch.verify_sources(plan)
    parent = '8dbb8b8d6dc1d65e902373a9aa58693e33329c14'
    source = square['source_commit']
    changed = subprocess.check_output(['git', '-C', str(root), 'diff', '--name-only', parent, source], text=True).splitlines()
    assert changed == ['gato/dynamics/tiago_right/tiago_right_limits.cuh']
    header = subprocess.check_output(['git', '-C', str(root), 'show', source+':'+changed[0]])
    assert header == subprocess.check_output(['git', '-C', str(root), 'show', square['accepted_internal_native_source_commit']+':'+changed[0]])
    controls = header.decode().split('CTRL_LIMITS_DATA')[1].split('};')[0]
    assert controls.count('static_cast<T>(-24)') == controls.count('static_cast<T>(24)') == 7
    collector, _ = demo.collector_source(root, tmp_path/'build', square['collector_source_commit'])
    selected = collector/'experiments/deployment/demo/proposal_torque24_limits_only.json'
    frozen = root/next(iter(square['inputs']))
    assert selected.read_bytes() == frozen.read_bytes()
    proposal = json.loads(frozen.read_text())
    old = json.loads((root/'experiments/inputs/square_four_corner_v10/proposal.json').read_text())
    for key, value in old.items():
        if key not in ('id', 'name', 'status'):
            assert proposal[key] == value
    assert proposal['live_controller']['max_abs_torque'] == 26
    assert proposal['live_controller']['clamp_torque'] is False
    worker = demo.load_collector(collector/'experiments/deployment/demo/run_demo.py')
    resolved = worker.resolve('gazebo', 'square')
    assert resolved['runtime_pin'] == source and worker.INPUT == selected
    for key, value in proposal.items():
        assert resolved[key] == value
    preview = json.loads(subprocess.check_output([sys.executable, str(selected.with_name('demo_gazebo.py'))], text=True))
    assert preview['protocols'][0] == resolved
    for name in ('demo_robot.py', 'demo_internal.py'):
        assert selected.with_name(name).read_bytes() == subprocess.check_output(['git', '-C', str(root), 'show', '91118e3:experiments/deployment/demo/'+name])
    old_recipe = deepcopy(square)
    old_recipe['source_commit'] = parent
    abi = {'python_abi': list(sys.version_info[:2])}
    assert run.build_key(square, {'cuda_arch': '120'}, sys.executable, abi) != run.build_key(old_recipe, {'cuda_arch': '120'}, sys.executable, abi)
