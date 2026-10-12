"""CPU contract check: the archived collector and manual robot bind the passed square."""
import json
from pathlib import Path
import subprocess
import sys

from adapters import demo
import machine_launch


def test_four_corner_collector_archive_and_standalones_select_passed_input(tmp_path):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root/'experiments/launch/rtx5060_gazebo_four_corner_v10.json').read_text())
    upward, square, goal1 = plan['experiments']
    previous = json.loads((root/'experiments/launch/rtx5060_gazebo_upward_selected.json').read_text())
    previous['experiments'][1]['name'] = 'Upward-pointing figure-eight (12 s)'
    assert upward == previous['experiments'][1]
    assert goal1['enabled'] is False
    assert sum(len(e['cases']) for e in plan['experiments'] if e.get('enabled', True)) == 2
    machine_launch.verify_sources(plan)
    collector, pin = demo.collector_source(root, tmp_path/'build', square['collector_source_commit'])
    assert pin == square['collector_source_commit']
    frozen = root/next(iter(square['inputs']))
    selected = collector/'experiments/deployment/demo/proposal_right10_roll.json'
    assert selected.read_bytes() == frozen.read_bytes()
    proposal = json.loads(frozen.read_text())
    assert len(proposal['goals']) == 4 and proposal['tracking_tasks'] == []
    assert proposal['square_side_m'] == .1
    assert proposal['square_center_torso_m'] == [.5, -.35, -.15]
    assert [g['id'] for g in proposal['goals']] == ['bottom-right', 'top-right', 'top-left', 'bottom-left']
    worker = demo.load_collector(collector/'experiments/deployment/demo/run_demo.py')
    assert worker.INPUT == selected and worker.TASKS == ('square',)
    for backend in ('gazebo', 'physical_robot', 'internal'):
        resolved = worker.resolve(backend, 'square')
        for key, value in proposal.items():
            assert resolved[key] == value
    for script in ('demo_gazebo.py', 'demo_robot.py', 'demo_internal.py'):
        result = subprocess.run([sys.executable, str(selected.with_name(script))],
                                text=True, capture_output=True, check=True)
        preview = json.loads(result.stdout)
        assert [r['task_id'] for r in preview['selection']] == ['square']
        assert len(preview['protocols']) == 1
        for key, value in proposal.items():
            assert preview['protocols'][0][key] == value
