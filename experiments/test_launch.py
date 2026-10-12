"""Focused CPU checks of the manual-launch handoff and existing runner interface."""
import json
from pathlib import Path
import sys

import machine_launch as launch


def test_frozen_manual_plans_hold_only_square_and_original_force():
    for machine, count in [('rtx5060', 712), ('jetson', 368)]:
        plan = json.loads((launch.HERE / 'launch' / (machine + '.json')).read_text())
        assert sum(len(e['cases']) * e.get('repeats', 1) for e in plan['experiments']) == count
        assert not any(e['id'] == 'E02-original' for e in plan['experiments'])
        assert not any('gazebo' in e.get('backend', '').lower() or 'robot' in e.get('backend', '').lower()
                       for e in plan['experiments'])
        assert not any(e['id'] == 'demo_visual_square_internal' and c['id'] == 'square'
                       for e in plan['experiments'] for c in e['cases'])
        assert len(next(e for e in plan['experiments'] if e['id'] == 'BB')['cases']) == 200
        if machine == 'rtx5060':
            assert len(next(e for e in plan['experiments'] if e['id'] == 'demo_visual_square_internal')['cases']) == 3
        launch.verify_sources(plan)


def test_complete_launch_uses_runner_and_exports_every_attempt(tmp_path, monkeypatch):
    # Stub measurements; use the actual existing collect/export functions.
    root = tmp_path / 'checkout';here = root / 'experiments';(here / 'launch').mkdir(parents=True)
    (here / 'launch/rtx5060.json').write_text(json.dumps(dict(version=1, experiments=[])))
    monkeypatch.setattr(launch, 'ROOT', root);monkeypatch.setattr(launch, 'HERE', here)
    monkeypatch.setattr(launch, 'verify_sources', lambda *a: None)
    monkeypatch.setattr(launch, 'check_dependencies', lambda *a: None)
    monkeypatch.setattr(launch.run, 'git', lambda *a: 'test-source')
    calls = []
    def execute(args):
        calls.append(args)
        if not args.dry_run:
            attempt = Path(args.output) / 'rtx5060/test/case/attempt-001';attempt.mkdir(parents=True)
            (attempt / 'receipt.json').write_text(json.dumps(dict(status='failed', error='retained failure')))
            (attempt / 'partial-raw.bin').write_bytes(b'retained scientific data')
            return 1
        return 0
    monkeypatch.setattr(launch.run, 'run', execute)
    monkeypatch.setattr(sys, 'argv', ['launch.py', 'rtx5060', '--python', 'python3'])
    assert launch.main() == 1
    assert [a.dry_run for a in calls] == [True, False]
    assert Path(calls[-1].python).is_file()
    output = Path(calls[-1].output)
    assert json.loads((output / 'index.json').read_text())[0]['status'] == 'failed'
    import tarfile
    archive = next((root / '.experiment-work').glob('*-results.tar.gz'))
    with tarfile.open(archive) as package:
        assert any(n.endswith('partial-raw.bin') for n in package.getnames())
        assert 'results/deployment.json' in package.getnames()
    monkeypatch.setattr(sys, 'argv', ['launch.py', 'rtx5060', '--collect-only'])
    assert launch.main() == 0
    assert len(calls) == 2
