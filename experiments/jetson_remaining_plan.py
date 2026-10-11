#!/usr/bin/env python3
"""Prepare unchanged Jetson study proposals, without executing measurements."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def prepare():
    selections = (
        ('campaign.json', 'T01', 't01_jetson_proposed.json', 'Reaching and figure-eight tracking'),
        ('f01c.json', 'F01c', 'f01c_jetson_proposed.json', 'Same-candidate batching timing'),
        ('ba_rtx5060.json', None, 'ba_jetson_proposed.json', 'Batching quality and nominal computation'),
    )
    for source, selected, destination, name in selections:
        original = json.loads((ROOT / source).read_text())
        entries = [copy.deepcopy(e) for e in original['experiments'] if selected is None or e['id'] == selected]
        if not entries:
            raise ValueError('Missing original study: ' + source)
        for entry in entries:
            entry['machines'] = ['jetson']
            entry['name'] = name
        plan = dict(version=1, experiments=entries,
            status='proposed_not_executed_target_duration_discussion_pending',
            proposal_source=source,
            note='Unchanged coverage and repeats. Discuss retained Jetson timing, estimated duration/storage and any shorter protocol with Felix before execution. F01ab is completed and excluded.')
        (ROOT / destination).write_text(json.dumps(plan, indent=2) + '\n')


if __name__ == '__main__':
    prepare()
