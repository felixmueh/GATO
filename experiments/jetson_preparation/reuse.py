"""One setup-correction cohort, reusing qualified native evidence without reruns."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import full_run as F
import target

OLD_RELEASE_SHA256='7a4106c0baa74b62d6b066e997f2da571860a02fb10350b469bc44b0ca39fe58'
OLD_LAUNCHER='ba55f91'


def validate_parent(parent,sha):
    parent=Path(parent).resolve()
    release_path=parent/'jetson-release.json'
    if sha(release_path)!=OLD_RELEASE_SHA256:raise ValueError('Reuse requires the original qualified Jetson release')
    state=json.loads((parent/'launch-state.json').read_text())
    if state.get('status')!='held' or state.get('stage')!='canonical_A_preflight':raise ValueError('Reuse only accepts the pre-owner screening setup failure')
    if state.get('completed_cells') or any((parent/name).exists() for name in ['runner-repo','runner-cache','results','jetson-f01ab-plan.json']):raise ValueError('Matrix already started; no reuse retry')
    family=json.loads((parent/'family-state.json').read_text())
    if family.get('status')!='held' or family.get('machine')!='jetson' or family.get('release_manifest_sha256')!=OLD_RELEASE_SHA256:raise ValueError('Original family hold/release differs')
    metrics_path=parent/'readiness/f01-raw/metrics.json'
    item=family.get('readiness_metrics')
    if item and (Path(item['path']).resolve()!=metrics_path or sha(metrics_path)!=item['sha256']):raise ValueError('Original failed readiness metrics changed')
    mirrored=parent/'readiness/metrics.json'
    if mirrored.exists() and mirrored.read_bytes()!=metrics_path.read_bytes():raise ValueError('Mirrored failed readiness metrics differ')
    metrics=json.loads(metrics_path.read_text())
    expected=dict(campaign_id='F01-A-preflight',machine='jetson',timing_mode='A',execution_backend='internal',task_success=None,physical_pass=None,execution_error='Screening data changed or failed',outcome='execution_failed',evidence_kind='measurement')
    if metrics!=expected:raise ValueError('Only the saved pre-owner screening failure may reuse qualification')
    raw=metrics_path.parent;manifest=json.loads((raw/'SHA256SUMS.json').read_text())
    for name,digest in manifest.items():
        path=raw/name
        if not path.resolve().is_relative_to(raw) or sha(path)!=digest:raise ValueError('Original readiness raw changed: '+name)
    if manifest.get('metrics.json')!=sha(metrics_path):raise ValueError('Failed readiness manifest omits metrics')
    events=raw/'events.jsonl'
    if (events.exists() and events.read_text().strip()) or any((raw/'raw').iterdir()):raise ValueError('Native/task/plant events already exist; no reuse retry')
    release=json.loads(release_path.read_text())
    prep=json.loads((parent/'preparation.json').read_text())
    if prep.get('runtime_source')!=F.RUNTIME or prep.get('science_source')!=F.SCIENCE or prep.get('cuda_arch')!='87-real':raise ValueError('Original preparation source/target differs')
    for item in prep['build_receipts']+prep['suite_summaries']:
        if sha(Path(item['path']))!=item['sha256']:raise ValueError('Original preparation receipt changed')
    old=F.load(parent/'source/experiments/frequency_campaign/release.py','original_qualified_release')
    for n in [16,32,64,128]:old.validate(release_path,OLD_RELEASE_SHA256,'jetson',n)
    # Review itself remains hashed by old release. Current launcher paths may now
    # contain the published correction; their historical reviewed bytes live in Git.
    review=json.loads(Path(release['build_review']['path']).read_text())
    for policy in review['policy']:
        name=Path(policy['path']).name
        blob=subprocess.check_output(['git','-C',str(F.ROOT),'show',f'{OLD_LAUNCHER}:experiments/jetson_preparation/{name}'])
        import hashlib
        if hashlib.sha256(blob).hexdigest()!=policy['sha256']:raise ValueError('Original reviewed launcher policy differs')
    return dict(path=str(parent),release=F.pin(release_path,sha),launch_state=F.pin(parent/'launch-state.json',sha),family_state=F.pin(parent/'family-state.json',sha),failed_readiness=F.pin(metrics_path,sha),preparation=F.pin(parent/'preparation.json',sha),reason='Authorized setup correction: frozen screening metadata was recomputed before any native/task event')


def reuse_qualified(parent,work,python,sha,write,command,device=None):
    parent=Path(parent).expanduser().resolve();work=Path(work).expanduser().absolute()
    allowed=F.ROOT/'example_artifacts/jetson_f01ab'
    if not parent.is_relative_to(allowed.resolve()) or not work.resolve().is_relative_to(allowed.resolve()):raise ValueError('Reuse artifacts must remain below this checkout')
    if work.exists():raise FileExistsError('Fresh correction cohort required')
    for marker in allowed.glob('*/launch-state.json'):
        if marker.parent.resolve()!=parent and json.loads(marker.read_text()).get('status') in ['running','held']:raise RuntimeError('Another campaign is running/held; preserve it')
    for receipt in allowed.glob('*/reuse-parent.json'):
        if json.loads(receipt.read_text()).get('path')==str(parent):raise RuntimeError('Original qualification already reused; no automatic repeat')
    original=validate_parent(parent,sha)
    prep=json.loads((parent/'preparation.json').read_text())
    python=str(Path(shutil.which(python) or python).expanduser().absolute())
    if python!=prep['python']:raise ValueError('Keep the original qualified Python environment')
    selection=target.select_target(device)
    work.mkdir(parents=True);(work/'logs').mkdir();(work/'tmp').mkdir();(work/'environment-cache').mkdir()
    os.environ.update(TMPDIR=str(work/'tmp'),XDG_CACHE_HOME=str(work/'environment-cache'),PYTHONDONTWRITEBYTECODE='1')
    write(work/'reuse-parent.json',original);write(work/'target-selection.json',selection)
    write(work/'launch-state.json',dict(status='running',stage='setup_correction',parent_failed_attempt=original))
    # run_campaign validates unchanged qualified native/suite evidence, executes
    # one actual canonical preflight and the unchanged64 cells; no build or Make.
    F.run_campaign(parent/'source',work,python,sha,write,command,prepared=parent,parent=original)
