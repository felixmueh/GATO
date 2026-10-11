#!/usr/bin/env python3
"""Index, import and export saved measurements without running experiments."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import tarfile


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def records(output):
    """Keep failed/partial measurements visible; a malformed record affects itself."""
    output = Path(output).resolve()
    rows = []
    for path in sorted(output.rglob('receipt.json')):
        if 'attempt-' not in path.parent.name:
            continue
        relative = path.parent.relative_to(output).as_posix()
        try:
            receipt = read(path)
            identity = receipt.get('identity', {})
            spec = identity.get('experiment', {})
            if not isinstance(spec, dict):
                spec = {}
            case = receipt.get('case', {})
            metrics_path = path.parent / 'metrics.json'
            metrics = read(metrics_path) if metrics_path.exists() else {}
            if not isinstance(metrics, dict):
                raise ValueError('metrics.json must contain an object')
            question = {'F01ab':'RQ-timing', 'T01':'RQ-control', 'V01':'RQ-control', 'BB':'RQ-batching'}.get(metrics.get('campaign_id'))
            rows.append(dict(machine=receipt.get('machine'), experiment=receipt.get('experiment'),
                research_question=receipt.get('research_question', spec.get('research_question', metrics.get('research_question', question))),
                case=case.get('id'), repeat=case.get('repeat', 0), status=receipt.get('status', 'unknown'),
                execution_backend=metrics.get('execution_backend', receipt.get('backend', spec.get('execution_backend', spec.get('backend')))),
                timing_mode=metrics.get('timing_mode', spec.get('timing_mode')),
                source_commit=receipt.get('source_commit', identity.get('source_commit', spec.get('source_commit', metrics.get('source_commit')))),
                protocol=receipt.get('protocol', identity.get('protocol', spec.get('protocol'))),
                attempt=relative, imported='_imports' in path.relative_to(output).parts,
                metrics=metrics, error=receipt.get('error')))
        except (OSError, ValueError, TypeError, AttributeError) as error:
            rows.append(dict(attempt=relative, status='unreadable', error=str(error), metrics={}))
    return rows


def collect(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = records(output)
    write(output / 'index.json', rows)
    fields = ['machine', 'experiment', 'research_question', 'case', 'repeat', 'status',
              'execution_backend', 'timing_mode', 'source_commit', 'protocol', 'attempt', 'imported', 'error', 'metrics']
    temporary = output / 'summary.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'metrics': json.dumps(row.get('metrics', {}), sort_keys=True)})
    temporary.replace(output / 'summary.csv')
    lines = ['# Experiment results', '', '| Machine | Experiment / question | Case | Status | Data |',
             '| --- | --- | --- | --- | --- |']
    def label(value):
        return str(value if value is not None else '').replace('|', '\\|').replace('\n', ' ')
    for row in rows:
        lines.append(f"| {label(row.get('machine'))} | {label(row.get('experiment'))} / {label(row.get('research_question'))} | "
                     f"{label(row.get('case'))} / {label(row.get('repeat'))} | {label(row['status'])} | [attempt]({row['attempt']}/) |")
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return rows


def result_files(root):
    """Accept portable result folders, never an execution workspace by accident."""
    root = Path(root).resolve()
    forbidden = {'.git', '.worktrees', 'build', 'builds', 'cache', '.experiment-cache',
                 'collector-source', 'campaign-source', 'campaign-native', '__pycache__', '.pytest_cache'}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Result export/import requires ordinary files, not symlink: {path}')
        if path.is_dir() and (path.name in forbidden or path.name.startswith('build-')):
            raise ValueError(f'This is an execution workspace, not a result collection: {path}. Use the experiment result exporter.')
        if path.is_file():
            yield path


def register(output, imported, plan_path, experiment_id):
    """Explicitly map old completed measurements to matching planned cases.

    This is a user-selected migration, never automatic inference from a changed
    environment. Original receipts are untouched and remain the provenance.
    """
    plan = read(plan_path)
    matches = [e for e in plan['experiments'] if e['id'] == experiment_id]
    if len(matches) != 1:
        raise ValueError('Select exactly one experiment present in the plan')
    experiment = matches[0]
    registrations = read(output / 'imports.json') if (output / 'imports.json').exists() else []
    added = []
    for path in imported.rglob('receipt.json'):
        if 'attempt-' not in path.parent.name:
            continue
        receipt = read(path)
        if receipt.get('status') != 'completed' or receipt.get('experiment') != experiment_id:
            continue
        old_identity = receipt.get('identity', {})
        old_spec = old_identity.get('experiment', {})
        if not isinstance(old_spec, dict):
            old_spec = old_identity
        if experiment.get('parameters', {}) != old_spec.get('parameters', {}):
            raise ValueError(f'Experiment parameters differ for {path}')
        metrics = read(path.parent / 'metrics.json')
        for key in ('timing_mode', 'backend'):
            old_value = metrics.get('execution_backend' if key == 'backend' else key, old_spec.get(key))
            if experiment.get(key) is not None and old_value != experiment[key]:
                raise ValueError(f'Imported {key} differs or is unknown for {path}')
        old_case = receipt.get('case', {})
        matched = [case for case in experiment['cases'] if case.get('parameters', {}) == old_case.get('parameters', {})]
        if len(matched) != 1:
            raise ValueError(f'No unique matching planned parameters for {path}; import remains available without scheduling registration')
        repeat = old_case.get('repeat', 0)
        if repeat not in range(experiment.get('repeats', 1)):
            raise ValueError(f'Imported repeat outside planned coverage: {path}')
        item = dict(machine=receipt['machine'], experiment=experiment_id,
                    source_commit=experiment['source_commit'], protocol=experiment.get('protocol'),
                    inputs=experiment.get('inputs', {}), timing_mode=experiment.get('timing_mode'),
                    backend=experiment.get('backend', experiment.get('execution_backend')),
                    case=dict(id=matched[0]['id'], parameters=matched[0].get('parameters', {}), repeat=repeat),
                    status='completed', path=path.parent.relative_to(output).as_posix())
        if experiment.get('parameters'):
            item['parameters'] = experiment['parameters']
        if item not in registrations and item not in added:
            added.append(item)
    if not added:
        print('No new completed cases registered; saved evidence remains indexed.')
    write(output / 'imports.json', registrations + added)
    return added


def import_results(source, output, name, plan=None, experiment=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Import name must be a single directory name')
    if output == source or output.is_relative_to(source):
        raise ValueError('Collection must be outside the imported directory')
    # Validate once before copying, so a workspace cannot bloat the collection.
    list(result_files(source))
    if not source.is_dir():
        raise ValueError('Import source must be an extracted result directory')
    destination = output / '_imports' / name
    if destination.exists():
        raise ValueError(f'Import already exists; original evidence will not be overwritten: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    try:
        if plan:
            register(output, destination, plan, experiment)
    finally:
        collect(output)
    return destination


def export_results(output, archive):
    output, archive = Path(output).resolve(), Path(archive).resolve()
    if archive.is_relative_to(output):
        raise ValueError('Place the export archive outside the result directory')
    files = list(result_files(output))
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + '.tmp')
    try:
        with tarfile.open(temporary, 'w:gz', compresslevel=1) as package:
            for path in files:
                package.add(path, arcname='results/' + path.relative_to(output).as_posix(), recursive=False)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Central or per-machine result directory')
    parser.add_argument('--import-results', type=Path, help='Extracted portable results, e.g. Jetson F01ab export')
    parser.add_argument('--name', help='Unique name for the preserved import')
    parser.add_argument('--plan', type=Path, help='Explicitly register matching imported cases against this plan')
    parser.add_argument('--experiment', help='Experiment ID for explicit scheduling registration')
    parser.add_argument('--export', type=Path, help='Write a portable tar.gz outside the result directory')
    args = parser.parse_args()
    if args.import_results:
        if not args.name:
            parser.error('--import-results needs --name')
        if bool(args.plan) != bool(args.experiment):
            parser.error('--plan and --experiment must be given together')
        import_results(args.import_results, args.output, args.name, args.plan, args.experiment)
    elif args.plan or args.experiment:
        parser.error('--plan/--experiment apply to --import-results')
    rows = collect(args.output)
    if args.export:
        export_results(args.output, args.export)
    print(f'{len(rows)} saved attempts indexed in {args.output}')


if __name__ == '__main__':
    main()
