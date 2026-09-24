"""Portable coordinator for separate GATO failure-mode experiments.

Use --dry-run to inspect every command. The committed protocol is a planned
matrix, not a claim that every cell is supported or has already been validated.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / 'examples/randomized_multimodal'
DEFAULT_PROTOCOL = ROOT / 'experiments/tiago_solver_limits/protocol.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def artifact_manifest(directory):
    return {str(path.relative_to(directory)): {'bytes': path.stat().st_size, 'sha256': digest(path)}
            for path in sorted(directory.rglob('*')) if path.is_file()
            and path.suffix in ('.json', '.npz', '.csv', '.log')
            and path.name not in ('suite.json', 'artifact_manifest.json')}


def valid_receipt(record):
    receipts = record.get('artifacts', {})
    return bool(receipts) and all(Path(path).is_file() and digest(path) == sha
                                  for path, sha in receipts.items())


def capture(command):
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
        return {'command': command, 'returncode': result.returncode,
                'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': command, 'error': str(error)}


def environment():
    versions = {}
    for package in ('numpy', 'scipy', 'pin', 'pinocchio', 'matplotlib', 'pybind11'):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    return {'utc': datetime.now(timezone.utc).isoformat(), 'host': platform.node(),
            'platform': platform.platform(), 'python': sys.version, 'executable': sys.executable,
            'packages': versions,
            'gpu': capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version,compute_cap', '--format=csv']),
            'nvcc': capture(['nvcc', '--version']), 'cmake': capture(['cmake', '--version']),
            'git_head': capture(['git', 'rev-parse', 'HEAD']),
            'git_status': capture(['git', 'status', '--porcelain']),
            'environment': {name: os.environ.get(name) for name in
                            ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}


def source_hashes():
    paths = {ROOT / 'CMakeLists.txt', ROOT / 'python/bindings.cu'}
    for directory in (ROOT / 'gato', EXAMPLES):
        paths.update(path for path in directory.rglob('*')
                     if path.is_file() and path.suffix in ('.cu', '.cuh', '.h', '.py', '.urdf'))
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def cell_name(n, duration):
    return f'N{n}_T{duration:g}'


def option_values(command, flag):
    values=[]
    for value in command[command.index(flag)+1:]:
        if value.startswith('--'): break
        values.append(value)
    return values


def child_completion_failures(kind, command, output):
    """Check recorded coverage, retaining complete records of numerical failures."""
    output = Path(output)
    failures = []
    def require(path):
        if not path.is_file(): failures.append(f'Missing artifact: {path}')
    def read(path):
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict): raise ValueError('Expected JSON object')
            return value
        except (OSError, ValueError) as error:
            failures.append(f'Invalid summary {path}: {error}')
            return {}
    expected_batches = option_values(command, '--batches')
    if kind == 'openloop':
        for pair in option_values(command, '--cells'):
            n, duration = pair.split(':')
            for task_id in option_values(command, '--tasks'):
                cell = output/f'{cell_name(int(n),float(duration))}_task{task_id}'
                value = read(cell/'summary.json')
                branches = value.get('branches', {})
                if value.get('failure') or not isinstance(branches, dict):
                    failures.append(f'Failed cell: {cell}'); continue
                for batch in expected_batches:
                    branch = branches.get(batch)
                    if not isinstance(branch, dict):
                        failures.append(f'Missing batch {batch}: {cell}')
                    require(cell/f'b{batch}.npz')
                    if '--reference' in command:
                        if not isinstance(branch, dict) or not isinstance(branch.get('cpu_reference'), dict):
                            failures.append(f'Missing CPU reference for batch {batch}: {cell}')
                        require(cell/f'b{batch}_cpu.npz')
    elif kind == 'mpc-interval':
        value = read(output/'summary.json')
        branches = value.get('branches', {})
        steps = int(option_values(command, '--steps')[0])
        clock = float(option_values(command, '--clock')[0])
        for batch in expected_batches:
            branch = branches.get(batch) if isinstance(branches, dict) else None
            if not isinstance(branch, dict):
                failures.append(f'Missing MPC branch {batch}'); continue
            require(output/f'b{batch}_executed.npz')
            stages = branch.get('stages')
            required = {'executed', 'failure', 'simulated_seconds', 'complete'}
            if not required.issubset(branch) or not isinstance(stages, list) or not isinstance(branch.get('executed'), dict):
                failures.append(f'Incomplete MPC branch {batch}'); continue
            # A physical/integration failure can terminate early and remains a
            # valid recorded outcome. An unexplained truncated run cannot.
            failed = bool(branch['failure'])
            elapsed = branch['simulated_seconds']
            if (len(stages) > steps or (not failed and len(stages) != steps)
                    or not isinstance(elapsed, (int, float))
                    or not 0 <= elapsed <= steps*clock+1e-9
                    or (not failed and abs(elapsed-steps*clock) > 1e-9)):
                failures.append(f'Truncated or inconsistent MPC branch {batch}')
            for step, stage in enumerate(stages):
                if not isinstance(stage, dict):
                    failures.append(f'Invalid MPC stage {batch}/{step}')
                elif stage.get('failure'):
                    if not failed or step != len(stages)-1:
                        failures.append(f'Inconsistent failed MPC stage {batch}/{step}')
                else:
                    require(output/f'b{batch}_stage{step}.npz')
    elif kind == 'latency':
        value = read(output/'summary.json')
        repeats = int(option_values(command, '--repeats')[0])
        expected = Counter((int(batch), repeat) for batch in expected_batches for repeat in range(repeats))
        records = value.get('records', [])
        try:
            actual = Counter((row['batch'], row['repeat']) for row in records)
            if actual != expected: failures.append('Latency repeat/batch coverage mismatch')
            fields = {'starts', 'calls', 'solve_wall_seconds', 'finite_outputs',
                      'budget_counts_valid', 'pcg_breakdown_events', 'work_valid', 'output_sha256'}
            if any(not fields.issubset(row) for row in records):
                failures.append('Incomplete latency observations')
            summaries = Counter(row['batch'] for row in value.get('summary', []))
            if summaries != Counter(map(int, expected_batches)):
                failures.append('Latency batch summary coverage mismatch')
        except (KeyError, TypeError):
            failures.append('Malformed latency observations')
        require(output/'inputs.npz'); require(output/'inputs.json')
    return failures


def make_plan(protocol, args):
    """Return independent commands; dependency preparation precedes MPC cases."""
    output = args.output
    solver = protocol['solver']
    common = ['--seed', str(protocol['seed']), '--iters', str(solver['iterations']),
              '--passes', str(solver['passes']), '--rho', str(solver['rho'])]
    plan = []
    def add(case_id, kind, command, folder):
        plan.append({'id': case_id, 'kind': kind, 'command': [str(v) for v in command],
                     'output': str(folder)})
    def openloop(case_id, cells, tasks, folder, extra=()):
        command = [sys.executable, EXAMPLES/'tiago_horizon.py', '--cells',
                   *[f'{n}:{duration:.12g}' for n, duration in cells], '--tasks', *map(str, tasks),
                   '--batches', *map(str, protocol['batches']), *common, *extra,
                   '--output', folder]
        if args.reference: command.append('--reference')
        if args.resume: command.append('--resume')
        if args.inputs_root:
            command.extend(['--inputs-root', args.inputs_root/folder.relative_to(output)])
        add(case_id, 'openloop', command, folder)
    for study in args.only:
        config = protocol['studies'][study]
        if study == 'resources':
            for n in config['knots']:
                for batch in config['batches']:
                    if batch not in (1, 16): raise ValueError('Resource probe supports B1/B16.')
                    add(f'resources_N{n}_B{batch}', study,
                        [args.build_dir/'bin'/f'tiago_resources_N{n}', '--batch', batch, '--smoke'],
                        output/'resources')
                    if n in config.get('optin_knots', []):
                        add(f'resources_N{n}_B{batch}_optin', study,
                            [args.build_dir/'bin'/f'tiago_resources_N{n}', '--batch', batch,
                             '--smoke', '--pcg-opt-in'], output/'resources')
        elif study == 'batch':
            for n, duration in config['cells']:
                name = cell_name(n, duration); folder = output/'batch'/name
                command = [sys.executable, EXAMPLES/'batch_equivalence.py', '--knots', n,
                           '--duration', duration, '--batches', *config['batches'],
                           '--seed', protocol['seed'], '--iterations', solver['iterations'],
                           '--passes', solver['passes'], '--repeats', config['repeats'],
                           '--profiles', 'fixed', 'production', '--output', folder]
                if args.resume: command.append('--resume')
                if args.inputs_root: command.extend(['--input', args.inputs_root/'batch'/name/'inputs.npz'])
                add(f'batch_{name}', study, command, folder)
        elif study == 'latency':
            for n, duration in config['cells']:
                name = cell_name(n, duration); folder = output/study/name
                command = [sys.executable, EXAMPLES/'tiago_latency.py', '--knots', n,
                           '--duration', duration, '--batches', *config['batches'],
                           '--iterations', config['iterations'], '--repeats', config['repeats'],
                           '--seed', protocol['seed'], '--output', folder]
                if args.inputs_root: command.extend(['--input', args.inputs_root/study/name/'inputs.npz'])
                add(f'latency_{name}', study, command, folder)
        elif study == 'discretization':
            for duration in config['durations']:
                for n in config['knots']:
                    name = cell_name(n, duration); folder = output/study/name
                    openloop(f'{study}_{name}', [(n, duration)], protocol['tasks'], folder)
        elif study == 'horizon':
            for n in config['knots']:
                duration = (n-1)*config['dt']; name = cell_name(n, duration)
                openloop(f'{study}_{name}', [(n, duration)], protocol['tasks'], output/study/name)
        elif study == 'mpc-interval':
            n, duration = config['cell']; source = output/study/'initial'
            openloop('mpc_initial', [(n, duration)], config['tasks'], source)
            for task in config['tasks']:
                for clock in config['clocks']:
                    steps = round(config['execution_duration']/clock)
                    if steps <= 0 or abs(steps*clock-config['execution_duration']) > 1e-9:
                        raise ValueError('MPC execution_duration must be divisible by each clock.')
                    for mode in config['initializations']:
                        name = f'mpc_task{task}_clock{clock:g}_{mode}'; folder = output/study/name
                        command = [sys.executable, EXAMPLES/'tiago_horizon_mpc.py', '--source', source,
                                   '--knots', n, '--duration', duration, '--task', task,
                                   '--batches', *config.get('batches',protocol['batches']), '--steps', steps, '--clock', clock,
                                   '--iters', solver['iterations'], '--passes', solver['passes'], '--rho', solver['rho'],
                                   '--initialization', mode, '--warm-state-source', 'planned', '--output', folder]
                        add(name, study, command, folder)
        elif study == 'effort':
            for n, duration in config['cells']:
                for passes in config['passes']:
                    name = f'{cell_name(n,duration)}_passes{passes}'
                    openloop(f'effort_{name}', [(n,duration)], config['tasks'], output/study/name,
                             ['--passes', str(passes)])
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', nargs='+', choices=['resources','batch','discretization','horizon','mpc-interval','effort','latency'])
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build/solver-validation')
    parser.add_argument('--architecture', help='CMake CUDA architecture override; default protocol/native')
    parser.add_argument('--jobs', type=int)
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--reference', action='store_true', help='Include costly independent CPU local references')
    parser.add_argument('--inputs-root', type=Path, help='Prior suite output for exact saved inputs on another GPU')
    args = parser.parse_args()
    args.output = args.output.resolve(); args.build_dir = args.build_dir.resolve()
    if args.inputs_root: args.inputs_root = args.inputs_root.resolve()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get('schema_version') != 1: parser.error('Unsupported protocol schema.')
    args.only = args.only or list(protocol['studies'])
    if len(set(args.only)) != len(args.only): parser.error('Duplicate experiment names.')
    if args.jobs is not None and args.jobs < 1: parser.error('--jobs must be positive.')
    if args.inputs_root and (args.inputs_root == args.output or args.inputs_root in args.output.parents
                             or args.output in args.inputs_root.parents):
        parser.error('Output and --inputs-root must be separate, non-nested directories.')
    if args.inputs_root and not args.inputs_root.is_dir():
        parser.error('--inputs-root must be an existing directory.')
    if any(study in args.only for study in ('batch','latency')) and protocol['solver']['rho'] != .01:
        parser.error('Batch/latency audits currently fix rho=.01; use that declared protocol or run horizon-only studies.')
    plan = make_plan(protocol, args)
    architecture = args.architecture or protocol['build']['architecture']
    knots = protocol['build']['knots']; jobs = args.jobs or protocol['build']['jobs']
    configure = ['cmake','-S',str(ROOT),'-B',str(args.build_dir),
                 '-DCMAKE_BUILD_TYPE=Release','-DBUILD_TESTING=OFF',
                 f'-DPython3_EXECUTABLE={sys.executable}',f'-DCMAKE_CUDA_ARCHITECTURES={architecture}',
                 '-DPLANT=tiago_right_multimodal','-DKNOTS='+';'.join(map(str,knots))]
    builds = [[ 'cmake','--build',str(args.build_dir),'--parallel',str(jobs),'--target',target]
              for n in knots for target in (f'bsqpN{n}_tiago_right_multimodal',f'tiago_resources_N{n}')]
    if args.dry_run:
        print(json.dumps({'protocol':protocol,'selected_studies':args.only},indent=2))
        if not args.skip_build:
            for command in [configure,*builds]: print(shlex.join(command))
        if not args.build_only:
            for case in plan: print(case['id']+': '+shlex.join(case['command']))
        return
    manifest_path = args.output/'suite.json'
    if args.output.exists() and any(args.output.iterdir()) and not manifest_path.exists():
        parser.error('Nonempty output lacks suite.json; use a fresh output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    current_environment = environment()
    identity = {'protocol':protocol,'selected_studies':args.only,'reference':args.reference,
                'architecture':architecture,'source_hashes':source_hashes(),
                'inputs_root':str(args.inputs_root) if args.inputs_root else None,
                'input_artifacts':artifact_manifest(args.inputs_root) if args.inputs_root else None,
                'build_dir':str(args.build_dir),'configure_command':configure,'build_commands':builds,
                'runtime':{key:current_environment[key] for key in ('python','executable','packages','gpu','nvcc')}}
    if manifest_path.exists():
        if not args.resume: parser.error('Output already contains suite.json; use a new directory or --resume.')
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity:
            parser.error('Resume protocol/source/input mismatch; use a new output directory.')
    else:
        manifest = {'identity':identity,'environment':current_environment,'records':{},'plan':plan,
                    'scope':'Process completion is not numerical success. Read each child result.'}
        write_json(manifest_path, manifest)
    def execute(case_id, command, kind, output=None):
        previous = manifest['records'].get(case_id)
        if args.resume and previous and previous.get('status') in ('completed','unsupported_resource'):
            if not valid_receipt(previous):
                raise SystemExit(f'Resume artifact missing/changed for {case_id}; preserve this run and use a new output directory.')
            print('RESUME completed',case_id,flush=True); return 0
        attempt = previous.get('attempt',1)+1 if previous else 1
        log = args.output/'logs'/f'{case_id}.attempt{attempt}.log'; log.parent.mkdir(exist_ok=True)
        print('RUN',case_id,shlex.join(command),flush=True)
        start=time.monotonic()
        try:
            with log.open('w') as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            code=result.returncode
        except OSError as error:
            log.write_text(str(error)+'\n'); code=127
        record={'kind':kind,'command':command,'returncode':code,'seconds':time.monotonic()-start,
                'log':str(log),'output':str(output) if output else None,'attempt':attempt,
                'status':'completed' if code == 0 else 'process_failed'}
        if previous: record['previous_attempts'] = [*previous.get('previous_attempts',[]),
            {key:value for key,value in previous.items() if key != 'previous_attempts'}]
        if kind=='resources':
            rows=[]
            log_lines=log.read_text(errors='replace').splitlines()
            for line in log_lines:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
            record['observations']=rows
            resource=next((row for row in rows if row.get('kind')=='resources'),None)
            if code and resource:
                limit = resource['shared_optin_per_block'] if resource.get('optin_requested') else resource['shared_default_per_block']
                launch_error=any(line.startswith('GPUassert: ') and 'pcg.cuh' in line
                    and any(message in line for message in
                            ('invalid argument','invalid configuration','too many resources'))
                    for line in log_lines)
                if launch_error and resource['pcg_dynamic_shared_bytes']+resource['pcg_static_shared_bytes'] > limit:
                    record['status']='unsupported_resource'
        files=[log]
        if kind == 'build' and code == 0:
            if case_id == 'configure':
                required=[args.build_dir/'CMakeCache.txt']
            elif command[-1].startswith('bsqp'):
                required=list((ROOT/'python/bsqp').glob(command[-1]+'.*.so'))
            else:
                required=[args.build_dir/'bin'/command[-1]]
            if not required or not all(path.is_file() for path in required):
                record['status']='incomplete_build'
            else:files.extend(required)
        if output:
            files.extend(path for path in Path(output).rglob('*') if path.is_file()
                         and path.suffix in ('.json','.npz','.csv'))
        if kind in ('openloop','latency','mpc-interval') and code == 0:
            failures = child_completion_failures(kind, command, output)
            if failures:
                record['status']='incomplete_child';record['child_failures']=failures
        if kind in ('batch','latency','mpc-interval') and code == 0 and not (Path(output)/'summary.json').is_file():
            record['status']='incomplete_child'
        if kind == 'batch' and code == 0 and (Path(output)/'summary.json').is_file():
            child=json.loads((Path(output)/'summary.json').read_text())
            profiles=child.get('profiles',{})
            required_profiles=option_values(command,'--profiles')
            if any(name not in profiles or not profiles[name].get('comparisons') for name in required_profiles):
                record['status']='incomplete_child'
            record['batch_verdicts']={name:value.get('verdict') for name,value in profiles.items()}
        record['artifacts']={str(path):digest(path) for path in files}
        manifest['records'][case_id]=record; write_json(manifest_path,manifest)
        print('DONE',case_id,'exit',code,'log',log,flush=True)
        return 0 if record['status'] in ('completed','unsupported_resource') else code or 1
    if not args.skip_build:
        if execute('configure',configure,'build'): raise SystemExit('CMake configure failed; see log.')
        build_failures=[]
        for command in builds:
            if execute('build_'+command[-1],command,'build'): build_failures.append(command[-1])
        if build_failures:
            manifest['status']='build_failed';manifest['process_failures']=build_failures
            write_json(manifest_path,manifest)
            raise SystemExit('Build failures retained; numerical experiments blocked to prevent loading stale extensions.')
    binary_paths=list((ROOT/'python/bsqp').glob('bsqpN*_tiago_right_multimodal*.so'))
    binary_paths+=list((args.build_dir/'bin').glob('tiago_resources_N*'))
    binaries={str(path):digest(path) for path in binary_paths if path.is_file()}
    if args.resume and manifest.get('binary_hashes') not in (None,binaries):
        parser.error('Built/loaded binary hashes changed; use a new output directory.')
    manifest['binary_hashes']=binaries;write_json(manifest_path,manifest)
    if args.build_only: return
    failures=[]
    for case in plan:
        if execute(case['id'],case['command'],case['kind'],case['output']): failures.append(case['id'])
    manifest['process_failures']=failures;write_json(manifest_path,manifest)
    manifest['status']='incomplete' if failures else 'completed'
    write_json(manifest_path,manifest)
    write_json(args.output/'artifact_manifest.json',artifact_manifest(args.output))
    print('Suite records:',manifest_path,flush=True)
    if failures:
        print('Process failures retained:',', '.join(failures),flush=True)
        raise SystemExit(1)


if __name__=='__main__': main()
