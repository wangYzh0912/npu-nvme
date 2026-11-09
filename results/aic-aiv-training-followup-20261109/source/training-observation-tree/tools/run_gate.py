#!/usr/bin/env python3
"""Run each declared pytest case in a bounded process and retain raw evidence."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from npu_nvme.schemas.evidence import (aggregate, junit_counts, sha256_file,
                                      validate_manifest, validate_profile)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    profile_path = Path(args.profile)
    if not profile_path.is_file():
        profile_path = ROOT / 'config/gates' / (args.profile + '.json')
    try:
        profile = json.loads(profile_path.read_text())
        validate_profile(profile)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out/'profile.json', profile)
    sources = {}
    for directory in ('python', 'src', 'include', 'tests', 'tools', 'config', 'experiments', 'scripts'):
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.c','.h','.json'):
                sources[str(path.relative_to(ROOT))] = sha256_file(path)
    if (ROOT/'train.py').is_file(): sources['train.py'] = sha256_file(ROOT/'train.py')
    write(out/'source_manifest.json', sources)
    results = []
    env = dict(os.environ, PYTHONPATH=str(ROOT)+':'+str(ROOT/'python')+':'+os.environ.get('PYTHONPATH',''))
    if os.environ.get('GATE_PYTHONPATH'):
        env['PYTHONPATH'] += ':' + os.environ['GATE_PYTHONPATH']
    pytest_python = os.environ.get('GATE_PYTHON', sys.executable)
    probe = subprocess.run([pytest_python, '-c',
        'import sys,json,pytest; print(json.dumps(dict(executable=sys.executable,version=sys.version,pytest_version=pytest.__version__,pytest_path=pytest.__file__)))'],
        env=env, capture_output=True, text=True, timeout=30)
    if probe.returncode:
        write(out/'test_environment_error.json', dict(returncode=probe.returncode, stderr=probe.stderr))
        return 3
    test_runtime = json.loads(probe.stdout)
    for index, spec in enumerate(profile['cases']):
        run = out/f'case-{index:03d}'
        run.mkdir()
        junit = run/'junit.xml'
        command = [pytest_python,'-m','pytest','-q','--disable-warnings',spec['nodeid'],
                   '--junitxml',str(junit)]
        result = dict(id=spec['id'],required=spec['required'],tier=spec['tier'],argv=command,
                      counts=dict(tests=0,skipped=0,failures=0,errors=0),junit=None)
        begin = time.monotonic()
        if spec['tier'].startswith('HW'):
            stdout, stderr, rc = '', 'Hardware profiles require the explicit device/region preflight; unavailable in this runner', 3
            execution, validation = 'blocked', 'invalid'
        else:
            process = subprocess.Popen(command, cwd=ROOT, env=dict(env, GATE_ARTIFACT_DIR=str(run)), stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,text=True,start_new_session=True)
            try:
                stdout,stderr = process.communicate(timeout=spec['timeout_seconds'])
                rc = process.returncode
                execution,validation = 'completed','fail'
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                stdout,stderr = process.communicate()
                rc = process.returncode
                execution,validation = 'aborted','invalid'
            if junit.exists():
                try:
                    result['counts'] = junit_counts(junit)
                    result['junit'] = str(junit.relative_to(out))
                except ValueError:
                    execution,validation = 'blocked','invalid'
            counts = result['counts']
            if execution == 'completed':
                if rc == 0 and counts['tests'] and not any(counts[k] for k in ('skipped','errors','failures')):
                    validation = 'pass'
                elif counts['skipped'] or not counts['tests']:
                    execution,validation = 'blocked','invalid'
        (run/'stdout.txt').write_text(stdout)
        (run/'stderr.txt').write_text(stderr)
        result.update(execution_status=execution,validation_status=validation,returncode=rc,
                      elapsed_seconds=time.monotonic()-begin,
                      stdout=str((run/'stdout.txt').relative_to(out)),stderr=str((run/'stderr.txt').relative_to(out)))
        result['binary_artifacts'] = [str(p.relative_to(out)) for p in run.rglob('*.elf')]
        results.append(result)
        print(f"{spec['id']}: {execution}/{validation}",flush=True)
    execution,validation = aggregate(results)
    # A moving checkout cannot substantiate a run against one source snapshot.
    changed = [name for name,digest in sources.items()
               if not (ROOT/name).is_file() or sha256_file(ROOT/name) != digest]
    if changed:
        for case in results:
            case.update(execution_status='aborted',validation_status='invalid')
        execution,validation = aggregate(results)
    result = dict(schema_version=1,profile_id=profile['profile_id'],profile_sha256=sha256_file(out/'profile.json'),
                  execution_status=execution,validation_status=validation,cases=results,
                  case_count=sum(r['counts']['tests'] for r in results),
                  environment={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                               'python':test_runtime['version'],'executable':test_runtime['executable'],'binary_sha256':None,
                               'test_runtime':test_runtime,
                               'source_manifest_sha256':sha256_file(out/'source_manifest.json')},
                  invocation={'argv':sys.argv,'cwd':str(ROOT),'script_sha256':sha256_file(Path(__file__))},
                  metrics={},faults_injected=[],changed_sources=changed,
                  boundaries={'can_prove':['declared host test outcomes'],
                              'cannot_prove':['device behavior','hardware DMA stop',
                                              'C binary identity unless reported by C_IMPL test output']})
    write(out/'result.json',result)
    write(out/'evidence_manifest.json',dict(schema_version=1,artifacts=[
        dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha256_file(p))
        for p in sorted(out.rglob('*')) if p.is_file()]))
    validate_manifest(out/'evidence_manifest.json')
    return 0 if validation == 'pass' else 3 if execution=='blocked' else 4 if execution=='aborted' else 1

if __name__ == '__main__':
    sys.exit(main())
