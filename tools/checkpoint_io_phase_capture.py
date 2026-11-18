#!/usr/bin/env python3
"""Collect two bounded device PMU windows during one observed Qwen continuation."""
import argparse
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.source / 'python'))
    from user_environment import read_profile, isolated_environment
    profile = read_profile(a.manifest, 'candidate', a.source)
    env = isolated_environment(profile, a.source)
    env['ASCEND_RT_VISIBLE_DEVICES'] = '4,5,6,7'
    uid = Path(profile['toolkit']).stat().st_uid
    account = pwd.getpwuid(uid)
    env.update(HOME=account.pw_dir, USER=account.pw_name, LOGNAME=account.pw_name)
    a.output.mkdir(parents=True, exist_ok=False)
    os.chown(a.output, uid, account.pw_gid)
    report = dict(status='watching', duration_per_capture_seconds=20,
                  device=4, scopes='device-wide HBM and engine PMU; partial phase windows only', captures=[])

    def record():
        (a.output / 'captures.json').write_text(json.dumps(report, indent=2)+'\n')

    record()
    stream = None
    deadline = time.monotonic() + 3600
    expected = {'owner_restore_chunk_begin': 'restore', 'owner_payload_begin': 'save'}
    seen = set()
    try:
        while time.monotonic() < deadline and len(seen) < 2:
            if stream is None:
                path = a.raw / 'io-events' / 'owner.jsonl'
                if path.exists():
                    stream = path.open()
            line = stream.readline() if stream else ''
            if not line:
                path = a.raw / 'result.json'
                if path.exists() and json.loads(path.read_text())['execution_status'] != 'running':
                    raise RuntimeError('training ended before both phase captures')
                time.sleep(1)
                continue
            row = json.loads(line)
            phase = expected.get(row['event'])
            if phase is None or phase in seen:
                continue
            seen.add(phase)
            command = [str(Path(profile['toolkit']) / 'bin' / 'msprof'),
                       '--output=' + str(a.output / phase), '--sys-devices=4', '--sys-period=20',
                       '--ai-core=on', '--aic-mode=sample-based', '--aic-freq=100',
                       '--aic-metrics=PipeUtilization', '--task-time=off',
                       '--sys-hardware-mem=on', '--sys-hardware-mem-freq=50', '--type=text']
            capture = dict(phase=phase, trigger=row, command=command,
                           launch_monotonic_ns=time.monotonic_ns(), launch_realtime_ns=time.time_ns())
            report['captures'].append(capture)
            report['status'] = 'capturing_' + phase
            record()
            with (a.output / (phase + '.log')).open('w') as log:
                child = subprocess.Popen(command, env=env, cwd=a.source,
                                         user=uid, group=account.pw_gid,
                                         extra_groups=os.getgrouplist(account.pw_name, account.pw_gid),
                                         stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = child.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=30)
                    raise
            capture.update(returncode=code, export_end_monotonic_ns=time.monotonic_ns(),
                           export_end_realtime_ns=time.time_ns())
            if code:
                raise RuntimeError(f'{phase} msprof returned {code}')
            databases = sorted((a.output / phase).glob('PROF_*/msprof_*.db'))
            if len(databases) != 1:
                raise RuntimeError('one exported msprof database required')
            capture['database'] = str(databases[0])
            report['status'] = 'watching'
            record()
        if len(seen) != 2:
            raise TimeoutError('phase capture watch deadline')
        report['status'] = 'pass'
    except BaseException as error:
        report.update(status='fail', error=repr(error))
        raise
    finally:
        if stream:
            stream.close()
        record()
    print(json.dumps(dict(status=report['status'], captures=len(seen))))


if __name__ == '__main__':
    main()
