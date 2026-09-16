#!/usr/bin/env python3
"""Build both ABI2 runtimes without replacing historical installations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from user_environment import (read_profile, with_python_identity, isolated_environment,
                              verify_library_paths, verify_runtime_abi)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--spdk', required=True, type=Path)
    parser.add_argument('--dpdk-ring', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / 'config/user_environments.json').read_text())
    source = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)).hexdigest())
    write(out/'source.json', source)
    results = []
    for name, profile in manifest['profiles'].items():
        directory = out/name
        directory.mkdir()
        toolkit = Path(profile['toolkit']).resolve()
        profile['toolkit'] = str(toolkit)
        build = directory/'build'
        commands = [
            ['cmake','-S',str(ROOT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
             '-DASCEND_CANN_PACKAGE_PATH='+str(toolkit),'-DSOC_VERSION=Ascend910B3',
             '-DSPDK_ROOT_DIR='+str(args.spdk.resolve()),
             '-DDPDK_MEMPOOL_RING_FIXED_LIB='+str(args.dpdk_ring.resolve())],
            ['cmake','--build',str(build),'--target','npu_nvme','-j4']]
        for index, command in enumerate(commands):
            write(directory/('command-%d.json'%index), command)
            with (directory/('command-%d.stdout'%index)).open('w') as stdout, (directory/('command-%d.stderr'%index)).open('w') as stderr:
                result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=1800)
            if result.returncode:
                write(directory/'result.json',dict(status='failed',returncode=result.returncode))
                return result.returncode
        profile['library'] = str((build/'libnpu_nvme.so.2').resolve())
        profile['required_abi_major'] = 2
        write(out/'environments.json', manifest)
        identity = with_python_identity(read_profile(out/'environments.json',name,ROOT))
        environment = isolated_environment(identity,ROOT)
        resolved = verify_library_paths(identity,environment)
        runtime = verify_runtime_abi(identity,environment)
        write(directory/'environment.lock.json',dict(identity, resolved_libraries=resolved,runtime_abi=runtime))
        results.append(dict(profile=name,library=profile['library'],sha256=identity['library_sha256'],abi=runtime['abi_major']))
        print(name+': build, dependency isolation and ABI2 load passed',flush=True)
    write(out/'result.json',dict(status='pass',scope='build and load only; no hardware I/O',profiles=results))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
