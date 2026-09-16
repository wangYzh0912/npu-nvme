#!/usr/bin/env python3
"""Noninteractive candidate-only build for the Qwen training baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))


def main():
    from user_environment import read_profile,with_python_identity,isolated_environment,verify_library_paths,verify_runtime_abi
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--spdk',type=Path,required=True)
    parser.add_argument('--dpdk-ring',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text());profile=manifest['profiles']['candidate']
    spdk=args.spdk.resolve();ring=args.dpdk_ring.resolve()
    if not spdk.is_dir() or not ring.is_file():raise ValueError('explicit built SPDK and patched DPDK archive required')
    report=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)).hexdigest(),
        spdk=str(spdk),dpdk_ring=str(ring),dpdk_ring_sha256=hashlib.sha256(ring.read_bytes()).hexdigest(),commands=[])
    def record():(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    record()
    commands=[['cmake','-S',str(ROOT),'-B',str(out/'build'),'-DCMAKE_BUILD_TYPE=Release',
        '-DASCEND_CANN_PACKAGE_PATH='+profile['toolkit'],'-DSOC_VERSION=Ascend910B3',
        '-DSPDK_ROOT_DIR='+str(spdk),'-DDPDK_MEMPOOL_RING_FIXED_LIB='+str(ring)],
        ['cmake','--build',str(out/'build'),'--target','npu_nvme','-j4'],
        [profile['python'],str(ROOT/'tools/build_mindspore_address.py')]]
    try:
        for index,command in enumerate(commands):
            report['commands'].append(command);record()
            with (out/f'build-{index}.log').open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        profile['library']=str((out/'build/libnpu_nvme.so.2').resolve());profile['required_abi_major']=2
        (out/'environments.json').write_text(json.dumps(manifest,indent=2)+'\n')
        identity=with_python_identity(read_profile(out/'environments.json','candidate',ROOT))
        env=isolated_environment(identity,ROOT)
        report.update(status='pass',environment=identity,resolved_libraries=verify_library_paths(identity,env),
                      runtime_abi=verify_runtime_abi(identity,env),scope='build and load; hardware acceptance separate')
        record();print(str(out/'environments.json'));return 0
    except BaseException as error:
        report.update(status='failed',error=repr(error));record();raise


if __name__=='__main__':raise SystemExit(main())
