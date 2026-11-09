#!/usr/bin/env python3
"""Build the allocation-metadata adapter using the selected MindSpore headers."""
import importlib.util
import importlib.metadata
import json
import re
from pathlib import Path
import subprocess
import sysconfig


def main():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.find_spec('mindspore')
    package = Path(spec.origin).parent
    include = package / 'include'
    version = importlib.metadata.version('mindspore')
    if not version.startswith('2.7.'):
        raise RuntimeError('allocation adapter requires the verified MindSpore 2.7 headers')
    target = root/'python/npu_nvme/framework'/('_address_native'+sysconfig.get_config_var('EXT_SUFFIX'))
    wheel = next(package.glob('_c_expression.*.so'))
    abi = re.search(rb'__pybind11_internals_v4_gcc_libstdcpp_cxxabi(\d+)__', wheel.read_bytes())
    if abi is None:
        raise RuntimeError('unsupported MindSpore pybind compiler ABI')
    # Match the wheel's C++ ABI and pybind type registry, using a real GCC ABI
    # compatibility mode rather than casting Python object internals.
    abi_version = int(abi[1]) - 1000
    command = ['g++', '-shared', '-fPIC', '-O2', '-std=c++17', '-fabi-version='+str(abi_version),
               '-D_GLIBCXX_USE_CXX11_ABI=0',
               '-DNPU_NVME_MINDSPORE_VERSION='+json.dumps(version)]
    for path in (sysconfig.get_path('include'), include, include/'third_party',
                 include/'mindspore/core/include', include/'mindspore/ccsrc'):
        command += ['-I', str(path)]
    command += [str(root/'src/framework/mindspore_address.cc'), '-L'+str(package/'lib'),
                '-lmindspore_common', '-lmindspore_core', '-Wl,-rpath,'+str(package/'lib'),
                '-o', str(target)]
    subprocess.run(command, check=True)
    print(json.dumps(dict(framework_version=version, target=str(target), command=command)))


if __name__ == '__main__':
    main()
