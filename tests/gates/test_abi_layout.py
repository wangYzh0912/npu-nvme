"""Compare compiled public C layouts with the canonical ctypes declarations."""
import ctypes
import json
from pathlib import Path
import subprocess

from npu_nvme.storage.bindings import NPUNVMEStats,NPUNVMERetainedSlot

ROOT=Path(__file__).resolve().parents[2]


def test_public_struct_sizes_and_offsets(tmp_path):
    types=[NPUNVMEStats,NPUNVMERetainedSlot]
    expected={};statements=[]
    for cls in types:
        expected[cls.__name__+'.sizeof']=ctypes.sizeof(cls)
        statements.append(f'printf("{cls.__name__}.sizeof %zu\\n",sizeof({cls.__name__}));')
        for name,_ in cls._fields_:
            expected[cls.__name__+'.'+name]=getattr(cls,name).offset
            statements.append(f'printf("{cls.__name__}.{name} %zu\\n",offsetof({cls.__name__},{name}));')
    source=tmp_path/'abi.c';binary=tmp_path/'abi'
    source.write_text('#include <stddef.h>\n#include <stdio.h>\n#include "npu_nvme.h"\nint main(void){\n'+'\n'.join(statements)+'\nreturn 0;}\n')
    subprocess.run(['cc','-I'+str(ROOT/'include'),str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
    output=subprocess.check_output([str(binary)],text=True)
    actual={key:int(value) for key,value in (line.split() for line in output.splitlines())}
    assert actual==expected
    print(json.dumps(actual,sort_keys=True))
