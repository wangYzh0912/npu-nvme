import ctypes as C
import json
from pathlib import Path
import subprocess
from npu_nvme.storage.bindings import (NPUNVMETransferSpec, NPUNVMETransferItem,
    NPUNVMETransferReceipt, NPUNVMETransferDigest, NPUNVMECapabilities)


def test_c_python_transfer_layout(tmp_path):
    root=Path(__file__).resolve().parents[2]
    types=[NPUNVMECapabilities,NPUNVMETransferSpec,NPUNVMETransferItem,NPUNVMETransferReceipt,NPUNVMETransferDigest]
    statements=[];expected=[]
    for typ in types:
        statements.append('printf("%zu\\n", sizeof('+typ.__name__+'));')
        expected.append(C.sizeof(typ))
        for field,_ in typ._fields_:
            statements.append('printf("%zu\\n", offsetof('+typ.__name__+','+field+'));')
            expected.append(getattr(typ,field).offset)
    c=tmp_path/'layout.c';exe=tmp_path/'layout'
    c.write_text('#include <stdio.h>\n#include "npu_nvme.h"\nint main(void){'+''.join(statements)+'return 0;}')
    subprocess.run(['cc','-I'+str(root/'include'),str(c),'-o',str(exe)],check=True)
    assert [int(n) for n in subprocess.check_output([str(exe)],text=True).split()]==expected
