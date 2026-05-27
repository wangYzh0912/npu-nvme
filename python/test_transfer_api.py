import os
import sys

from transfer_api import TransferAPI


def main():
    lib_path = os.environ.get("NPU_NVME_LIB")
    api = TransferAPI(lib_path=lib_path) if lib_path else TransferAPI()

    pci_addr = os.environ.get("NPUNVME_TEST_PCI_ADDR")
    npu_id = int(os.environ.get("NPUNVME_TEST_NPU_ID", "0"))

    if not pci_addr:
        print("[Test] Loaded libnpu_nvme.so and bound transfer symbols.")
        print("[Test] Set NPUNVME_TEST_PCI_ADDR to run init/cleanup smoke test.")
        return 0

    rc = api.init(pci_addr=pci_addr, npu_id=npu_id)
    print(f"[Test] init rc={rc}")
    api.cleanup()
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
