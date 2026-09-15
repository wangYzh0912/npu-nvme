"""Explicit read-only disk inspection command."""
import argparse
import json
import sys
from npu_nvme.storage.inspection import inspect_disk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pci', default='0000:83:00.0')
    p.add_argument('--npu', type=int, default=7)
    args = p.parse_args()
    try:
        print(json.dumps(inspect_disk(args.pci, args.npu), indent=2))
        return 0
    except Exception as error:
        print(json.dumps({'status':'fail','error':str(error)}), file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
