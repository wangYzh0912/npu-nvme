#!/usr/bin/env python3
"""Validate gate artifacts and required-case results."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from npu_nvme.schemas.evidence import validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    try:
        result = validate_manifest(args.manifest)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f'invalid evidence: {error}', file=sys.stderr)
        return 2
    print(f"{result['execution_status']}/{result['validation_status']}")
    return 0 if result['validation_status'] == 'pass' else 1

if __name__ == '__main__':
    sys.exit(main())
