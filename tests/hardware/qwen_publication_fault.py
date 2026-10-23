"""Fail after all raw-commit completions, before filesystem publication.

This is a worker entry for full_checkpoint_acceptance.py, not a production
configuration flag or a power-loss simulation.
"""
import sys

from npu_nvme.runtime import training_catalog as catalog
from npu_nvme.workloads.qwen import main


def fail_publication(path, *, world=4):
    reservation=catalog.read_checked(path/'reservation.json')
    rows=[catalog.read_checked(path/f'rank_{r}/complete.json') for r in range(world)]
    if reservation['method']!='ours' or reservation['step']!=4 or world!=4:
        raise ValueError('fault fixture requires the first TP4 Ours commit at step 4')
    receipts=[row['backend']['receipt'] for row in rows]
    if any(receipt!=receipts[0] for receipt in receipts):
        raise ValueError('fault fixture did not reach a common raw commit')
    catalog.write_checked(path/'publication-fault.json',dict(injected=True,
        boundary='raw_committed_all_ranks_complete_before_publication',receipt=receipts[0]))
    raise RuntimeError('injected filesystem publication interruption')


if __name__=='__main__':
    catalog.publish=fail_publication
    sys.exit(main())
