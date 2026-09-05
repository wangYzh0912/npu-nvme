"""Baseline adapter registry."""

from .base import Adapter, NoneAdapter, adapter_status
from .mindspore_sync import MindSporeSyncAdapter
from .ours import OursAdapter
from .datastates_acl import DataStatesAclAdapter
from .pccheck_acl import PCCheckAclAdapter
from .bytecheckpoint_host import ByteCheckpointHostAdapter
from .fastpersist_host import FastPersistHostAdapter

ADAPTERS = {
    "none": NoneAdapter,
    "mindspore_sync": MindSporeSyncAdapter,
    "ours": OursAdapter,
    "datastates_acl": DataStatesAclAdapter,
    "pccheck_acl": PCCheckAclAdapter,
    "bytecheckpoint_host": ByteCheckpointHostAdapter,
    "fastpersist_host": FastPersistHostAdapter,
}
