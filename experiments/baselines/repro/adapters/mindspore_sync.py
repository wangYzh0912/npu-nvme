from .base import DurableFileAdapter


class MindSporeSyncAdapter(DurableFileAdapter):
    name = "mindspore_sync"
    kind = "framework-reference"

