"""JSON-line worker helpers shared by CPU checkpoint backends."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np


def attach_arrays(descriptor):
    shm = shared_memory.SharedMemory(name=descriptor["name"])
    arrays = {}
    for field in descriptor["fields"]:
        begin = int(field["offset"])
        end = begin + int(field["nbytes"])
        arrays[field["name"]] = np.ndarray(
            tuple(field["shape"]), dtype=np.dtype(field["dtype"]),
            buffer=shm.buf[begin:end])
    control = descriptor["controls"]
    begin = int(control["offset"])
    end = begin + int(control["nbytes"])
    arrays["controls/state"] = np.ndarray(
        (int(control["nbytes"]),), dtype=np.uint8, buffer=shm.buf[begin:end])
    return shm, arrays


def canonical_digest(arrays):
    digest = hashlib.sha256()
    for name in sorted(name for name in arrays if name != "controls/state"):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(arrays[name]).tobytes())
    digest.update(np.ascontiguousarray(arrays["controls/state"]).tobytes())
    return digest.hexdigest()


def durable_tree(path):
    path = Path(path)
    for entry in sorted(path.rglob("*")):
        if entry.is_file():
            with entry.open("rb", buffering=0) as stream:
                os.fsync(stream.fileno())
    current = path
    while True:
        descriptor = os.open(str(current), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if current.parent == current:
            break
        if current == path.parent:
            break
        current = current.parent


def serve(handler):
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("operation") == "close":
                response = handler.close()
                print(json.dumps({"status": "closed", **(response or {})}),
                      flush=True)
                return
            response = handler.execute(request)
            print(json.dumps({"status": "ok", **response}, sort_keys=True),
                  flush=True)
        except BaseException as error:
            import traceback
            print(json.dumps({"status": "error", "error": repr(error),
                              "traceback": traceback.format_exc()}), flush=True)
