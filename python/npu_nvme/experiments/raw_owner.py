"""Single-SPDK-owner append service for phase-one incremental frames."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time

from npu_nvme.d2 import wire
from npu_nvme.experiments.raw_store import (ALIGN, SUPER_BYTES, align,
    pack_commit, pack_frame_prefix, pack_superblock, unpack_superblock)
from npu_nvme.storage.bindings import load_backend
from npu_nvme.storage.full_transport import FullTransport


def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class RawOwner:
    def __init__(self, config, output):
        self.config = config; self.output = Path(output)
        self.deadline = time.monotonic() + config["timeout_seconds"]
        for name,value in config.get('native_scheduler',{}).items():
            if name not in ('NPU_NVME_COPY_BYTES','NPU_NVME_CHECKSUM_BYTES') or not 1<=int(value)<=16<<20:
                raise ValueError('invalid native scheduler override')
            os.environ[name]=str(value)
        backend = load_backend(config["library"])
        self.transport = FullTransport(backend, pci=config["pci_addr"], npu=-1,
            depth=4, chunk_size=config["chunk_bytes"], profiling_dir=self.output,
            role="host_owner")
        self.lock = threading.Lock()
        self.catalog_path = Path(config["catalog"])
        self.catalog = self._mount_or_format()
        self.shadow=None
        if config.get('verify_before_ack'):
            from npu_nvme.experiments.media_shadow import MediaShadow
            try:
                self.shadow=MediaShadow(config['initial_full'],config['strategy'],config['media_shadow'],config['block_elements'])
            except BaseException:
                self.transport.close(config['timeout_seconds'])
                raise

    def _mount_or_format(self):
        capacity = int(self.transport.total_bytes)
        if (type(self.config.get('maximum_run_bytes')) is not int or
                self.config['maximum_run_bytes']<=0 or
                SUPER_BYTES+self.config['maximum_run_bytes']+ALIGN>=capacity):
            raise MemoryError('raw namespace cannot admit declared run')
        before = self.transport.read(0, SUPER_BYTES)
        before_sha = hashlib.sha256(before).hexdigest()
        if self.catalog_path.exists():
            catalog = json.loads(self.catalog_path.read_text())
            header = unpack_superblock(before)
            if (catalog.get("config_sha256") != self.config["config_sha256"] or
                    catalog.get("namespace_bytes") != capacity or catalog.get("cursor", 0) < SUPER_BYTES or
                    catalog.get('cursor', capacity + 1) > capacity or
                    header['campaign_id'] != self.config['campaign_id'] or
                    header['config_sha256'] != self.config['config_sha256']):
                raise ValueError("raw experiment catalog identity differs")
            return catalog
        superblock = pack_superblock(namespace_bytes=capacity,
            config_sha256=self.config["config_sha256"], campaign_id=self.config["campaign_id"])
        self.transport.write(0, superblock); self.transport.flush()
        catalog = {"schema_version": 1, "campaign_id": self.config["campaign_id"],
            "config_sha256": self.config["config_sha256"], "namespace_bytes": capacity,
            "header_before_sha256": before_sha, "cursor": SUPER_BYTES,
            "next_generation": 1, "commits": []}
        _write_json(self.catalog_path, catalog)
        return catalog

    def _write(self, offset, data):
        if offset < SUPER_BYTES or offset % ALIGN or offset + align(len(data)) > self.transport.total_bytes:
            raise ValueError('raw write outside admitted namespace')
        with self.lock:
            chunk=self.config['chunk_bytes']
            for cursor in range(0,len(data),chunk):
                self.transport.write(offset+cursor,data[cursor:cursor+chunk])

    def _receive_rank(self, sock, assignment, prefix):
        remaining = assignment["payload_bytes"]
        cursor = 0; digest = hashlib.sha256(); frame_digest = hashlib.sha256(prefix)
        while remaining:
            control, payload = wire.receive(sock, deadline=self.deadline,
                max_payload=self.config["chunk_bytes"])
            if (control.get("kind") != "chunk" or control.get("offset") != cursor or
                    not payload or len(payload) != control.get("bytes") or len(payload) > remaining or
                    hashlib.sha256(payload).hexdigest() != control.get("sha256")):
                raise ValueError("increment payload chunk differs")
            if len(payload) % ALIGN and len(payload) != remaining:
                raise ValueError('unaligned intermediate chunk')
            self._write(assignment["payload_offset"] + cursor, payload + bytes(align(len(payload)) - len(payload)))
            digest.update(payload); frame_digest.update(payload); cursor += len(payload); remaining -= len(payload)
        if digest.hexdigest() != assignment["payload_sha256"]:
            raise ValueError("increment payload digest differs")
        return frame_digest.hexdigest()

    def run(self, connections):
        operation = 0
        while True:
            operation_begin_ns=time.monotonic_ns()
            requests = []
            for rank, sock in sorted(connections.items()):
                control, descriptor_raw = wire.receive(sock, deadline=self.deadline, max_payload=256 << 20)
                if control == {"kind": "close", "rank": rank, "operation": operation}:
                    requests.append(None); continue
                if (control.get("kind") != "begin" or control.get("rank") != rank or
                        control.get("operation") != operation or control.get("payload_bytes", 0) <= 0 or
                        len(control.get("payload_sha256", "")) != 64):
                    raise ValueError("increment begin identity differs")
                descriptor = json.loads(descriptor_raw)
                requests.append((control, descriptor))
            if all(item is None for item in requests):
                for rank, sock in connections.items():
                    wire.send(sock, {"kind": "closed", "rank": rank}, b"",
                              deadline=self.deadline, max_payload=0)
                return
            if any(item is None for item in requests):
                raise ValueError("ranks disagree on close")
            common = {(item[0]["run_id"], item[0]["step"]) for item in requests}
            if len(common) != 1:
                raise ValueError("ranks disagree on run or step")
            generation = self.catalog["next_generation"]
            assignments = []
            prefixes = []
            cursor = self.catalog["cursor"]
            for rank, (control, descriptor) in enumerate(requests):
                prefix = pack_frame_prefix(generation=generation, step=control["step"],
                    descriptor=descriptor, payload_bytes=control["payload_bytes"],
                    payload_sha256=control["payload_sha256"])
                frame_offset = cursor; payload_offset = frame_offset + len(prefix)
                frame_bytes = len(prefix) + align(control["payload_bytes"])
                assignment = {"kind": "assigned", "generation": generation,
                    "frame_offset": frame_offset, "prefix_bytes": len(prefix),
                    "payload_offset": payload_offset, "payload_bytes": control["payload_bytes"],
                    "payload_sha256": control["payload_sha256"], "frame_bytes": frame_bytes}
                cursor += frame_bytes
                assignments.append(assignment)
                prefixes.append(prefix)
            if cursor + len(assignments) * ALIGN > self.transport.total_bytes:
                raise MemoryError('increment plus commits exceeds namespace capacity')
            prefix_begin_ns=time.monotonic_ns()
            for rank, (assignment, prefix) in enumerate(zip(assignments, prefixes)):
                self._write(assignment['frame_offset'], prefix)
                wire.send(connections[rank], assignment, b"", deadline=self.deadline, max_payload=0)
            payload_begin_ns=time.monotonic_ns()
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self._receive_rank, connections[rank], assignment, prefixes[rank])
                           for rank, assignment in enumerate(assignments)]
                frame_digests = [future.result() for future in futures]
            payload_end_ns=time.monotonic_ns()
            self.transport.flush()
            rows = []
            for rank, assignment in enumerate(assignments):
                frame_sha = frame_digests[rank]
                commit = pack_commit(generation=generation, frame_offset=assignment["frame_offset"],
                    frame_bytes=assignment["prefix_bytes"] + assignment["payload_bytes"],
                    frame_sha256=frame_sha)
                self._write(cursor, commit); commit_offset = cursor; cursor += ALIGN
                rows.append({"rank": rank, **assignment, "commit_offset": commit_offset,
                             "frame_sha256": frame_sha, "descriptor_sha256": hashlib.sha256(json.dumps(requests[rank][1],sort_keys=True,separators=(",",":")).encode()).hexdigest()})
            self.transport.flush()
            run_id, step = next(iter(common))
            receipt = {"generation": generation, "run_id": run_id, "step": step,
                       "ranks": rows, "committed_ns": time.monotonic_ns(),
                       "owner_timing":dict(begin_ns=operation_begin_ns,prefix_begin_ns=prefix_begin_ns,
                           payload_begin_ns=payload_begin_ns,payload_end_ns=payload_end_ns,
                           commit_end_ns=time.monotonic_ns())}
            if self.config.get('verify_before_ack'):
                from npu_nvme.experiments.verify_media import verify_receipt
                self.shadow.begin(step)
                receipt['media_verified']=verify_receipt(self.transport,receipt,chunk_bytes=self.config['chunk_bytes'],consume=self.shadow.consume)['status']=='pass'
                receipt['shadow_replay']=self.shadow.finish()
            self.catalog["commits"].append(receipt); self.catalog["cursor"] = cursor
            self.catalog["next_generation"] += 1; _write_json(self.catalog_path, self.catalog)
            for rank, sock in connections.items():
                wire.send(sock, {"kind": "committed", "receipt": receipt}, b"",
                          deadline=self.deadline, max_payload=0)
            operation += 1

    def close(self):
        self.transport.close(self.config["timeout_seconds"])


def serve(config, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    socket_path = Path(config["socket"]); socket_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX); server.bind(str(socket_path)); server.listen(4)
    owner = None; connections = {}; completed = False
    try:
        owner = RawOwner(config, output)
        _write_json(output / "ready.json", {"status": "ready", "namespace_bytes": owner.transport.total_bytes,
            "copy_bytes_per_tick":owner.transport.capabilities.copy_bytes_per_tick,
            "checksum_bytes_per_tick":owner.transport.capabilities.checksum_bytes_per_tick})
        while len(connections) < 4:
            connection, _ = server.accept()
            control, payload = wire.receive(connection, deadline=owner.deadline, max_payload=0)
            rank = control.get("rank")
            if payload or control != {"kind": "hello", "rank": rank} or rank in connections or rank not in range(4):
                raise ValueError("invalid rank hello")
            connections[rank] = connection
        owner.run(connections)
        completed = True
        _write_json(output / "result.json", {"status": "pass", "closed": False})
    finally:
        for connection in connections.values(): connection.close()
        server.close(); socket_path.unlink(missing_ok=True)
        if owner:
            owner.close()
            _write_json(output / "result.json", {"status": "pass" if completed else "fail", "closed": True,
                "commits": len(owner.catalog["commits"])})
