"""Qwen callback-side weight scoring, selection, capture and raw submission."""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
from pathlib import Path
import socket
import threading
import time

import numpy as np

from npu_nvme.d2 import wire


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))
    temporary.replace(path)


def _wait(paths, deadline, failed):
    while not all(Path(path).exists() for path in paths):
        if any(Path(path).exists() for path in failed):
            raise RuntimeError("another rank failed incremental operation")
        if time.monotonic() >= deadline:
            raise TimeoutError("incremental rank coordination timeout")
        time.sleep(.01)


class IncrementalController:
    def __init__(self, ms, network, options, *, rank, output):
        from npu_nvme.framework.parameters import get_dev_ptr
        from npu_nvme.storage.bindings import load_backend
        from npu_nvme.experiments.device_score import make_score_cell
        from npu_nvme.experiments.tp_blocks import logical_fragments
        self.ms = ms; self.options = options; self.rank = rank; self.output = Path(output); self.network=network
        self.deadline = time.monotonic() + options["timeout_seconds"]
        schema = json.loads(Path(options["strategy"]).read_text())
        model_names = {row["name"] for row in schema["tensors"] if row.get("role") == "model"}
        registry = {}
        for _, parameter in network.parameters_and_names():
            if parameter.name in model_names and parameter.name not in registry:
                registry[parameter.name] = parameter
        if registry.keys() != model_names:
            missing = sorted(model_names - registry.keys())
            raise ValueError("incremental model inventory differs: " + repr(missing[:8]))
        self.parameters = registry
        self.block_elements = options["block_elements"]
        self.blocks = []
        self.references = {}; self.cells = {}; self.fragments = {}; self.granules = {}
        schemas = {row['name']: row for row in schema['tensors'] if row.get('role') == 'model'}
        self.candidate_count=sum(math.ceil(math.prod(row.get('global_shape',row['local_shape']))/self.block_elements)
            for row in schemas.values() if math.prod(row.get('global_shape',row['local_shape']))>=self.block_elements)
        for name, parameter in sorted(registry.items()):
            count = int(parameter.size); dtype = np.dtype(ms.dtype_to_nptype(parameter.dtype))
            if dtype != np.dtype(np.float32):
                raise ValueError("phase-one Qwen weights must be FP32")
            fragments = [row for row in logical_fragments(schemas[name], self.block_elements) if row['rank'] == rank]
            self.fragments[name] = fragments
            if not fragments: continue
            for row in fragments:
                self.blocks.append(dict(row, element_offset=row['local_element_offset'],
                    dtype=dtype.str, itemsize=dtype.itemsize))
            reference = ms.Parameter(ms.ops.zeros_like(parameter), requires_grad=False,
                                     name=f"incremental_reference_{rank}_{len(self.references)}")
            ms.ops.assign(reference, parameter); self.references[name] = reference
            if not fragments[0]['small']:
                granule = math.gcd(count, *(value for row in fragments for value in
                                           (row['local_element_offset'], row['element_count'])))
                self.granules[name] = granule
                if (count, granule) not in self.cells:
                    self.cells[count, granule] = make_score_cell(ms, count, granule)
        ms.runtime.synchronize()
        self.pointers = {name: get_dev_ptr(parameter, expected_device=rank)
                         for name, parameter in registry.items()}
        self.reference_pointers = {name: get_dev_ptr(reference, expected_device=rank)
                                   for name, reference in self.references.items()}
        backend = load_backend(options["library"]); self.acl = backend.acl_lib
        if self.acl.aclrtSetDevice(rank): raise RuntimeError("incremental aclrtSetDevice failed")
        self.stream = ctypes.c_void_p(); self.staging = ctypes.c_void_p()
        if self.acl.aclrtCreateStream(ctypes.byref(self.stream)):
            raise RuntimeError("incremental capture stream creation failed")
        self.staging_bytes = options["hbm_staging_bytes"]
        if self.acl.aclrtMalloc(ctypes.byref(self.staging), self.staging_bytes, 0):
            raise MemoryError("incremental HBM staging allocation failed")
        from npu_nvme.experiments.capture_timing import CaptureTiming
        chunks=2*math.ceil(sum(int(p.size)*4 for p in self.parameters.values())/self.staging_bytes)+2
        self.capture_timing=CaptureTiming(self.acl,self.stream,chunks)
        self.socket = socket.socket(socket.AF_UNIX); self.socket.connect(options["socket"])
        wire.send(self.socket, {"kind": "hello", "rank": rank}, b"",
                  deadline=self.deadline, max_payload=0)
        self.operation = 0; self.pending = None; self.timings = []
        self.memory=dict(reference_bytes=sum(int(p.size)*4 for p in self.references.values()),
                         acl_staging_bytes=self.staging_bytes,snapshot_bytes=0,
                         largest_per_parameter_difference_bytes=max(int(p.size)*4 for p in self.references.values()),
                         workspace_bytes=None,temporary_policy='per-parameter subtract and square; fusion not assumed')
        self.failed = False
        from npu_nvme.experiments.fidelity import Ages
        self.ages=Ages({(r['name'],r['block_index']) for r in self.blocks if not r['small']})

    def warmup(self):
        if self.options['group']=='B1':
            return
        for name, parameter in self.parameters.items():
            if name in self.granules:
                self.cells[int(parameter.size), self.granules[name]](parameter, self.references[name]).asnumpy()
        self.ms.runtime.synchronize()

    def _scores(self):
        started = time.monotonic_ns(); rows = [];device_ns=0
        for name, parameter in sorted(self.parameters.items()):
            if name not in self.granules: continue
            granule = self.granules[name]
            device_started=time.monotonic_ns()
            values = self.cells[int(parameter.size), granule](parameter, self.references[name]).asnumpy()
            device_ns+=time.monotonic_ns()-device_started
            if len(values)!=math.ceil(int(parameter.size)/granule):raise ValueError("local score geometry changed")
            partial = {}
            for fragment in self.fragments[name]:
                first = fragment['local_element_offset'] // granule
                count = fragment['element_count'] // granule
                index = fragment['block_index']
                partial[index] = partial.get(index, 0.0) + float(np.sum(values[first:first + count], dtype=np.float64))
            for index, value in sorted(partial.items()):
                if not math.isfinite(value): raise ValueError("nonfinite device block score")
                rows.append({"rank": self.rank, "name": name, "block_index": index, "score": float(value)})
        total=time.monotonic_ns()-started
        self.score_detail=dict(device_score_and_score_d2h_ns=device_ns,host_fragment_reduce_ns=total-device_ns)
        return rows,total

    def _select(self, step, scores):
        directory = self.output / "incremental-coordination" / f"step-{step:04d}"
        _write(directory / f"scores-{self.rank}.json", scores)
        paths = [directory / f"scores-{rank}.json" for rank in range(4)]
        failed = [self.output / f"failed-rank-{rank}.json" for rank in range(4)]
        _wait(paths, self.deadline, failed)
        selection = directory / "selection.json"
        if self.rank == 0:
            from npu_nvme.experiments.tp_blocks import aggregate_scores
            totals = aggregate_scores({rank: json.loads(path.read_text()) for rank, path in enumerate(paths)})
            combined = [dict(name=name, block_index=index, score=score) for (name,index),score in totals.items()]
            combined.sort(key=lambda row: (-row["score"], row["name"], row["block_index"]))
            count = int(math.ceil(self.options["ratio"] * len(combined)))
            _write(selection, {"candidate_blocks": len(combined), "selected_blocks": count,
                               "rows": combined[:count]})
        _wait([selection], self.deadline, failed)
        document = json.loads(selection.read_text())
        if document["candidate_blocks"]!=self.candidate_count:raise ValueError("global score coverage differs")
        selected = {(row["name"], row["block_index"]) for row in document["rows"]}
        return selected, document

    def _capture(self, selected):
        chosen = [row for row in self.blocks if row["small"] or
                  (row["name"], row["block_index"]) in selected]
        merged=[]
        for row in chosen:
            if (merged and merged[-1]['name']==row['name'] and
                    merged[-1]['element_offset']+merged[-1]['element_count']==row['element_offset'] and
                    (merged[-1]['element_count']+row['element_count'])*row['itemsize']<=self.staging_bytes):
                merged[-1]['element_count']+=row['element_count']
                merged[-1]['fragments'].append(dict(row))
            else:
                merged.append(dict(row,fragments=[dict(row)]))
        chosen=merged
        total = sum(row["element_count"] * row["itemsize"] for row in chosen)
        host = ctypes.c_void_p()
        if self.acl.aclrtMallocHost(ctypes.byref(host), total):
            raise MemoryError("incremental pinned output allocation failed")
        records = []; host_cursor = staging_cursor = 0
        self.capture_timing.reset()
        try:
            for row in chosen:
                byte_count = row["element_count"] * row["itemsize"]
                if staging_cursor and staging_cursor + byte_count > self.staging_bytes:
                    self.capture_timing.mark()
                    if self.acl.aclrtMemcpyAsync(ctypes.c_void_p(host.value + host_cursor - staging_cursor),
                            staging_cursor, self.staging, staging_cursor, 2, self.stream):
                        raise RuntimeError("incremental staged D2H failed")
                    self.capture_timing.mark()
                    staging_cursor = 0
                if not staging_cursor:self.capture_timing.mark()
                source = self.pointers[row["name"]] + row["element_offset"] * row["itemsize"]
                if self.acl.aclrtMemcpyAsync(ctypes.c_void_p(self.staging.value + staging_cursor),
                        byte_count, ctypes.c_void_p(source), byte_count, 3, self.stream):
                    raise RuntimeError("incremental D2D pack failed")
                records.append(dict(row, payload_offset=host_cursor, payload_bytes=byte_count))
                staging_cursor += byte_count; host_cursor += byte_count
                if staging_cursor == self.staging_bytes:
                    self.capture_timing.mark()
                    if self.acl.aclrtMemcpyAsync(ctypes.c_void_p(host.value + host_cursor - staging_cursor),
                            staging_cursor, self.staging, staging_cursor, 2, self.stream):
                        raise RuntimeError("incremental staged D2H failed")
                    self.capture_timing.mark()
                    staging_cursor = 0
            if staging_cursor:
                self.capture_timing.mark()
                if self.acl.aclrtMemcpyAsync(ctypes.c_void_p(host.value + host_cursor - staging_cursor),
                        staging_cursor, self.staging, staging_cursor, 2, self.stream):
                    raise RuntimeError("incremental final D2H failed")
                self.capture_timing.mark()
            if self.acl.aclrtSynchronizeStream(self.stream):
                raise RuntimeError("incremental capture synchronization failed")
            self.capture_detail=self.capture_timing.result()
            view = memoryview((ctypes.c_ubyte * total).from_address(host.value)).cast('B')
            return host, view, records
        except BaseException:
            self._release_after_stop(host)
            raise

    def _release_after_stop(self, host):
        if self.acl.aclrtSynchronizeStream(self.stream):
            self.failed = True
            self.retained_host = host
            _write(self.output / f'incremental-rank-{self.rank}-status.json',
                   {'status': 'retained', 'reason': 'DMA stream stop not proven', 'rank': self.rank})
            raise RuntimeError('DMA stream stop not proven; allocation retained')
        if self.acl.aclrtFreeHost(host):
            raise RuntimeError('pinned buffer release failed')

    def _submit(self, step, host, view, descriptor, payload_sha):
        try:
            serialize_begin=time.monotonic_ns()
            raw = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
            self.pending['serialization_ns']=time.monotonic_ns()-serialize_begin
            self.pending['descriptor_bytes']=len(raw)
            assignment_begin=time.monotonic_ns()
            control = {"kind": "begin", "rank": self.rank, "operation": self.operation,
                "run_id": self.options["run_id"], "step": step, "payload_bytes": len(view),
                "payload_sha256": payload_sha}
            wire.send(self.socket, control, raw, deadline=self.deadline, max_payload=256 << 20)
            assigned, payload = wire.receive(self.socket, deadline=self.deadline, max_payload=0)
            if payload or assigned.get("kind") != "assigned": raise ValueError("raw assignment differs")
            if (assigned.get("payload_bytes") != len(view) or assigned.get("payload_offset", -1) < 0 or
                    assigned.get("frame_bytes", 0) < assigned.get("prefix_bytes", 0) + len(view)):
                raise ValueError("raw assignment geometry differs")
            self.pending['begin_and_assignment_ns']=time.monotonic_ns()-assignment_begin
            chunks_begin=time.monotonic_ns()
            chunk = self.options["chunk_bytes"]
            for offset in range(0, len(view), chunk):
                data = bytes(view[offset:offset + chunk])
                wire.send(self.socket, {"kind": "chunk", "offset": offset, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest()}, data,
                    deadline=self.deadline, max_payload=chunk)
            self.pending['chunk_send_ns']=time.monotonic_ns()-chunks_begin
            ack_begin=time.monotonic_ns()
            committed, payload = wire.receive(self.socket, deadline=self.deadline, max_payload=0)
            self.pending['ack_wait_ns']=time.monotonic_ns()-ack_begin
            if payload or committed.get("kind") != "committed": raise ValueError("raw commit ACK differs")
            receipt = committed.get('receipt', {})
            if (receipt.get('run_id') != self.options['run_id'] or receipt.get('step') != step or
                    receipt.get('generation') != assigned['generation']):
                raise ValueError('raw commit lineage differs')
            ranks = [row for row in receipt.get('ranks', []) if row.get('rank') == self.rank]
            if len(ranks) != 1 or ranks[0].get('payload_sha256') != payload_sha:
                raise ValueError('raw commit payload identity differs')
            if ranks[0].get('descriptor_sha256') != hashlib.sha256(raw).hexdigest():
                raise ValueError('raw committed descriptor identity differs')
            if self.options.get('canonical') and receipt.get('media_verified') is not True:
                raise ValueError('canonical payload lacks media readback')
            self.pending["receipt"] = committed["receipt"]
            self.pending['committed_ns']=time.monotonic_ns()
            self.pending['owner_committed_ns']=receipt['committed_ns']
            _write(self.output/f'rank_{self.rank}'/'incremental-receipts'/f'step-{step:04d}.json',
                   dict(step=step,receipt=receipt,trigger_ns=self.pending['trigger_ns'],
                        committed_ns=self.pending['committed_ns']))
        except BaseException as error:
            self.pending["error"] = error

    def _finalize(self):
        if self.pending is None: return 0
        started = time.monotonic_ns(); pending = self.pending
        try:
            pending["thread"].join()
            if pending.get("error"): raise pending["error"]
            reference_started=time.monotonic_ns()
            for row in pending["records"]:
                source = pending["host"].value + row["payload_offset"]
                target = self.reference_pointers[row["name"]] + row["element_offset"] * row["itemsize"]
                if self.acl.aclrtMemcpyAsync(ctypes.c_void_p(target), row["payload_bytes"],
                        ctypes.c_void_p(source), row["payload_bytes"], 1, self.stream):
                    raise RuntimeError("incremental reference H2D failed")
            if self.acl.aclrtSynchronizeStream(self.stream):
                raise RuntimeError("incremental reference synchronization failed")
            self.operation += 1
            _write(self.output/f'rank_{self.rank}'/'incremental-completions'/f"step-{pending['step']:04d}.json",
                   dict(step=pending['step'],trigger_ns=pending['trigger_ns'],
                        committed_ns=pending['committed_ns'],reference_ready_ns=time.monotonic_ns(),
                        reference_update_ns=time.monotonic_ns()-reference_started,
                        wait_ns=reference_started-started,
                        updated_bytes=sum(r['payload_bytes'] for r in pending['records']),
                        wait_and_reference_ns=time.monotonic_ns()-started,
                        transport={key:pending.get(key) for key in ('serialization_ns','descriptor_bytes',
                            'begin_and_assignment_ns','chunk_send_ns','ack_wait_ns')}))
            return time.monotonic_ns() - started
        finally:
            self._release_after_stop(pending["host"])
            self.pending = None

    def save(self, logical_step):
        started = time.monotonic_ns()
        anchor_wall=time.time_ns();anchor_after=time.monotonic_ns()
        phase_intervals=[]
        self.ms.runtime.synchronize()
        stable_ns=time.monotonic_ns()-started
        phase_intervals.append(dict(name='source_stable',start_ns=started,end_ns=time.monotonic_ns()))
        phase_begin=time.monotonic_ns()
        wait_ns = self._finalize()
        phase_intervals.append(dict(name='wait_previous_and_reference',start_ns=phase_begin,end_ns=time.monotonic_ns()))
        phase_begin=time.monotonic_ns()
        if self.options["group"] == "B1":
            scores = []; selected = {(row["name"], row["block_index"]) for row in self.blocks if not row["small"]}
            selection = {"candidate_blocks": self.candidate_count, "selected_blocks": self.candidate_count, "rows": []}
            score_ns = select_ns = 0
        else:
            scores, score_ns = self._scores(); before = time.monotonic_ns()
            phase_intervals.append(dict(name='score',start_ns=phase_begin,end_ns=before))
            selected, selection = self._select(logical_step, scores); select_ns = time.monotonic_ns() - before
            phase_intervals.append(dict(name='selection',start_ns=before,end_ns=time.monotonic_ns()))
        before = time.monotonic_ns(); host, view, records = self._capture(selected)
        capture_ns = time.monotonic_ns() - before
        phase_intervals.append(dict(name='capture',start_ns=before,end_ns=time.monotonic_ns()))
        before=time.monotonic_ns();payload_sha = hashlib.sha256(view).hexdigest();checksum_ns=time.monotonic_ns()-before
        phase_intervals.append(dict(name='payload_checksum',start_ns=before,end_ns=time.monotonic_ns()))
        descriptor = {"schema_version": 1, "group": self.options["group"], "ratio": self.options.get("ratio"),
            "rank": self.rank, "logical_step": logical_step, "records": records,
            "candidate_blocks": selection["candidate_blocks"], "selected_blocks_global": selection["selected_blocks"]}
        self.pending = {"host": host, "view": view, "records": records, "error": None, "receipt": None,
                        'step':logical_step,'trigger_ns':started}
        thread = threading.Thread(target=self._submit,
            args=(logical_step, host, view, descriptor, payload_sha), daemon=True)
        self.pending["thread"] = thread; thread.start()
        row = {"step": logical_step, "wait_previous_ns": wait_ns, "score_ns": score_ns,
            "source_stable_ns": stable_ns,
            "select_ns": select_ns, "capture_ns": capture_ns, "critical_ns": time.monotonic_ns() - started,
            "checksum_ns": checksum_ns,
            "payload_bytes": len(view), "saved_blocks": len(records),
            "small_bytes": sum(item["payload_bytes"] for item in records if item["small"]),
            "pending_requests": 1}
        row['phase_intervals']=phase_intervals
        row['clock_anchor']=dict(monotonic_before_ns=started,wall_ns=anchor_wall,monotonic_after_ns=anchor_after)
        row['block_age']=self.ages.advance(logical_step,selected & self.ages.saved.keys())
        row['logical_selected_blocks_global']=selection['selected_blocks']
        row['packed_ranges']=len(records)
        row['host_buffer_bytes']=len(view)
        row['hbm_output_bytes']=self.staging_bytes
        row['memory_budget']=self.memory
        row.update(getattr(self,"score_detail",{}))
        row.update(getattr(self,"capture_detail",{}))
        self.timings.append(row)
        if self.options.get('canonical'):
            self.last_selected=selected
            self.validate_state(logical_step)
        return row

    def media_shadow(self, name):
        directory=Path(self.options['media_shadow'])/f'rank_{self.rank}'
        if not hasattr(self,'shadow_index'):self.shadow_index=json.loads((directory/'index.json').read_text())
        return np.load(directory/self.shadow_index[name],mmap_mode='r',allow_pickle=False)

    def validate_state(self, step):
        """Compare committed reference against complete current weights outside timing."""
        from npu_nvme.experiments.fidelity import fragment_error, reduce_error
        self._finalize()
        ready=json.loads((Path(self.options['media_shadow'])/'ready.json').read_text())
        if ready['step']!=step:raise ValueError('media shadow step differs')
        totals={}
        for name, reference in self.references.items():
            current=self.parameters[name].asnumpy()
            shadow=self.media_shadow(name)
            if not np.array_equal(shadow,reference.asnumpy()):raise ValueError('device reference differs from decoded media shadow: '+name)
            for fragment in self.fragments[name]:
                if fragment['small'] or (name,fragment['block_index']) in self.last_selected:
                    start=fragment['local_element_offset'];end=start+fragment['element_count']
                    if not np.array_equal(shadow.reshape(-1)[start:end],current.reshape(-1)[start:end]):
                        raise ValueError('committed selected block differs from stable source: '+name)
            totals.update(fragment_error({name:shadow},{name:current},self.fragments[name]))
        directory=self.output/'fidelity'/f'step-{step:04d}'
        _write(directory/f'rank-{self.rank}.json',totals)
        paths=[directory/f'rank-{rank}.json' for rank in range(4)]
        _wait(paths,self.deadline,[self.output/f'failed-rank-{rank}.json' for rank in range(4)])
        if self.rank==0:
            result=reduce_error([json.loads(path.read_text()) for path in paths])
            result.update(step=step,source='independent shadow decoded from actual raw media readback',
                          loss_difference=None,loss_difference_status='not_measured')
            result['block_age_by_rank_pending']=True
            _write(directory/'result.json',result)
        from npu_nvme.experiments.evaluate import compare
        evaluation=compare(self,self.network)
        # The next pure-training loss observes exactly this post-update state.
        # This also detects a forward graph accidentally reading stale aliases.
        baseline=self.output.parent/(self.options['model_role']+'-b0-rep0')/'rank_0'/'training.json'
        if step<20 and baseline.exists():
            from npu_nvme.runtime.training_catalog import read_checked
            oracle=read_checked(baseline)
            if oracle.get('status')=='pass':
                expected=oracle['losses'][self.options['warmup_steps']+step]['loss']
                evaluation['expected_next_training_loss']=expected
                evaluation['forward_vs_training_absolute_difference']=abs(evaluation['full_loss']-expected)
                if abs(evaluation['full_loss']-expected)>1e-6+1e-5*abs(expected):
                    raise ValueError('full forward evaluation differs from pure-training post-update state')
        evaluation['block_age']=self.timings[-1]['block_age']
        _write(directory/f'loss-rank-{self.rank}.json',evaluation)
        loss_paths=[directory/f'loss-rank-{rank}.json' for rank in range(4)]
        _wait(loss_paths,self.deadline,[self.output/f'failed-rank-{rank}.json' for rank in range(4)])
        if self.rank==0:
            values=[json.loads(path.read_text()) for path in loss_paths]
            result.update(loss_difference=values[0]['loss_difference'],loss_difference_status='measured',
                          evaluation_by_rank=values,maximum_block_age=max(v['block_age']['maximum'] for v in values))
            result.pop('block_age_by_rank_pending',None)
            _write(directory/'result.json',result)

    def close(self):
        if getattr(self,'closed',False):raise RuntimeError('incremental controller was already closed')
        self.closed=True
        try:
            drain = self._finalize()
            wire.send(self.socket, {"kind": "close", "rank": self.rank, "operation": self.operation}, b"",
                      deadline=self.deadline, max_payload=0)
            reply, payload = wire.receive(self.socket, deadline=self.deadline, max_payload=0)
            if payload or reply != {"kind": "closed", "rank": self.rank}:
                raise ValueError("raw owner close differs")
            return drain
        finally:
            self.socket.close()
            if not self.failed:
                self.capture_timing.close()
                if self.staging.value: self.acl.aclrtFree(self.staging)
                if self.stream.value: self.acl.aclrtDestroyStream(self.stream)
