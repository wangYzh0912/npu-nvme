"""Collective periodic FULL checkpoints for the fixed TP4 training baseline."""
import json
import os
from pathlib import Path
import shutil
import struct
import time

from npu_nvme.runtime import training_catalog as catalog


class CheckpointController:
    def __init__(self, config, *, rank, npu_id, output):
        self.config=config;self.rank=rank;self.npu_id=npu_id
        self.output=Path(output);self.rank_dir=self.output/f'rank_{rank}'
        self.root=Path(config['checkpoint_root']);self.method=config['method']
        self.d2=None
        self.sequence=0

    def wait(self, path):
        deadline=time.monotonic()+self.config['operation_timeout_seconds']
        while not path.exists():
            if any(self.output.glob('failed-rank-*.json')):
                raise RuntimeError('another rank failed checkpoint operation')
            if time.monotonic()>=deadline:raise TimeoutError('checkpoint publication deadline')
            time.sleep(.05)
        return catalog.read_checked(path)

    def _d2(self):
        if self.d2 is None:
            from .training_checkpoint import TrainingCheckpoint
            config=json.loads(Path(self.config['d2_connection']).read_text())
            self.d2=TrainingCheckpoint(config,rank=self.rank,npu_id=self.npu_id,output=self.rank_dir)
        return self.d2

    def save(self, ms, network, *, step, state, controls, barrier):
        if self.method=='none':return None
        self.sequence+=1
        reservation_file=self.output/f'checkpoint-operation-{self.sequence:06d}.json'
        if self.rank==0:
            catalog.initialize(self.root,identity=self.config['identity'],method=self.method)
            if self.method!='ours':
                required=sum(row['bytes'] for row in state.values())*4+(1<<30)
                if shutil.disk_usage(self.root).free<required:
                    raise OSError('insufficient filesystem space for pending TP4 FULL checkpoint')
            path,row=catalog.reserve(self.root,run_id=self.config['run_id'],step=step)
            catalog.write_checked(reservation_file,dict(path=str(path),reservation=row))
        reserved=self.wait(reservation_file);path=Path(reserved['path']);row=reserved['reservation']
        if row['step']!=step or row['run_id']!=self.config['run_id']:
            raise ValueError('checkpoint reservation differs')
        rank_dir=path/f'rank_{self.rank}';rank_dir.mkdir(parents=True,exist_ok=False)
        controls=json.loads(json.dumps(controls))
        started=time.monotonic()
        ms.hal.synchronize()
        if self.method=='mindspore_native_save':
            payload=rank_dir/'native';payload.mkdir()
            # Cell serialization filters graph parameters using sliced/has_init.
            # set_data during restore changes these flags for small parameters.
            # Save the explicit, observed rank-local FULL inventory instead.
            entries=[];seen=set()
            for _,parameter in network.parameters_and_names():
                if id(parameter) in seen:continue
                seen.add(id(parameter))
                entries.append(dict(name=parameter.name,data=ms.Tensor(parameter.data)))
            if len(entries)!=len(state) or {entry['name'] for entry in entries}!=set(state):
                raise ValueError('native FULL source inventory differs')
            ms.save_checkpoint(entries,str(payload/'full.safetensors'),integrated_save=False,
                               async_save=False,format='safetensors')
            with (payload/'full.safetensors').open('rb') as stream:
                header_bytes=struct.unpack('<Q',stream.read(8))[0]
                if header_bytes>16<<20:raise ValueError('native FULL header exceeds budget')
                header=json.loads(stream.read(header_bytes))
            if set(header)-{'__metadata__'}!=set(state):
                raise ValueError('native FULL serialized inventory differs')
            for name,spec in state.items():
                entry=header[name]
                if entry['shape']!=spec['shape'] or entry['data_offsets'][1]-entry['data_offsets'][0]!=spec['bytes']:
                    raise ValueError('native FULL serialized geometry differs: '+name)
            for item in payload.iterdir():
                with item.open('rb') as stream:os.fsync(stream.fileno())
            catalog.fsync_directory(payload)
            backend=dict(kind='mindspore.save_checkpoint',format='safetensors')
        elif self.method=='bytecheckpoint_host':
            from qwen_host_checkpoint import checkpoint
            backend=checkpoint(ms,network,operation='save',rank=self.rank,out=path,
                controls=controls,generation=row['generation'],step=step,
                timeout=self.config['operation_timeout_seconds'])
        elif self.method=='ours':
            from qwen_d2_state import partitions_for
            partitions=partitions_for(state,controls['small_parameters'],self.config['strategy'])
            backend=self._d2().transfer(ms,network,operation='save',step=step,
                partitions=partitions,controls=controls)
        else:raise ValueError('unknown checkpoint method')
        # Metadata and hashing are included in the blocking save boundary.
        files=catalog.payload_files(rank_dir)
        complete=dict(rank=self.rank,generation=row['generation'],step=step,
            method=self.method,identity=self.config['identity'],state=state,controls=controls,
            backend=backend,files=files,save_seconds=time.monotonic()-started)
        catalog.write_checked(rank_dir/'complete.json',complete)
        barrier()
        if self.rank==0:
            catalog.publish(path,world=4)
            catalog.prune(self.root,self.config['retention'])
        committed=self.wait(path/'checkpoint.json')
        barrier()
        return dict(generation=row['generation'],step=step,path=str(path),
                    seconds=time.monotonic()-started,backend=backend,
                    committed=committed['status']=='committed')

    def restore(self, ms, network, *, data_sha256, lr_horizon, barrier):
        from qwen_native_state import parameter_manifest,compare_manifests,capture_control,restore_control
        path=Path(self.config['restore_checkpoint'])
        value=catalog.read_checked(path/'checkpoint.json')
        if value['identity']!=self.config['identity'] or value['method']!=self.method:
            raise ValueError('checkpoint training identity differs')
        row=value['ranks'][self.rank]
        if row['rank']!=self.rank:raise ValueError('checkpoint rank ordering differs')
        rank_dir=path/f'rank_{self.rank}'
        catalog.verify_files(rank_dir,row['files'])
        step=value['step'];expected=row['controls']
        if (expected['logical_optimizer_step']!=step or expected['next_data_row']!=step or
                expected['data_sha256']!=data_sha256 or expected['lr_horizon']!=lr_horizon):
            raise ValueError('checkpoint step/data/LR contract differs')
        def verify(controls, stored_step):
            if controls!=expected or stored_step!=step:
                raise ValueError('checkpoint restored controls differ')
            restore_control(ms,network,controls);ms.hal.synchronize()
            actual,small=parameter_manifest(network)
            compare_manifests(row['state'],actual)
            readback=capture_control(ms,step=step,data_sha256=data_sha256,lr_horizon=lr_horizon,small=small)
            if json.loads(json.dumps(readback))!=controls:
                raise ValueError('checkpoint restored RNG/control readback differs')
            catalog.write_checked(self.rank_dir/'restored-state.json',actual)
            catalog.write_checked(self.rank_dir/'restored-control.json',readback)
        if self.method=='mindspore_native_save':
            parameters=ms.load_checkpoint(str(rank_dir/'native/full.safetensors'),format='safetensors')
            missing,unexpected=ms.load_param_into_net(network,parameters,strict_load=True)
            if missing or unexpected:raise ValueError('native FULL checkpoint parameter coverage differs')
            verify(expected,step)
        elif self.method=='bytecheckpoint_host':
            from qwen_host_checkpoint import checkpoint
            checkpoint(ms,network,operation='restore',rank=self.rank,out=self.output,
                source=path,verify=verify,barrier=barrier,generation=value['generation'],step=step,
                timeout=self.config['operation_timeout_seconds'])
        elif self.method=='ours':
            from qwen_d2_state import partitions_for
            partitions=partitions_for(row['state'],expected['small_parameters'],self.config['strategy'])
            self._d2().transfer(ms,network,operation='restore',step=step,partitions=partitions,
                generation=row['backend']['receipt']['generation'],verify_controls=verify,barrier=barrier)
        else:raise ValueError('none cannot restore')
        barrier()
        return step

    def close(self):
        if self.d2:self.d2.close()
