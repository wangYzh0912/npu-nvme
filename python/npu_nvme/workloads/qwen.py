"""Qwen3-8B TP4 fixed-sample training with periodic FULL restart support."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from contextlib import nullcontext

from npu_nvme.runtime.training_catalog import write_checked


def records_state(*,step,stop,period,method,experiment):
    return step==stop or (step%period==0 and not (
        method=='none' and (experiment or {}).get('skip_periodic_state')))


def run(options):
    rank=int(os.environ.get('RANK_ID','0'));out=Path(options['output'])
    directory=out/f'rank_{rank}';directory.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',rank=rank,pid=os.getpid(),losses=[],checkpoints=[],
                method=options['method'],begin_ns=time.monotonic_ns())
    controller=None
    experiment=options.get('io_experiment',{})
    closure=experiment.get('closure',False)
    if closure:report['lifecycle']={}
    def record():write_checked(directory/'training.json',report)
    record()
    try:
        import numpy as np
        import mindspore as ms
        import mindformers
        from mindformers import Trainer,build_context
        from mindformers.tools.register import MindFormerConfig
        from mindformers.core.context import is_legacy_model
        from mindspore.communication import get_rank,get_group_size
        from mindspore.communication.comm_func import barrier
        from transformers import AutoTokenizer
        from npu_nvme.framework.checkpoint_controller import CheckpointController
        from qwen_native_state import parameter_manifest,capture_control

        if ms.__version__!='2.7.1' or not mindformers.__version__.startswith('1.7.'):
            raise RuntimeError('baseline requires validated MindSpore 2.7.1 / MindFormers 1.7')
        source=Path(options['model']);model=json.loads((source/'config.json').read_text())
        if (model.get('model_type'),model.get('hidden_size'),model.get('num_hidden_layers'))!=('qwen3',4096,36):
            raise ValueError('baseline requires full Qwen3-8B')
        start=options['start_step'];stop=options['stop_step'];horizon=options['lr_horizon']
        sequence=options['seq_length'];period=options['checkpoint_interval']
        template=Path(mindformers.__file__).parent.parent/'configs/qwen3/finetune_qwen3.yaml'
        config=MindFormerConfig(str(template))
        config.output_dir=str(out/'training');config.pretrained_model_dir=str(source)
        config.load_checkpoint=str(source);config.use_legacy=False;config.run_mode='finetune';config.seed=42
        config.parallel_config.update(data_parallel=1,model_parallel=4,pipeline_stage=1,
            micro_batch_num=1,vocab_emb_dp=False,use_seq_parallel=True)
        config.context.ascend_config.pop('parallel_speed_up_json_path',None)
        config.context.deterministic='ON'
        config.parallel.strategy_ckpt_config.save_file=str(directory/'strategy.ckpt')
        config.model.model_config.update(seq_length=sequence,batch_size=1,params_dtype='float32',
            compute_dtype='bfloat16',residual_dtype='float32',input_sliced_sig=True)
        config.runner_config.update(epochs=1,sink_mode=True,sink_size=1,stop_step=stop-start)
        config.lr_schedule.total_steps=horizon
        config.train_dataset.data_loader=MindFormerConfig(type='GeneratorDataset')
        config.callbacks=[c for c in config.callbacks if c.get('type')!='CheckpointMonitor']
        # CheckpointController owns periodic publication for all backends,
        # including synchronous native MindSpore safetensors serialization.
        write_checked(directory/'resolved-config.json',dict(config))
        build_context(config)
        physical_devices=options['identity']['physical_npu_ids']
        if (is_legacy_model() or get_group_size()!=4 or get_rank()!=rank or
                ms.get_context('device_id')!=rank or physical_devices not in ([0,1,2,3],[4,5,6,7]) or
                os.environ['ASCEND_RT_VISIBLE_DEVICES']!=','.join(map(str,physical_devices))):
            raise RuntimeError('baseline TP4 physical device mapping differs')
        ms.set_seed(42);ms.manual_seed(42);np.random.seed(42);random.seed(42)
        tokenizer=AutoTokenizer.from_pretrained(str(source),local_files_only=True,trust_remote_code=False)
        text=('Checkpoint recovery restores model parameters and optimizer state. '
              'Tensor parallel training divides the model across accelerator devices. ')
        tokens=tokenizer.encode(text*(sequence//8+2),add_special_tokens=False)
        ids=np.asarray(tokens[:sequence+1],dtype=np.int32)
        if ids.size!=sequence+1:raise ValueError('fixed token fixture too short')
        data_hash=hashlib.sha256(ids.tobytes()).hexdigest()
        report['data_sha256']=data_hash
        np.save(directory/'input_ids.npy',ids,allow_pickle=False)
        row=(ids[:-1],ids[1:],np.ones(sequence,np.float32),np.arange(sequence,dtype=np.int32),
             np.triu(np.ones((1,sequence,sequence),np.uint8),k=1))
        dataset=ms.dataset.GeneratorDataset([row]*(stop-start),
            column_names=['input_ids','labels','loss_mask','position_ids','attention_mask'],
            shuffle=False,num_parallel_workers=1).batch(1,drop_remainder=True)
        controller=CheckpointController(options,rank=rank,npu_id=rank,output=out)

        class Progress(ms.Callback):
            def on_train_step_begin(self,context):
                if experiment.get('device_profile') and start+len(report['losses'])+1==28:
                    from mindspore.profiler import Profiler,ProfilerLevel,AicoreMetrics
                    self.profiler=Profiler(start_profile=False,output_path=str(directory/'device-profile'),
                        profiler_level=ProfilerLevel.Level1,aic_metrics=AicoreMetrics.PipeUtilization)
                    report['device_profile']=dict(status='running',steps=[28,32],
                        collection_range='full callbacks for steps 28..32, including saves',
                        requested=['AscendCL','NPU task timing','PipeUtilization MAC/vector/MTE'],
                        limitations=['one PMU group; HBM bandwidth group not collected',
                                      'no late-start GE allocation trace; resource samples are separate'],
                        start_anchor=dict(monotonic_ns=time.monotonic_ns(),realtime_ns=time.time_ns()))
                    self.profiler.start()
                self.step_begin_ns=time.monotonic_ns()
                if closure:report['lifecycle'].setdefault('first_step_begin_ns',self.step_begin_ns)

            def on_train_begin(self,context):
                if closure:report['lifecycle']['initial_validation_begin_ns']=time.monotonic_ns()
                network=context.original_args().train_network
                if options['method'] in ('ours','checkpoint_io_sink'):controller._d2()._open()
                controller.admit(ms,network,step=start,operation='restore' if options.get('restore_checkpoint') else 'initial',barrier=barrier)
                if options.get('restore_checkpoint'):
                    restored=controller.restore(ms,network,data_sha256=data_hash,lr_horizon=horizon,barrier=barrier)
                    if restored!=start:raise ValueError('restored step differs from execution start')
                    report['restore_global_ready_ns']=time.monotonic_ns()
                    write_checked(out/f'restore-ready-{rank}.json',dict(rank=rank,run_id=options['run_id'],step=start))
                    released=controller.wait(out/'restore-pin-released.json')
                    if released!={'run_id':options['run_id']}:raise ValueError('restore release identity differs')
                else:
                    state,small=parameter_manifest(network)
                    write_checked(directory/'state-initial.json',state)
                    write_checked(directory/'control-initial.json',capture_control(ms,step=0,
                        data_sha256=data_hash,lr_horizon=horizon,small=small))
                if closure:report['lifecycle']['initial_validation_end_ns']=time.monotonic_ns()
                record()

            def on_train_step_end(self,context):
                cb=context.original_args();outputs=cb.net_outputs
                loss=outputs[0] if isinstance(outputs,(tuple,list)) else outputs
                loss=float(np.asarray(loss.asnumpy()).mean())
                overflow=bool(np.asarray(outputs[1].asnumpy()).any()) if isinstance(outputs,(tuple,list)) else False
                if not np.isfinite(loss) or overflow:raise RuntimeError('nonfinite loss or skipped optimizer update')
                step=start+int(cb.cur_step_num)
                if step!=start+len(report['losses'])+1:raise ValueError('callback optimizer step is not sequential')
                begin_ns=getattr(self,'step_begin_ns',None)
                end_ns=time.monotonic_ns()
                report['losses'].append(dict(step=step,loss=loss,overflow=overflow,
                                            begin_ns=begin_ns,end_ns=end_ns))
                experiment=options.get('io_experiment')
                if experiment and experiment.get('event_root'):
                    from npu_nvme.experiments.checkpoint_io import emit
                    emit(str(Path(experiment['event_root'])/f'rank-{rank}.jsonl'),event='training_step',run_id=options['run_id'],
                         rank=rank,physical_npu=options['identity']['physical_npu_ids'][rank],step=step,
                         byte_count=None,status='ok')
                should_record_state=records_state(step=step,stop=stop,period=period,
                    method=options['method'],experiment=experiment)
                # Final oracle validation is outside T0's measured training window.
                report['losses'][-1]['resume_ns']=end_ns
                if should_record_state:
                    capture_begin=time.monotonic_ns()
                    boundary=None
                    if closure:
                        from npu_nvme.experiments.checkpoint_io_closure import Boundary
                        boundary=Boundary(step);capture_begin=boundary.row['begin_ns']
                    stage=boundary.stage if boundary else lambda name:nullcontext()
                    with stage('admission_collectives'):
                        controller.admit(ms,cb.train_network,step=step,operation='save',barrier=barrier)
                    with stage('framework_synchronize'):ms.hal.synchronize()
                    with stage('manifest_read_hash') as measured:
                        measurement={} if closure and experiment.get('events') else None
                        state,small=parameter_manifest(cb.train_network,measurement=measurement)
                        if measurement is not None:measured['aggregate']=measurement
                    with stage('control_capture'):
                        controls=capture_control(ms,step=step,data_sha256=data_hash,lr_horizon=horizon,small=small)
                    with stage('sidecar_write'):
                        write_checked(directory/f'state-step-{step}.json',state)
                        write_checked(directory/f'control-step-{step}.json',controls)
                    if closure and step==stop:
                        report['lifecycle'].update(final_validation_begin_ns=capture_begin,
                            final_validation_end_ns=time.monotonic_ns(),
                            final_validation_scope='manifest/control validation; included in final save when applicable')
                    # The final step is observed even when it is not a save boundary.
                    if step%period==0 and options['method']!='none':
                        saved=controller.save(ms,cb.train_network,step=step,state=state,controls=controls,
                                              barrier=barrier,boundary=boundary)
                        if saved:
                            saved['blocking_seconds']=(time.monotonic_ns()-capture_begin)/1e9
                            report['checkpoints'].append(saved)
                            report['losses'][-1]['resume_ns']=time.monotonic_ns()
                    if boundary:
                        full=boundary.finish()
                        full['operation']='save' if step%period==0 and options['method']!='none' else 'final_validation'
                        report.setdefault('boundaries',[]).append(full)
                        if full['operation']=='final_validation':
                            report['lifecycle'].update(final_validation_begin_ns=full['begin_ns'],
                                                       final_validation_end_ns=full['end_ns'])
                        else:
                            saved['blocking_seconds']=(full['end_ns']-full['begin_ns'])/1e9
                            report['losses'][-1]['resume_ns']=full['end_ns']
                if closure:report['lifecycle']['last_step_resume_ns']=report['losses'][-1]['resume_ns']
                if experiment and experiment.get('event_root'):
                    emit(str(Path(experiment['event_root'])/f'rank-{rank}.jsonl'),event='training_resume',
                         run_id=options['run_id'],rank=rank,
                         physical_npu=options['identity']['physical_npu_ids'][rank],step=step)
                if experiment.get('device_profile') and step==32:
                    self.profiler.stop()
                    report['device_profile'].update(status='captured',
                        stop_anchor=dict(monotonic_ns=time.monotonic_ns(),realtime_ns=time.time_ns()))
                    self.profiler.analyse()
                record()
                print(f'QWEN_TRAIN_STEP rank={rank} step={step} loss={loss:.8f}',flush=True)

        if closure:report['lifecycle']['trainer_init_begin_ns']=time.monotonic_ns()
        trainer=Trainer(args=config,train_dataset=dataset,callbacks=[Progress()])
        if closure:
            report['lifecycle']['trainer_init_end_ns']=time.monotonic_ns()
            report['lifecycle']['finetune_begin_ns']=time.monotonic_ns()
        trainer.finetune(resume_from_checkpoint=str(source),auto_trans_ckpt=True)
        if closure:report['lifecycle']['finetune_end_ns']=time.monotonic_ns()
        if len(report['losses'])!=stop-start:raise RuntimeError('training stopped before requested optimizer step')
        report.update(status='pass',end_ns=time.monotonic_ns(),final_step=stop)
        record()
        return 0
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        write_checked(out/f'failed-rank-{rank}.json',dict(rank=rank,error=repr(error)))
        record();traceback.print_exc()
        return 1
    finally:
        if closure:report['lifecycle']['close_begin_ns']=time.monotonic_ns()
        try:
            if controller:controller.close()
        except BaseException as error:
            report.update(status='failed',close_error=repr(error));record();raise
        if closure:
            report['lifecycle']['safe_close_end_ns']=time.monotonic_ns()
            report['closed']=True;record()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    return run(json.loads(args.config.read_text()))


if __name__=='__main__':sys.exit(main())
