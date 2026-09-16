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

from npu_nvme.runtime.training_catalog import write_checked


def run(options):
    rank=int(os.environ.get('RANK_ID','0'));out=Path(options['output'])
    directory=out/f'rank_{rank}';directory.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',rank=rank,pid=os.getpid(),losses=[],checkpoints=[],
                method=options['method'],begin_ns=time.monotonic_ns())
    controller=None
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
        if is_legacy_model() or get_group_size()!=4 or get_rank()!=rank or ms.get_context('device_id')!=rank:
            raise RuntimeError('baseline requires TP4 on physical NPU 0–3')
        ms.set_seed(42);np.random.seed(42);random.seed(42)
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
            def on_train_begin(self,context):
                network=context.original_args().train_network
                if options['method']=='ours':controller._d2()._open()
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
                record()

            def on_train_step_end(self,context):
                cb=context.original_args();outputs=cb.net_outputs
                loss=outputs[0] if isinstance(outputs,(tuple,list)) else outputs
                loss=float(np.asarray(loss.asnumpy()).mean())
                overflow=bool(np.asarray(outputs[1].asnumpy()).any()) if isinstance(outputs,(tuple,list)) else False
                if not np.isfinite(loss) or overflow:raise RuntimeError('nonfinite loss or skipped optimizer update')
                step=start+int(cb.cur_step_num)
                if step!=start+len(report['losses'])+1:raise ValueError('callback optimizer step is not sequential')
                report['losses'].append(dict(step=step,loss=loss,overflow=overflow))
                if step%period==0 or step==stop:
                    ms.hal.synchronize();state,small=parameter_manifest(cb.train_network)
                    controls=capture_control(ms,step=step,data_sha256=data_hash,lr_horizon=horizon,small=small)
                    write_checked(directory/f'state-step-{step}.json',state)
                    write_checked(directory/f'control-step-{step}.json',controls)
                    # The final step is observed even when it is not a save boundary.
                    if step%period==0:
                        saved=controller.save(ms,cb.train_network,step=step,state=state,controls=controls,barrier=barrier)
                        if saved:report['checkpoints'].append(saved)
                record()
                print(f'QWEN_TRAIN_STEP rank={rank} step={step} loss={loss:.8f}',flush=True)

        trainer=Trainer(args=config,train_dataset=dataset,callbacks=[Progress()])
        trainer.finetune(resume_from_checkpoint=str(source),auto_trans_ckpt=True)
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
        if controller:controller.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    return run(json.loads(args.config.read_text()))


if __name__=='__main__':sys.exit(main())
