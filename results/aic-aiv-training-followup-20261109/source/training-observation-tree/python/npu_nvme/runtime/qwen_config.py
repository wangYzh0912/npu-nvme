"""Explicit, portable run settings for the Qwen3 training baseline."""
import hashlib
import json
from pathlib import Path

from npu_nvme.d2.tp_schema import digest,validate

METHODS=('none','mindspore_native_save','ours','bytecheckpoint_host')


def resolve(value, repo):
    repo=Path(repo).resolve();value=dict(value)
    allowed={'schema_version','model','environment_manifest','method','stop_step','lr_horizon',
             'seq_length','checkpoint_interval','retention','timeout_seconds','operation_timeout_seconds',
             'copy_timeout_ms','shm_base','strategy','region','hardware_lock','checkpoint_root',
             'host_headroom_bytes','hbm_headroom_bytes','npu_ids','activity_profile',
             'micro_batch','jit_level'}
    if set(value)-allowed:raise ValueError('unknown training settings: '+repr(sorted(set(value)-allowed)))
    if value.get('schema_version')!=2:raise ValueError('Qwen training configuration requires schema_version 2')
    defaults=dict(model='/models/Qwen3-8B',method='ours',stop_step=24,lr_horizon=24,
        seq_length=128,checkpoint_interval=4,retention=3,timeout_seconds=14400,
        operation_timeout_seconds=1800,copy_timeout_ms=30000,shm_base=79000,
        strategy='config/qwen_runtime_schema.json',region='config/d2_qwen_region.json',
        hardware_lock='/models/npu_nvme_exp/user7-stack/hardware-campaign.lock',
        host_headroom_bytes=16<<30,hbm_headroom_bytes=1<<30,micro_batch=1,jit_level='O0')
    config=dict(defaults,**value)
    config.setdefault('npu_ids',[0,1,2,3])
    profile=config.get('activity_profile')
    if profile is not None:
        if (not isinstance(profile,dict) or set(profile)-{'start_step','stop_step','aic_metrics','profiler_level','activities'} or
                not {'start_step','stop_step'}<=set(profile) or
                any(type(profile[k]) is not int or profile[k]<1 for k in ('start_step','stop_step')) or
                profile['stop_step']<=profile['start_step'] or
                profile['stop_step']>=config['stop_step']):
            raise ValueError('activity_profile requires an interior, nonempty step interval')
        if profile.get('aic_metrics','pipe') not in ('pipe','none'):raise ValueError('unsupported activity PMU mode')
        if profile.get('profiler_level',1) not in (0,1):raise ValueError('unsupported activity profiler level')
        if profile.get('activities','cpu_npu') not in ('cpu_npu','npu'):raise ValueError('unsupported activity collection scope')
        if config['method']!='none':raise ValueError('activity profiling requires method=none')
    if config['npu_ids'] not in ([0,1,2,3],[4,5,6,7]):
        raise ValueError('Qwen TP4 mapping must be physical NPU 0-3 or 4-7')
    if config['method'] not in METHODS:raise ValueError('unknown checkpoint method')
    for name in ('host_headroom_bytes','hbm_headroom_bytes'):
        if type(config[name]) is not int or config[name]<0:raise ValueError('nonnegative bytes required: '+name)
    for name in ('stop_step','lr_horizon','seq_length','checkpoint_interval','retention',
                 'timeout_seconds','operation_timeout_seconds','copy_timeout_ms','shm_base'):
        if type(config[name]) is not int or config[name]<1:raise ValueError('positive integer required: '+name)
    if config['stop_step']>config['lr_horizon']:raise ValueError('stop_step exceeds LR horizon')
    if config['seq_length']<8 or config['seq_length']%4:raise ValueError('sequence length must be >=8 and divisible by 4')
    if type(config['micro_batch']) is not int or config['micro_batch'] not in (1,2):raise ValueError('micro_batch must be 1 or 2')
    if config['jit_level'] not in ('O0','O1','O2'):raise ValueError('jit_level must be O0, O1 or O2')
    if config['operation_timeout_seconds']>config['timeout_seconds']:raise ValueError('operation timeout exceeds job timeout')
    if config['method']=='ours' and config['retention']!=3:
        raise ValueError('registered Qwen raw region has physical retention 3')
    for name in ('model','environment_manifest','strategy','region','hardware_lock','checkpoint_root'):
        if name in config:
            path=Path(config[name]);config[name]=str((path if path.is_absolute() else repo/path).resolve())
    if 'environment_manifest' not in config:raise ValueError('explicit environment_manifest required')
    model=Path(config['model']);raw=(model/'config.json').read_bytes()
    strategy=json.loads(Path(config['strategy']).read_text());validate(strategy)
    # Stop step, output location and backend do not change the optimizer trajectory.
    # Frozen model shard hashes are supplied by preflight before any training run.
    config['identity']=dict(model=model.name,model_config_sha256=hashlib.sha256(raw).hexdigest(),
        physical_npu_ids=config['npu_ids'],
        topology=dict(tp=4,dp=1,pp=1),seed=42,default_generator_seed=42,sequence_length=config['seq_length'],
        state_scope='full_state',params_dtype='float32',compute_dtype='bfloat16',dropout=0,
        lr_horizon=config['lr_horizon'],micro_batch=config['micro_batch'],jit_level=config['jit_level'],
        strategy_sha256=digest(strategy),
        data_fixture='checkpoint-recovery-fixed-text-v1',optimizer='mindformers-1.7-qwen3-template')
    return config
