"""C1 configuration, source identity and result rules; standard library only."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
class CapabilityBlocked(ValueError):
    pass


METHODS = ('none', 'mindspore_native_save', 'ours', 'bytecheckpoint_host')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def source_identity(root=ROOT):
    root = Path(root)
    sources = {}
    for directory in ('python', 'src', 'include', 'experiments', 'scripts', 'tools', 'tests', 'config'):
        for p in sorted((root/directory).rglob('*')):
            if p.is_file() and p.suffix in ('.py', '.c', '.h', '.json', '.sh'):
                sources[str(p.relative_to(root))] = digest(p)
    for name in ('train.py', 'CMakeLists.txt'):
        if (root/name).is_file(): sources[name] = digest(root/name)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    patch = subprocess.check_output(['git', 'diff', 'HEAD', '--', 'python', 'src', 'include', 'experiments', 'scripts', 'tools', 'tests', 'config', 'train.py', 'CMakeLists.txt'], cwd=root)
    return dict(observed_commit=commit, source_digest=canonical(sources), sources=sources,
                dirty_diff_digest=hashlib.sha256(patch).hexdigest())


FIELDS = {
    'workload': {'model_id','model_revision','task','dtype','training_mode','batch','seq_len','seeds','dataset_revision','dropout','learning_rate'},
    'checkpoint': {'methods','state_scope','capture','transport','admission','wait_policy','interval','retention'},
    'runtime': {'admission_ms','wait_ms','drain_ms','close_ms','phase_ms','max_requests','max_batch_items','max_batch_bytes','max_metadata_bytes','host_budget_bytes','hbm_budget_bytes','chunk_bytes','dma_depth'},
    'storage': {'pci','namespace','offset','length','media_version','owner','mode','fs_root','authorization_file','slot_size_gb','shm_id'},
    'measurement': {'mode','timing_schema','integrity','warmup','formal_steps','continue_steps','restore_repetitions','cache_policy','run_order','loss_rtol','loss_atol'},
    'output': {'root'},
    'identity': {'expected_commit','allow_dirty'},
}


def positive(value, field):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{field} must be positive and finite')


def load_config(path):
    c = json.loads(Path(path).read_text())
    if not isinstance(c, dict) or set(c) != set(FIELDS)|{'schema_version'} or type(c['schema_version']) is not int or c['schema_version'] != 1:
        raise ValueError('unknown/missing configuration domains or version')
    for domain, fields in FIELDS.items():
        if not isinstance(c[domain], dict) or set(c[domain]) != fields:
            raise ValueError(f'unknown/missing fields in {domain}')
    w, k, r, s, m = (c[x] for x in ('workload','checkpoint','runtime','storage','measurement'))
    if w['model_id'] != 'gpt2' or w['task'] != 'causal_lm' or w['training_mode'] != 'single_rank' or w['batch'] != 1:
        raise ValueError('C1 workload is single-rank GPT-2 causal LM, batch 1')
    if w['dtype'] != 'fp32' or w['dropout'] != 0 or w['learning_rate'] != 1e-5 or w['dataset_revision'] != 'fixed-arithmetic-tokens-v1':
        raise ValueError('unsupported C1 dtype/dropout/LR/dataset')
    if not c['identity']['allow_dirty'] and not str(w['model_revision']).startswith('sha256:'):
        raise ValueError('formal workload must lock the resolved model configuration hash')
    if not isinstance(w['model_revision'], str) or not w['model_revision']:
        raise ValueError('model revision required')
    if type(w['seq_len']) is not int or w['seq_len'] < 2: raise ValueError('seq_len is input-record tokens, at least 2')
    if not isinstance(w['seeds'],list) or not w['seeds'] or any(type(x) is not int or x < 0 for x in w['seeds']) or len(set(w['seeds'])) != len(w['seeds']):
        raise ValueError('distinct integer seeds required')
    if not isinstance(k['methods'],list) or not k['methods'] or any(x not in METHODS for x in k['methods']) or len(set(k['methods'])) != len(k['methods']):
        raise ValueError('unknown/duplicate methods')
    if (k['state_scope'],k['capture'],k['transport'],k['admission'],k['wait_policy'],k['retention']) != ('full_state','frozen','legacy_sync','block','source_safe',2):
        raise CapabilityBlocked('C1 requires full_state/frozen/legacy_sync/block/source_safe/retention=2')
    for key,value in r.items(): positive(value, 'runtime.'+key)
    if any(type(v) is not int for v in r.values()): raise ValueError('runtime budgets use integer bytes/ms/counts')
    if r['max_requests'] != 1 or r['chunk_bytes'] > 1048576 or r['chunk_bytes'] % 4096 or r['dma_depth'] > 16:
        raise ValueError('D1 bounds: one pending, aligned chunk <= 1 MiB, depth <=16')
    if r['max_batch_items'] != 65536 or r['max_batch_bytes'] != 68719476736 or r['max_metadata_bytes'] != 1048576:
        raise ValueError('runtime exceeds frozen safety limits')
    if r['max_batch_bytes'] < r['chunk_bytes']: raise ValueError('batch budget smaller than chunk')
    if (s['pci'],s['namespace'],s['offset'],s['length'],s['media_version'],s['owner'],s['mode'],s['slot_size_gb']) != ('0000:83:00.0',1,0,'device_capacity_bytes','V2','single','read_write',10):
        raise ValueError('storage differs from authorized D1 namespace/geometry')
    if type(s['shm_id']) is not int or s['shm_id'] <= 0: raise ValueError('invalid shm_id')
    for key in ('interval',): positive(k[key],key)
    for key in ('formal_steps','continue_steps','restore_repetitions'): positive(m[key],key)
    if any(type(x) is not int for x in (k['interval'],m['formal_steps'],m['continue_steps'],m['restore_repetitions'],m['warmup'])) or m['warmup'] < 1:
        raise ValueError('step/repetition counts must be integers, warmup >=1')
    if m['formal_steps'] % k['interval'] or m['continue_steps'] < 3: raise ValueError('final step must checkpoint; continuation >=3')
    if (m['mode'],m['timing_schema'],m['integrity'],m['cache_policy'],m['run_order']) != ('correctness_and_timing',1,'mandatory','warm','rotating_serial'):
        raise ValueError('unsupported measurement contract')
    if m['loss_rtol'] != 1e-5 or m['loss_atol'] != 1e-6: raise ValueError('frozen H01 tolerances required')
    if type(c['identity']['allow_dirty']) is not bool: raise ValueError('allow_dirty must be boolean')
    expected = c['identity']['expected_commit']
    if not isinstance(expected,str) or len(expected)!=40 or any(x not in '0123456789abcdef' for x in expected): raise ValueError('expected_commit must be full commit SHA')
    for key in ('fs_root','authorization_file'):
        if not Path(s[key]).is_absolute(): raise ValueError(f'{key} must be absolute')
    if not Path(c['output']['root']).is_absolute(): raise ValueError('output root must be absolute')
    return c


def check_identity(config, identity):
    if config['identity']['expected_commit'] != identity['observed_commit']:
        raise ValueError('expected_commit differs from actual HEAD')
    if not config['identity']['allow_dirty']:
        committed = subprocess.check_output(['git','ls-files','--others','--exclude-standard','--','python','experiments','scripts','tools','tests','config','train.py'],cwd=ROOT,text=True).strip()
        if identity['dirty_diff_digest'] != hashlib.sha256(b'').hexdigest() or committed:
            raise ValueError('strict execution requires versioned source/config')


def loss_matches(actual, expected, steps, rtol=1e-5, atol=1e-6):
    return (len(actual)==len(expected)==steps and steps>=3 and
            all(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and
                abs(a-b)<=atol+rtol*abs(b) for a,b in zip(actual,expected)))


def outcome(row, source_rc=0, restore_rc=0):
    if source_rc or restore_rc: return 1
    if row.get('status') != 'trend_measured': return 1
    restored = row.get('restore', {})
    if row.get('adapter')=='none': return 0 if restored.get('status')=='not_applicable' else 1
    return 0 if restored.get('status')=='pass' and restored.get('byte_exact') is True and restored.get('verification_performed') is True else 1


def comparison(method):
    group = 'training-only' if method=='none' else 'raw-83' if method=='ours' else 'filesystem-84'
    return dict(comparable=method!='none', comparison_group=group,
                reasons=['Different physical devices: do not rank raw-83 against filesystem-84'] if method=='ours' else
                ['No checkpoint or recovery'] if method=='none' else [])


def validate_restore_report(restored, source, steps):
    if outcome(dict(source,restore=restored)):
        raise ValueError('restore verdict lacks byte-exact verification')
    if restored.get('mandatory_integrity') is not True or restored.get('controls_verified') is not True:
        raise ValueError('mandatory integrity/control readback missing')
    if not loss_matches(restored.get('restored_losses',[]),restored.get('source_oracle_losses',[]),steps):
        raise ValueError('restore continuation oracle incomplete or outside tolerance')
    if type(source.get('source_pid')) is not int or restored.get('source_pid') != source['source_pid'] or type(restored.get('restore_pid')) is not int or restored['restore_pid'] <= 0 or restored.get('restore_pid') == source['source_pid']:
        raise ValueError('restore is not an identified fresh process')
    records=[x for x in source.get('checkpoints',[]) if x.get('generation')==restored.get('generation')]
    if len(records)!=1 or records[0].get('step')!=restored.get('checkpoint_step'):
        raise ValueError('restored generation/step differs from source')
    expected=records[0].get('sha256')
    if not isinstance(expected,str) or len(expected)!=64 or restored.get('expected_state_sha256') != expected or restored.get('applied_state_sha256') != expected:
        raise ValueError('restored state digest differs from selected checkpoint')
