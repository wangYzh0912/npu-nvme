"""D1 manifest validation without framework allocations or native calls."""
import hashlib
import json
import math
import re
from collections.abc import Mapping

CHUNK_BYTES = 1 << 20
MAX_CHUNK_BYTES = 16 << 20
CONTROL_BYTES = 1 << 20
MANIFEST_BYTES = 16 << 20
MAX_FIELDS = 65536
DTYPES = {'bool': 1, 'uint8': 1, 'int8': 1, 'int16': 2, 'uint16': 2,
          'int32': 4, 'uint32': 4, 'int64': 8, 'uint64': 8,
          'float16': 2, 'float32': 4, 'float64': 8, 'complex64': 8, 'complex128': 16}
REQUIRED_CONTROLS = frozenset(('global_step', 'loss_scale', 'python_rng', 'numpy_rng',
                              'mindspore_seed', 'mindspore_rng', 'data_cursor'))
APPLICABILITY = frozenset(('model', 'optimizer', 'rng', 'data_cursor', 'scheduler', 'loss_scale'))


def integer(value, name, minimum=0, maximum=(1 << 64) - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'invalid {name}')
    return value


def digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('invalid SHA256 digest')
    return value


def align(size):
    return (size + 4095) // 4096 * 4096


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def manifest_digest(record):
    return hashlib.sha256(canonical({k: v for k, v in record.items() if k != 'manifest_sha256'})).hexdigest()


def spec_validate(spec):
    if not isinstance(spec, dict) or set(spec) != {'identity', 'parameters', 'control_names', 'applicability'}:
        raise ValueError('expected_spec requires identity, parameters, control_names and applicability')
    if not isinstance(spec['identity'], dict) or not spec['identity'] or len(canonical(spec['identity'])) > 65536:
        raise ValueError('invalid training identity')
    if not isinstance(spec['applicability'], dict):
        raise ValueError('applicability must be an explicit mapping')
    if set(spec['applicability']) != APPLICABILITY:
        raise ValueError('all state applicability decisions must be explicit')
    for name, value in spec['applicability'].items():
        if value != 'required' and not (isinstance(value, str) and value.startswith('not_applicable:') and len(value) > 15):
            raise ValueError('applicability requires a reason for omitted state')
    if any(spec['applicability'][k] != 'required' for k in APPLICABILITY - {'scheduler'}):
        raise ValueError('D1 FULL requires model, optimizer, RNG, cursor and loss scale')
    controls = spec['control_names']
    expected = REQUIRED_CONTROLS | ({'scheduler'} if spec['applicability']['scheduler'] == 'required' else set())
    if not isinstance(controls, list) or controls != sorted(expected):
        raise ValueError('control applicability set differs')
    params = spec['parameters']
    if not isinstance(params, dict) or not params or len(params) > MAX_FIELDS:
        raise ValueError('invalid parameter set')
    if not all(any(name.startswith(prefix) for name in params if isinstance(name,str))
               for prefix in ('model/', 'optimizer/')):
        raise ValueError('model and optimizer tensor namespaces are required')
    for name, info in params.items():
        if not isinstance(name, str) or not name.startswith(('model/', 'optimizer/')) or not name.split('/',1)[1] or len(name.encode()) > 1024:
            raise ValueError('invalid parameter name')
        tensor_validate(info)
    return spec


def tensor_validate(info):
    if not isinstance(info, Mapping) or info.get('dtype') not in DTYPES:
        raise ValueError('unsupported tensor dtype')
    shape = info.get('shape')
    if not isinstance(shape, (list, tuple)) or len(shape) > 32:
        raise ValueError('invalid tensor shape')
    for dim in shape:
        integer(dim, 'dimension', 1, 1 << 40)
    size = integer(info.get('size'), 'tensor bytes', 1, 64 << 30)
    if math.prod(shape) * DTYPES[info['dtype']] != size:
        raise ValueError('shape/dtype and byte size differ')
    return size


def validate_record(record, layout, expected_spec=None):
    if not isinstance(record, dict) or len(canonical(record)) > MANIFEST_BYTES:
        raise ValueError('manifest exceeds decoded budget')
    if (record.get('strict_contract') != 'D1' or record.get('type') != 'TRAINING_STATE_FULL'
            or record.get('schema_version') != 1 or record.get('rank_id') != 0 or record.get('world_size') != 1):
        raise ValueError('strict single-rank FULL record required; legacy record rejected')
    integer(record.get('generation'), 'generation', 1)
    integer(record.get('rank_id'), 'rank', 0, 0)
    integer(record.get('world_size'), 'world size', 1, 1)
    integer(record.get('state_step'), 'step')
    slot = integer(record.get('slot'), 'slot', 0, layout.full_slot_count - 1)
    chunk_size = integer(record.get('chunk_size'), 'chunk size', 4096, MAX_CHUNK_BYTES)
    if chunk_size % 4096:
        raise ValueError('chunk size is unaligned')
    if not isinstance(record.get('request_id'), str) or not record['request_id'] or not isinstance(record.get('writer_epoch'), str) or not record['writer_epoch']:
        raise ValueError('request identity missing')
    spec = spec_validate(record.get('spec'))
    if expected_spec is not None and canonical(spec_validate(expected_spec)) != canonical(spec):
        raise ValueError('target specification does not match checkpoint identity/state')
    params = record.get('params')
    expected_names = set(spec['parameters']) | {'control/' + n for n in spec['control_names']}
    if not isinstance(params, dict) or set(params) != expected_names:
        raise ValueError('tensor/control manifest set differs')
    begin = layout.full_base + slot * layout.full_slot_bytes
    end = begin + layout.full_slot_bytes
    extents = []
    control_total = 0
    total = count = 0
    for name, info in params.items():
        size = tensor_validate(info)
        if name.startswith('control/'):
            control_total += size
            if info['dtype'] != 'uint8' or info['shape'] != [size] or info.get('codec') != 'json-tagged-v1':
                raise ValueError('invalid control representation')
        elif (tuple(info['shape']) != tuple(spec['parameters'][name]['shape']) or
              any(info[k] != spec['parameters'][name][k] for k in ('dtype', 'size'))):
            raise ValueError('tensor differs from expected shape/dtype/size')
        offset = integer(info.get('offset'), 'offset', begin, end)
        if offset % 4096 or align(size) > end - offset:
            raise ValueError('tensor extent escapes reserved FULL slot')
        extents.append((offset, offset + align(size)))
        digest(info.get('sha256'))
        chunks = info.get('chunks')
        if not isinstance(chunks, list) or len(chunks) != (size + chunk_size - 1) // chunk_size:
            raise ValueError('per-chunk integrity descriptors required')
        cursor = 0
        for chunk in chunks:
            take = min(chunk_size, size - cursor)
            if not isinstance(chunk, dict) or chunk.get('offset') != cursor or chunk.get('size') != take:
                raise ValueError('chunk offset or byte count differs')
            digest(chunk.get('sha256'))
            cursor += take
        count += len(chunks)
        total += size
    if count > MAX_FIELDS or control_total > CONTROL_BYTES:
        raise ValueError('chunk/control budget exceeded')
    digest(record.get('data_sha256'))
    extents.sort()
    if any(b[0] < a[1] for a, b in zip(extents, extents[1:])):
        raise ValueError('overlapping tensor extents')
    if record.get('manifest_sha256') != manifest_digest(record):
        raise ValueError('manifest digest mismatch')
    return total, count
