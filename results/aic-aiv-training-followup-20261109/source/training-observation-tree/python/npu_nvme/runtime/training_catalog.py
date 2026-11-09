"""Durable checkpoint metadata shared by the four-method training entry.

Native and ByteCheckpoint retain their payload formats. Ours records its actual
media receipt. A catalog generation is scoped to a training lineage and is not
the optimizer step or the raw region's generation number.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest_file(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):hasher.update(block)
    return hasher.hexdigest()


def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_checked(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=canonical(dict(version=1,payload=value,sha256=hashlib.sha256(canonical(value)).hexdigest()))
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path);fsync_directory(path.parent)


def read_checked(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate metadata key')
            result[key]=value
        return result
    row=json.loads(Path(path).read_text(),object_pairs_hook=unique)
    if set(row)!={'version','payload','sha256'} or row['version']!=1:
        raise ValueError('invalid checkpoint metadata envelope')
    if hashlib.sha256(canonical(row['payload'])).hexdigest()!=row['sha256']:
        raise ValueError('checkpoint metadata digest differs')
    return row['payload']


@contextmanager
def catalog_lock(root,exclusive=True):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.catalog.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def initialize(root, *, identity, method):
    root=Path(root)
    with catalog_lock(root):
        header=root/'lineage.json'
        if header.exists():
            value=read_checked(header)
            if value['identity']!=identity or value['method']!=method:
                raise ValueError('checkpoint lineage identity/method differs')
        else:
            value=dict(lineage_id=uuid.uuid4().hex,identity=identity,method=method)
            write_checked(header,value);write_checked(root/'counter.json',dict(generation=0))
    return value


def reserve(root, *, run_id, step):
    root=Path(root)
    if type(step) is not int or step<=0:raise ValueError('invalid checkpoint step')
    with catalog_lock(root):
        header=read_checked(root/'lineage.json')
        generation=read_checked(root/'counter.json')['generation']+1
        write_checked(root/'counter.json',dict(generation=generation))
        path=root/f'generation-{generation:012d}';path.mkdir(exist_ok=False)
        row=dict(**header,generation=generation,run_id=run_id,step=step,state_scope='full_state')
        write_checked(path/'reservation.json',row)
    return path,row


def payload_files(directory):
    directory=Path(directory)
    result=[]
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():raise ValueError('checkpoint payload cannot contain symlinks')
        if path.is_file():
            result.append(dict(path=str(path.relative_to(directory)),bytes=path.stat().st_size,sha256=digest_file(path)))
    return result


def verify_files(directory,files):
    directory=Path(directory).resolve()
    for row in files:
        path=(directory/row['path']).resolve()
        if directory not in path.parents or not path.is_file() or path.stat().st_size!=row['bytes']:
            raise ValueError('missing, escaping or truncated checkpoint payload')
        if digest_file(path)!=row['sha256']:raise ValueError('checkpoint payload digest differs')


def publish(path, *, world=4):
    path=Path(path)
    with catalog_lock(path.parent):
        return _publish_locked(path,world)


def _publish_locked(path,world,*,recovered=False):
    reservation=read_checked(path/'reservation.json');rows=[]
    if (path/'checkpoint.json').exists():raise ValueError('generation already published')
    for rank in range(world):
        entry=read_checked(path/f'rank_{rank}'/'complete.json')
        _rank_identity(entry,reservation,rank)
        rows.append(entry)
    if reservation['method']=='ours':
        receipts=[row['backend']['receipt'] for row in rows]
        if any(row!=receipts[0] for row in receipts) or type(receipts[0].get('generation')) is not int:
            raise ValueError('rank media receipts differ')
    value=dict(reservation,ranks=rows,status='committed',world_size=world)
    if recovered:value['recovered']=True
    validate_publication(value,path)
    write_checked(path/'checkpoint.json',value)
    return value


def _rank_identity(entry,reservation,rank):
    if (entry['rank'],entry['generation'],entry['step'],entry['method'],entry['identity'])!=(
            rank,reservation['generation'],reservation['step'],reservation['method'],reservation['identity']):
        raise ValueError('rank completion identity differs')
    if not entry.get('state') or not entry.get('controls'):
        raise ValueError('FULL rank completion requires state and controls')


def reconcile(root, inventory):
    """Recover only journaled publications proven by a retained raw commit.

    Called under the hardware lease, before new writers or readers start. An
    absent receipt means unknown or retired, not proof that a commit aborted.
    Existing corrupt files remain evidence and are never repaired in place.
    """
    root=Path(root);report=dict(recovered=[],rejected=[])
    if not root.exists():return report
    by_request={row['receipt']['request_id']:row for row in inventory}
    with catalog_lock(root):
        for path in sorted(root.glob('generation-*')):
            if not path.is_dir():continue
            try:
                if (path/'checkpoint.json').exists():
                    validate_publication(read_checked(path/'checkpoint.json'),path)
                    continue
                reservation=read_checked(path/'reservation.json')
                if reservation['method']!='ours':continue
                header=read_checked(root/'lineage.json')
                if any(reservation[k]!=header[k] for k in ('lineage_id','identity','method')):
                    raise ValueError('reservation lineage differs')
                prepared=[read_checked(path/f'rank_{rank}/prepared.json') for rank in range(4)]
                for rank,row in enumerate(prepared):
                    _rank_identity(row,reservation,rank)
                    if row['preparation_version']!=1:raise ValueError('unknown preparation version')
                request=prepared[0]['raw_request_id']
                if any(row['raw_request_id']!=request for row in prepared):raise ValueError('rank request identities differ')
                raw=by_request.get(request)
                if raw is None:raise ValueError('raw commit unknown or no longer retained')
                receipt=raw['receipt']
                controls=[dict(rank=r,step=row['step'],controls=row['controls']) for r,row in enumerate(prepared)]
                if (receipt['identity']!=reservation['identity'] or raw['step']!=reservation['step'] or
                        raw['topology']!={'world_size':4} or
                        raw['controls_sha256']!=hashlib.sha256(canonical(controls)).hexdigest()):
                    raise ValueError('raw commit differs from prepared publication')
                completed=[]
                for rank,row in enumerate(prepared):
                    complete=path/f'rank_{rank}/complete.json'
                    if complete.exists():
                        entry=read_checked(complete)
                        _rank_identity(entry,reservation,rank)
                        if (entry['backend']['receipt']!=receipt or entry['state']!=row['state'] or
                                entry['controls']!=row['controls']):raise ValueError('completion conflicts with prepared commit')
                    else:
                        entry={k:v for k,v in row.items() if k not in ('preparation_version','raw_request_id')}
                        entry.update(backend=dict(row['backend'],receipt=receipt),recovered=True,save_seconds=None,
                            files=[dict(path='prepared.json',bytes=(complete.parent/'prepared.json').stat().st_size,
                                        sha256=digest_file(complete.parent/'prepared.json'))])
                    completed.append((complete,entry))
                # Validate the entire set before creating any missing completion.
                validate_publication(dict(reservation,ranks=[entry for _,entry in completed],
                    status='committed',world_size=4),path)
                for complete,entry in completed:
                    verify_files(complete.parent,entry['files'])
                for complete,entry in completed:
                    if not complete.exists():write_checked(complete,entry)
                _publish_locked(path,4,recovered=True)
                report['recovered'].append(reservation['generation'])
            except (ValueError,KeyError,TypeError,OSError) as error:
                report['rejected'].append(dict(path=str(path),reason=str(error)))
    return report


def validate_publication(value,path):
    if not isinstance(value,dict):raise ValueError('invalid checkpoint publication')
    generation=value.get('generation');world=value.get('world_size');rows=value.get('ranks')
    if (type(generation) is not int or generation<1 or
            path.name!=f'generation-{generation:012d}' or value.get('status')!='committed' or
            type(world) is not int or world not in (2,4) or not isinstance(rows,list) or len(rows)!=world):
        raise ValueError('invalid checkpoint publication')
    for rank,row in enumerate(rows):
        if (not isinstance(row,dict) or row.get('rank')!=rank or row.get('generation')!=generation or
                row.get('step')!=value.get('step') or row.get('identity')!=value.get('identity') or
                row.get('method')!=value.get('method') or not row.get('state') or not row.get('controls')):
            raise ValueError('invalid checkpoint rank publication')
    return value


def committed(root, *, rejected=None):
    result=[]
    for path in Path(root).glob('generation-*/checkpoint.json'):
        try:
            value=read_checked(path)
            validate_publication(value,path.parent)
        except (ValueError,KeyError,TypeError,OSError) as error:
            if rejected is None:raise
            rejected.append(dict(path=str(path.parent),reason=str(error)))
            continue
        result.append((value['generation'],path.parent,value))
    return sorted(result)


@contextmanager
def selected(root, selector='latest', *, validate=None, rejected=None):
    """Pin a verified generation; latest may fall back, explicit never does.

    ``validate(path, metadata)`` checks payloads and, for raw storage, that the
    media generation is still retained. Callers can retain rejection evidence.
    """
    root=Path(root);stream=None
    rejected=[] if rejected is None else rejected
    try:
        with catalog_lock(root,exclusive=False):
            if selector=='latest':rows=committed(root,rejected=rejected)
            else:
                if isinstance(selector,bool):raise ValueError('invalid generation selector')
                generation=int(selector)
                if generation<1:raise ValueError('invalid generation selector')
                path=root/f'generation-{generation:012d}'
                value=validate_publication(read_checked(path/'checkpoint.json'),path)
                rows=[(generation,path,value)]
            if not rows:raise ValueError('no matching committed checkpoint')
            for _,path,value in reversed(rows):
                stream=(path/'.reader.lock').open('a+b');fcntl.flock(stream,fcntl.LOCK_SH)
                try:
                    if validate is not None:validate(path,value)
                except (ValueError,OSError) as error:
                    stream.close();stream=None
                    if selector!='latest':raise
                    rejected.append(dict(path=str(path),reason=str(error)))
                    continue
                break
            if stream is None:raise ValueError('no valid committed checkpoint')
        yield path,value
    finally:
        if stream:stream.close()


def prune(root, retention):
    if type(retention) is not int or retention<1:raise ValueError('invalid retention')
    retired=[]
    with catalog_lock(root):
        rejected=[]
        rows=committed(root,rejected=rejected)
        if rejected:
            write_checked(Path(root)/'prune-rejections.json',rejected)
        for generation,path,_ in rows[:-retention]:
            with (path/'.reader.lock').open('a+b') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                retired_path=path.with_name('.retired-'+path.name)
                path.rename(retired_path);fsync_directory(path.parent)
                shutil.rmtree(retired_path)
                retired.append(generation)
    return retired
