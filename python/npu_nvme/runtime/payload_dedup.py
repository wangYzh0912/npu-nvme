"""Deduplicate immutable artifacts only after all training processes exit."""
import os
from pathlib import Path
import uuid

from .training_catalog import digest_file,write_checked


def deduplicate_completed(run,cache):
    run=Path(run);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    for cached in cache.iterdir():
        if cached.is_file() and cached.stat().st_nlink==1:cached.unlink()
    rows=[]
    for path in sorted(run.rglob('*')):
        if path.is_symlink() or not path.is_file() or path.suffix not in ('.safetensors','.distcp','.pt'):
            continue
        size=path.stat().st_size
        if size<1<<20:continue
        checksum=digest_file(path);target=cache/checksum
        if target.exists():
            if target.stat().st_size!=size or digest_file(target)!=checksum:
                raise ValueError('immutable payload cache corrupt')
            if path.stat().st_ino!=target.stat().st_ino:
                temporary=path.with_name(path.name+'.dedup-'+uuid.uuid4().hex)
                os.link(target,temporary);temporary.replace(path)
        else:os.link(path,target)
        rows.append(dict(path=str(path),bytes=size,sha256=checksum,shared_with=str(target)))
    write_checked(run/'payload-dedup.json',dict(scope='after process exit; content unchanged',files=rows))
    return rows
