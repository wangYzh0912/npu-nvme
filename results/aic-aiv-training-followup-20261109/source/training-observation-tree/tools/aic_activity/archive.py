"""Lossless durable copies with bounded storage and explicit provenance."""
import gzip,shutil,json,hashlib
from pathlib import Path

def copy_tree(source,destination):
    source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for src in sorted(source.rglob('*')):
        if not src.is_file():continue
        dst=destination/src.relative_to(source);dst.parent.mkdir(parents=True,exist_ok=True)
        compress=src.suffix=='.csv' and src.stat().st_size>100_000
        target=Path(str(dst)+'.gz') if compress else dst
        temp=target.with_name(target.name+'.copying')
        if compress:
            with src.open('rb') as f,gzip.open(temp,'wb') as g:shutil.copyfileobj(f,g)
        else:shutil.copy2(src,temp)
        temp.replace(target)
        if compress and dst.exists():dst.unlink()

def manifest(root):
    root=Path(root);rows=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name=='SHA256SUMS.json':continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(4<<20),b''):h.update(block)
        rows.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=h.hexdigest()))
    result=dict(total_bytes=sum(r['bytes'] for r in rows),files=rows)
    (root/'SHA256SUMS.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
