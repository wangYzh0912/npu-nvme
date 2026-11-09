"""Summarize rejected diverse-prompt runs without relaxing exact-token gates."""
from collections import Counter
import hashlib
import json


def summarize(root):
    rows=[]
    for path in sorted((root/'failure-evidence').glob('correctness*/*/windows.json')):
        windows=json.loads(path.read_text())
        requests=[w for w in windows if w['kind'].startswith('control')]
        if not requests:continue
        batch=len(requests[0]['output_token_ids'])
        tokens=[r['output_token_ids'] for r in requests]
        hashes=Counter(w['output_sha256'] for w in requests)
        per_request=[]
        for index in range(batch):
            sequences=[tuple(r[index]) for r in tokens]
            counts=Counter(sequences)
            mode,n=counts.most_common(1)[0]
            per_request.append(dict(index=index,distinct_outputs=len(counts),modal_count=n,
                modal_tokens=list(mode),
                modal_sha256=hashlib.sha256(json.dumps(list(mode)).encode()).hexdigest()))
        metrics=[r['request_metrics'] for r in requests if 'request_metrics' in r]
        rows.append(dict(run=path.parents[1].name,case=path.parent.name,control_calls=len(requests),
            distinct_batch_output_hashes=dict(hashes),per_request=per_request,
            metrics=metrics,exact_repeatability_pass=len(hashes)==1,
            interpretation='diagnostic only; modal agreement does not replace the rejected exact-output gate'))
    (root/'diverse-failed-run-analysis.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows
