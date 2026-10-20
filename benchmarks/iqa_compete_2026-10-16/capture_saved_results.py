#!/usr/bin/env python3
"""Read-only capture of finalized saved attempts; raw captures stay outside public git."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from export_chart_snapshot import export


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('spec',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--host',required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=True)
    if (a.output/'capture-receipt.json').exists():raise ValueError('completed capture is immutable')
    start=now();receipts=[]
    for name,root in json.loads(a.spec.read_text()).items():
        root=Path(root)
        if not (root/'report.json').exists():raise ValueError('missing saved campaign '+name)
        dest=a.output/name
        if not (dest/'snapshot.json').exists():
            export(root,dest,a.host,include_sealed=True,record_rejected_overlays=True)
        identity=json.loads((root/'identity.json').read_bytes())
        manifest=identity['frozen']['manifest']
        if not isinstance(manifest,dict):raise ValueError('embedded frozen manifest required')
        receipts.append(dict(campaign=name,capture=json.loads((dest/'snapshot.json').read_text()),
            manifest_sha256=identity['frozen']['manifest_sha256'],binary_sha256=identity['frozen']['binary_sha256'],
            inputs=identity['frozen']['input_sha256'],
            source_commit=manifest.get('metadata',{}).get('source_commit'),
            build_receipt_sha256=manifest.get('metadata',{}).get('build_receipt_sha256'),
            policy={k:identity['policy'].get(k) for k in ['max_attempts','jobs','memory','timeout_seconds','wrapper_sha256','wrapper_backend_sha256']},
            metrics=sorted({c['metric_name'] for c in manifest['cells']}),
            manifest_cells=len(manifest['cells'])))
    (a.output/'capture-receipt.json').write_text(json.dumps(dict(host=a.host,capture_started_utc=start,capture_finished_utc=now(),campaigns=receipts),indent=2)+'\n')
    print(json.dumps(dict(host=a.host,campaigns=len(receipts),capture_finished_utc=now())))

if __name__=='__main__':main()
