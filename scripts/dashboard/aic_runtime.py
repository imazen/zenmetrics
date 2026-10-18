"""Hash-checked benchmark snapshots and separately scoped runtime supplements."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from catalog import ROOT

SNAPSHOT = 'benchmarks/iqa_chart_snapshot_2026-10-18'
SUPPLEMENTS = 'benchmarks/aic_refusals_2026-10-18'


def snapshot(root=ROOT):
    hosts=[]; observations=[]; paths=[SNAPSHOT+'/README.md']
    fields=('source','metric_name','width','height','threads','timing_phase','only_mode','device','impl',
            'impl_binary_sha','reference_sha256','distorted_sha256')
    for host in ('i265','mac'):
        base=root/SNAPSHOT/host
        manifest=json.loads((base/'snapshot.json').read_text())
        for name,want in manifest['files'].items():
            if hashlib.sha256((base/name).read_bytes()).hexdigest()!=want:
                raise ValueError('benchmark snapshot hash mismatch: '+host+'/'+name)
        rows=[json.loads(line) for line in (base/'observations.jsonl').read_text().splitlines()]
        if len(rows)!=manifest['attempt_rows']:
            raise ValueError('benchmark snapshot row count differs')
        sources={s['id']:s for s in json.loads((base/'sources.json').read_text())}
        qualified=[r for r in rows if r['qualified_timing']]
        if len(qualified)!=manifest['qualified_attempt_rows']:
            raise ValueError('benchmark snapshot qualified count differs')
        groups={}
        for row in qualified:
            if row['status']!='ok' or row['source'] not in sources:
                raise ValueError('qualified snapshot row lacks successful provenance')
            if row['timing_phase']=='warm' and (row['warm_rounds_clean']<10 or row['warm_rounds_flagged']):
                raise ValueError('qualified warm row lacks ten clean rounds')
            value=row['median_ns']
            if value is None or not math.isfinite(value) or value<=0:
                raise ValueError('qualified snapshot timing is unavailable')
            groups.setdefault(tuple(row[k] for k in fields),[]).append(row)
        for key,attempts in groups.items():
            record=dict(zip(fields,key));record.update(host=host,attempts=len(attempts),
                median_ms=statistics.median(r['median_ns'] for r in attempts)/1e6,
                source_report=sources[record['source']]['report'],capture_utc=manifest['captured_utc'])
            observations.append(record)
        hosts.append(dict(host=host,captured_utc=manifest['captured_utc'],attempt_rows=len(rows),
                          qualified_attempt_rows=len(qualified),metric_names=len({r['metric_name'] for r in qualified}),
                          qualified_groups=len(groups)))
        paths += [str((base/name).relative_to(root)) for name in ('snapshot.json','sources.json','observations.jsonl')]
    return dict(hosts=hosts,observations=observations,paths=paths)


def supplements():
    result=[]
    for stem in ('hdr-ahiq','fast-routes','fsim-luma','dvifm'):
        path=ROOT/SUPPLEMENTS/(stem+'-qualified-perf.tsv')
        groups={}
        for row in csv.DictReader(path.read_text().splitlines(),delimiter='\t'):
            cpu=float(row['task_clock_ms_per_call']);wall=float(row['wall_ms_per_call'])
            if not all(math.isfinite(x) and x>0 for x in (cpu,wall)):
                raise ValueError('unqualified supplement timing: '+str(path))
            groups.setdefault((row['metric'],row['shape'],int(row['threads'])),[]).append((cpu,wall,row['score']))
        for (metric,shape,threads),values in groups.items():
            if len({v[2] for v in values})!=1:
                raise ValueError('supplement repeated score differs')
            result.append(dict(metric=metric,shape=shape,threads=threads,n=len(values),host='r5900xt',
                               cpu_ms=statistics.median(v[0] for v in values),wall_ms=statistics.median(v[1] for v in values),
                               source=path.relative_to(ROOT).as_posix(),
                               qualification='Unmerged FSIM PR 92; not a master CLI route' if stem=='fsim-luma' else
                                   'Private released-build adapter; aggregates only' if stem=='dvifm' else 'Explicit qualified route; see metadata'))
    return result
