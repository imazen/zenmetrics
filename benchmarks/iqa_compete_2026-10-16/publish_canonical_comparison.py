#!/usr/bin/env python3
"""Project retained comparative cohorts into small, source-linked README data.

Raw cohorts remain immutable. This writes one compact TSV per cohort and an
exact-name projection. It never treats an unassigned bound as a scalar pass.
"""
import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import statistics

from compare_canonical import sha, summarize


def read_cohort(root, dispositions):
    plan_path=root/'plan.json'
    plan=json.loads(plan_path.read_text())
    digest=sha(plan_path)
    cases={c['id']: c for c in plan['cases']}
    rows=[]
    for path in sorted(root.glob('*/summary.json')):
        row=json.loads(path.read_text())
        case=cases[row['case']]
        if row['plan_sha256']!=digest or row['rust_source']!=plan['rust_source']:
            raise ValueError('cohort plan/source mismatch: '+str(path))
        for field in ('metric','width','height','threads','absolute_bound','matched_scope','scope'):
            if row[field]!=case[field]:
                raise ValueError('case contract mismatch: '+field)
        blocks=[json.loads(p.read_text()) for p in sorted(path.parent.glob('block-*/result.json'))]
        expected=summarize(blocks,case['matched_scope'])
        if row['ratio']!=expected:
            raise ValueError('stored ratio differs from acquired blocks')
        record={k:row[k] for k in ('metric','case','width','height','threads','passed','absolute_bound',
                                   'max_abs_delta','ratio','matched_scope','rust_source','plan_sha256')}
        record['canonical']=row['canonical']
        measured=[b for b in blocks if 'rust_ns' in b and 'canonical_ns' in b]
        record['rust_ms']=statistics.median(b['rust_ns'] for b in measured)/1e6 if measured else None
        record['canonical_ms']=statistics.median(b['canonical_ns'] for b in measured)/1e6 if measured else None
        record['expected_cases']=sum(c['metric']==row['metric'] for c in cases.values())
        record['excluded_contract']=dispositions.get(digest+':'+row['metric'],'')
        record['summary_sha256']=sha(path)
        record['blocks']=len(blocks)
        rows.append(record)
    return rows


def project(rows):
    grouped=defaultdict(list)
    for row in rows:
        if not row['excluded_contract']:
            grouped[row['metric']].append(row)
    result={}
    for name, entries in grouped.items():
        expected=sum(next(r['expected_cases'] for r in entries if r['plan_sha256']==digest)
                     for digest in {r['plan_sha256'] for r in entries})
        failed=sum(r['passed'] is False for r in entries)
        passed=sum(r['passed'] is True for r in entries)
        deltas=[r['max_abs_delta'] for r in entries if r['max_abs_delta'] is not None]
        if failed:
            check=f'{failed} findings'
        elif passed==len(entries):
            check=f'{passed}/{expected} pass'
        else:
            check=f'Δ {max(deltas):.2g}' if deltas else 'unmeasured'
        # The row has room for one representative ratio. Keep its actual shape,
        # worker count and sample cohort in the expanded evidence model.
        eligible=[r for r in entries if r['ratio'] is not None]
        key=None
        if eligible and not failed:
            def preference(r):
                size=(r['width'],r['height'])
                return (r['threads']!=1, size!=(1024,1024), size!=(512,512), -r['width']*r['height'],r['threads'])
            chosen=min(eligible,key=preference)
            key=(chosen['width'],chosen['height'],chosen['threads'],chosen['rust_source'],chosen['plan_sha256'])
            selected=[r for r in eligible if (r['width'],r['height'],r['threads'],r['rust_source'],r['plan_sha256'])==key]
        else:
            selected=[]
        result[name]=dict(canonical=entries[0]['canonical'],check=check,
            ratio=statistics.median(r['ratio'] for r in selected) if selected else None,
            ratio_width=key[0] if key else None,ratio_height=key[1] if key else None,
            ratio_threads=key[2] if key else None,ratio_cases=len(selected),
            observed_cases=len(entries),expected_cases=expected,passed_cases=passed,failed_cases=failed,
            max_abs_delta=max(deltas) if deltas else None,
            rust_sources=sorted({r['rust_source'] for r in entries}),
            plan_sha256=sorted({r['plan_sha256'] for r in entries}),qualified=False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('cohorts',nargs='+',type=Path)
    p.add_argument('--dispositions',type=Path,help='Explicit plan-sha:metric -> contract-mismatch reason')
    a=p.parse_args()
    a.output.mkdir(exist_ok=True)
    dispositions=json.loads(a.dispositions.read_text()) if a.dispositions else {}
    combined=[];sources={}
    for root in a.cohorts:
        rows=read_cohort(root,dispositions)
        if not rows:
            raise ValueError('no acquired case summaries: '+str(root))
        stream=io.StringIO(newline='')
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
        data=stream.getvalue().encode()
        if len(data)>30_000:
            # Split by metric, retaining each original case and its artifact hash.
            for metric in sorted({r['metric'] for r in rows}):
                part=io.StringIO(newline='')
                w=csv.DictWriter(part,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
                w.writeheader();w.writerows(r for r in rows if r['metric']==metric)
                payload=part.getvalue().encode()
                if len(payload)>30_000:raise ValueError('metric shard exceeds generated-data cap')
                (a.output/(root.name+'-'+metric+'.tsv')).write_bytes(payload)
        else:
            (a.output/(root.name+'.tsv')).write_bytes(data)
        sources[root.name]=dict(plan_sha256=sha(root/'plan.json'),cases=len(rows),
                               expected_cases=len(json.loads((root/'plan.json').read_text())['cases']))
        combined.extend(rows)
    identities=[(r['plan_sha256'],r['case']) for r in combined]
    if len(identities)!=len(set(identities)):
        raise ValueError('duplicate acquired cases across selected cohorts')
    value=project(combined)
    text=json.dumps(value,indent=2)+'\n'
    if len(text.encode())>30_000:
        # Compact projection; per-case detail remains in the indexed TSVs.
        text=json.dumps(value,separators=(',',':'))+'\n'
    if len(text.encode())>30_000:raise ValueError('README projection exceeds generated-data cap')
    (a.output/'results.json').write_text(text)
    (a.output/'cohorts.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps(dict(metrics=len(value),cases=len(combined),
                         excluded=sum(bool(r['excluded_contract']) for r in combined))))


if __name__=='__main__':
    main()
