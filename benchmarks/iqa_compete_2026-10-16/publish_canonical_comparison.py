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
import re
import statistics

from compare_canonical import compare, sha, summarize
from reference_publication import register_outputs


def native_engine(plan):
    if plan.get('rust_engine'):
        return plan['rust_engine']
    with Path(plan['binary']).open('rb') as stream:
        worker=stream.read(8192)
    if not worker.startswith(b'#!'):
        return None
    text=worker.decode()
    engines=[]
    for path in plan['bindings']:
        if path.endswith('settings.json') and path in text:
            settings=json.loads(Path(path).read_text())
            engine=settings.get('environment',{}).get('ZENMETRICS_NN_ENGINE')
            if engine:engines.append(engine)
    if len(set(engines))>1:
        raise ValueError('ambiguous executed worker engine')
    return engines[0] if engines else None


def validate_numerics(blocks, case, row):
    """Recompute scalar decisions from the retained arms before publication."""
    for block in blocks:
        arms = block.get('arms', {})
        if len(arms) != 2:
            continue
        canonical = arms['canonical']
        scores = canonical.get('extra', {}).get('sample_scores', [canonical['score']])
        checks = [compare(arms['rust']['score'], score, case['absolute_bound'],
                          case.get('relative_bound', 0)) for score in scores]
        passed = False if any(ok is False for _, ok in checks) else None if case['absolute_bound'] is None else True
        if block['delta'] != max(delta for delta, _ in checks) or block['passed'] is not passed:
            raise ValueError('stored numerical decision differs from acquired arms')
        for role in ('rust', 'canonical'):
            if block[role+'_ns'] != statistics.median(arms[role]['times_ns']):
                raise ValueError('stored timing differs from acquired samples')
    passed = (False if any(block.get('errors') or block['passed'] is False for block in blocks)
              else None if any(block['passed'] is None for block in blocks) else True)
    delta = max((block['delta'] for block in blocks if 'delta' in block), default=None)
    if row['passed'] is not passed or row['max_abs_delta'] != delta:
        raise ValueError('summary numerical decision differs from acquired blocks')


def read_cohort(root, dispositions):
    plan_path=root/'plan.json'
    plan=json.loads(plan_path.read_text())
    digest=sha(plan_path)
    engine=native_engine(plan)
    cases={c['id']: c for c in plan['cases']}
    def disposition(case):
        return dispositions.get(digest+':'+case['metric']+':'+case['id'],
                                dispositions.get(digest+':'+case['metric'],''))
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
        validate_numerics(blocks, case, row)
        expected=summarize(blocks,case['matched_scope'])
        if row['ratio']!=expected:
            raise ValueError('stored ratio differs from acquired blocks')
        record={k:row[k] for k in ('metric','case','width','height','threads','passed','absolute_bound',
                                   'max_abs_delta','ratio','matched_scope','rust_source','plan_sha256')}
        record['canonical']=row['canonical']
        # This weightless route constructs CwSsimPyiqa directly; the model
        # engine environment is ignored by the executed native Rust callable.
        record['engine']='rust' if case['metric']=='cw-ssim-pyiqa' else engine
        record['relative_bound']=case.get('relative_bound',0)
        record['process_errors']=sum(bool(b.get('errors')) for b in blocks)
        measured=[b for b in blocks if 'rust_ns' in b and 'canonical_ns' in b]
        record['rust_ms']=statistics.median(b['rust_ns'] for b in measured)/1e6 if measured else None
        record['canonical_ms']=statistics.median(b['canonical_ns'] for b in measured)/1e6 if measured else None
        record['expected_cases']=sum(c['metric']==row['metric'] and not disposition(c) for c in cases.values())
        record['excluded_contract']=disposition(case)
        record['summary_sha256']=sha(path)
        record['blocks']=len(blocks)
        for role in ('rust','canonical'):
            rss=[]
            for memory in path.parent.glob('block-*/'+role+'/memory.txt'):
                found=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',memory.read_text())
                if found:rss.append(int(found[1]))
            record[role+'_max_process_rss_kbytes']=max(rss) if rss else None
        resources=[]
        for log in sorted(path.parent.glob('block-*-wrapper.log')):
            match=re.search(r'run-heavy: done rc=(\d+) .*?peak-RSS ([\d.]+)GiB \| min-avail (\d+)MiB \| peak-load ([\d.]+)',log.read_text())
            if match:
                resources.append((int(match[1]),float(match[2]),int(match[3]),float(match[4])))
        record['wrapper_receipts']=len(resources)
        record['wrapper_peak_rss_gib']=max((r[1] for r in resources),default=None)
        record['wrapper_min_available_mib']=min((r[2] for r in resources),default=None)
        record['wrapper_peak_load']=max((r[3] for r in resources),default=None)
        rows.append(record)
    return rows


def project(rows, preferred_sources=None):
    preferred_sources=preferred_sources or {}
    grouped=defaultdict(list)
    for row in rows:
        if not row['excluded_contract']:
            grouped[row['metric']].append(row)
    result={}
    for name, entries in grouped.items():
        expected=sum(next(r['expected_cases'] for r in entries if r['plan_sha256']==digest)
                     for digest in {r['plan_sha256'] for r in entries})
        errored=sum(bool(r.get('process_errors')) for r in entries)
        failed=sum(r['passed'] is False and not r.get('process_errors') for r in entries)
        passed=sum(r['passed'] is True for r in entries)
        deltas=[r['max_abs_delta'] for r in entries if r['max_abs_delta'] is not None]
        if failed:
            check=f'{failed} findings'+(f'; {errored} errors' if errored else '')
        elif errored:
            check=f'{errored} process errors'
        elif passed==len(entries):
            check=f'{passed}/{expected} pass'
        else:
            check=f'Δ {max(deltas):.2g}' if deltas else 'unmeasured'
        # The row has room for one representative ratio. Keep its actual shape,
        # worker count and sample cohort in the expanded evidence model.
        eligible=[r for r in entries if r['ratio'] is not None]
        preferred=preferred_sources.get(name)
        candidates=[r for r in eligible if not preferred or r['rust_source']==preferred]
        key=None
        # Runtime failures stay visible in `check` and the case ledger. They
        # do not erase a separate completed measurement; numerical findings do.
        if candidates and not failed:
            group_counts=defaultdict(int)
            for r in candidates:
                group_counts[(r['width'],r['height'],r['threads'],r['rust_source'],r['plan_sha256'])]+=1
            def preference(r):
                size=(r['width'],r['height'])
                group=(r['width'],r['height'],r['threads'],r['rust_source'],r['plan_sha256'])
                return (r['threads']!=1, -min(3,group_counts[group]),size!=(1024,1024),
                        size!=(512,512),-r.get('blocks',0),-r['width']*r['height'],r['threads'])
            chosen=min(candidates,key=preference)
            key=(chosen['width'],chosen['height'],chosen['threads'],chosen['rust_source'],chosen['plan_sha256'])
            selected=[r for r in candidates if (r['width'],r['height'],r['threads'],r['rust_source'],r['plan_sha256'])==key]
        else:
            selected=[]
        result[name]=dict(canonical=entries[0]['canonical'],check=check,
            ratio=statistics.median(r['ratio'] for r in selected) if selected else None,
            ratio_width=key[0] if key else None,ratio_height=key[1] if key else None,
            ratio_threads=key[2] if key else None,ratio_cases=len(selected),
            ratio_source=key[3] if key else None,ratio_plan_sha256=key[4] if key else None,
            ratio_engine=selected[0].get('engine') if selected else None,
            ratio_blocks=min(r.get('blocks',0) for r in selected) if selected else 0,
            worst_ratio=max(r['ratio'] for r in eligible) if eligible else None,
            observed_cases=len(entries),expected_cases=expected,passed_cases=passed,failed_cases=failed,
            process_error_cases=errored,
            max_abs_delta=max(deltas) if deltas else None,
            rust_sources=sorted({r['rust_source'] for r in entries}),
            plan_sha256=sorted({r['plan_sha256'] for r in entries}),qualified=False)
    return result


def attach_goldens(value,goldens,receipt_sha256):
    for name,receipt in goldens['metrics'].items():
        if receipt['checks_passed']!=receipt['checks_expected']:
            raise ValueError('incomplete golden receipt: '+name)
        if name not in value:
            raise ValueError('golden receipt has no comparative metric: '+name)
        row=value[name]
        row['golden_checks']=receipt
        if not row.get('failed_cases') and not row.get('process_error_cases'):
            row['check']=f"{receipt['checks_passed']}/{receipt['checks_expected']} pass"
        row['golden_receipt_sha256']=receipt_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('cohorts',nargs='+',type=Path)
    p.add_argument('--dispositions',type=Path,help='Explicit plan-sha:metric -> contract-mismatch reason')
    p.add_argument('--goldens',type=Path,help='Separate completed local golden-test receipts')
    p.add_argument('--prefer-source',action='append',default=[],metavar='METRIC=COMMIT',help='Select an explicitly revised source for the representative ratio; retain all historical cases')
    a=p.parse_args()
    a.output.mkdir(exist_ok=True)
    dispositions=json.loads(a.dispositions.read_text()) if a.dispositions else {}
    combined=[];sources={};published=[]
    for root in a.cohorts:
        label=root.name
        if label in sources:
            label=root.parent.name+'-'+label
        if label in sources:
            raise ValueError('cohort publication labels still collide: '+str(root))
        artifacts=[]
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
                artifact=a.output/(label+'-'+metric+'.tsv')
                artifact.write_bytes(payload)
                published.append((artifact,'cohort'))
                artifacts.append(dict(file=artifact.name,sha256=sha(artifact)))
        else:
            artifact=a.output/(label+'.tsv')
            artifact.write_bytes(data)
            published.append((artifact,'cohort'))
            artifacts.append(dict(file=artifact.name,sha256=sha(artifact)))
        absolute=root.absolute()
        raw='~/'+str(absolute.relative_to(Path.home())) if absolute.is_relative_to(Path.home()) else str(absolute)
        sources[label]=dict(raw_directory=raw,plan_sha256=sha(root/'plan.json'),cases=len(rows),
                               expected_cases=len(json.loads((root/'plan.json').read_text())['cases']),artifacts=artifacts)
        combined.extend(rows)
    identities=[(r['plan_sha256'],r['case']) for r in combined]
    if len(identities)!=len(set(identities)):
        raise ValueError('duplicate acquired cases across selected cohorts')
    value=project(combined,dict(item.split('=',1) for item in a.prefer_source))
    if a.goldens:
        attach_goldens(value,json.loads(a.goldens.read_text()),sha(a.goldens))
    text=json.dumps(value,indent=2)+'\n'
    if len(text.encode())>30_000:
        # Compact projection; per-case detail remains in the indexed TSVs.
        text=json.dumps(value,separators=(',',':'))+'\n'
    if len(text.encode())>30_000:
        shards=[];part={}
        def emit():
            name=f'results-{len(shards):02d}.json'
            path=a.output/name
            path.write_text(json.dumps(part,indent=2)+'\n')
            published.append((path,'summary'))
            if path.stat().st_size>30_000:raise ValueError('projection shard exceeds generated-data cap')
            shards.append(dict(file=name,sha256=sha(path),metrics=len(part)))
        for name,row in sorted(value.items()):
            candidate=dict(part,**{name:row})
            if len(json.dumps(candidate,indent=2).encode())>24_000 and part:
                emit();part={}
            part[name]=row
        if part:emit()
        text=json.dumps(dict(schema=2,metrics=len(value),shards=shards),indent=2)+'\n'
    (a.output/'results.json').write_text(text)
    cohort_index=json.dumps(sources,indent=2)+'\n'
    if len(cohort_index.encode())>30_000:
        cohort_index=json.dumps(sources,separators=(',',':'))+'\n'
    if len(cohort_index.encode())>30_000:
        raise ValueError('cohort index exceeds generated-data cap')
    (a.output/'cohorts.json').write_text(cohort_index)
    published.extend([(a.output/'results.json','summary'),(a.output/'cohorts.json','provenance')])
    register_outputs(published)
    print(json.dumps(dict(metrics=len(value),cases=len(combined),
                         excluded=sum(bool(r['excluded_contract']) for r in combined))))


if __name__=='__main__':
    main()
