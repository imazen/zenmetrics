#!/usr/bin/env python3
"""Allowlisted public timing projection; omit scores, paths and private assets."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path

FIELDS=('host campaign source source_commit build_receipt_sha256 metric_name size_name width height threads '
        'timing_phase cold_repeat only_mode engine impl_binary_sha reference_sha256 distorted_sha256 '
        'cell_id attempt status cell_status qualified_timing warm_rounds_clean warm_rounds_flagged '
        'median_ns mean_ns p10_ns p90_ns n peak_rss_gib_rounded peak_rss_precision_gib '
        'os_time_max_rss_bytes memory_evidence_status config_sha256 log_sha256 sealed_result_sha256 '
        'capture_utc protocol failure_class preferred round_means_ns placement_policy placement_cpus simd_tier rayon_threads_observed numerical_qualification resource_guard_stop').split()


def key(row):
    return tuple(str(row.get(k)) for k in ('host','metric_name','width','height','threads','timing_phase',
        'cold_repeat','only_mode','engine','impl_binary_sha','reference_sha256','distorted_sha256'))


def public_row(row, meta, host, campaign):
    out={k:row.get(k) for k in FIELDS}
    error=str(row.get('error') or row.get('reason') or '').lower()
    failure=('resource_guard_monitored_memory' if row.get('resource_guard_stop')=='monitored_memory_guard' else
             'input_contract_absolute_luminance' if 'absolute' in error and ('luminance' in error or 'nits' in error) else
             'input_contract_fixed_window' if '11x11' in error else
             'insufficient_warm_samples' if row.get('status')=='failed' and error.strip()=='insufficient warm samples' else
             'timeout' if 'timeout' in error else
             'worker_failure_unclassified' if row.get('status')=='failed' else None)
    out.update(failure_class=failure,host=host,campaign=campaign,source_commit=meta.get('source_commit'),
        build_receipt_sha256=meta.get('build_receipt_sha256'),capture_utc=meta['capture']['captured_utc'],
        protocol='canonical-cold3-warm10-100ms-scalar',preferred=False)
    # Warm cells: each zenbench round's batch mean (ns), ';'-joined; timing only, no scores.
    rounds=row.get('round_means_ns')
    out['round_means_ns']=';'.join(str(round(float(v))) for v in rounds) if isinstance(rounds,list) and rounds else None
    # zenbench placement as applied to the worker (policy and CPU count, no paths), SIMD tier, observed threads.
    placement=row.get('cpu_placement') if isinstance(row.get('cpu_placement'),dict) else {}
    out['placement_policy']=placement.get('policy')
    out['placement_cpus']=len(placement['cpus']) if isinstance(placement.get('cpus'),list) else None
    if campaign.startswith('private-nn-'):
        out['numerical_qualification']=('failed_native_gate' if out['metric_name']=='dmm-pyiqa'
                                        else 'separate_from_timing_control')
    return out


# One logical cell per host across a parent campaign and its successors (reruns of the same
# measurement under a newer build or input contract); the binary identity is deliberately absent.
CELL_KEY=('metric_name','size_name','width','height','threads','timing_phase','cold_repeat','only_mode','device','impl')
CELL_RANK={'ok':0,'refused_by_design':1,'failed':2,'pending':3}


def in_scope(c):
    """The public projection's scope: score mode, 1/8 threads, below 8K, canonical sampling."""
    required=1 if c['timing_phase']=='cold' else 10
    return (c['only_mode']=='score' and max(c['width'],c['height'])<7680 and c['threads'] in (1,8)
            and c.get('coverage_first') is False and c.get('sample_target_ns')==100_000_000
            and c.get('required_samples')==required and (c['timing_phase']!='cold' or c['cold_repeat'] in (0,1,2)))


def cell_coverage(cells):
    """Count each logical cell once: its best state over every campaign that planned it."""
    best={}
    for key,state in cells:
        if best.get(key) is None or CELL_RANK[state]<CELL_RANK[best[key]]:best[key]=state
    return dict(planned=len(best),**{k:sum(v==k for v in best.values()) for k in CELL_RANK})


def public_provenance(meta, campaign):
    out={k:meta.get(k) for k in ['campaign','manifest_sha256','binary_sha256','inputs',
         'source_commit','build_receipt_sha256','policy','metrics','manifest_cells']}
    if campaign.startswith('private-nn-'):
        # Private model file names and local paths stay in the private capture.
        out['inputs']=dict(content_sha256=sorted(set((meta.get('inputs') or {}).values())))
    return out


ROUND_FIELDS = ['host','campaign','source','source_commit','metric_name','size_name','width','height',
    'threads','timing_phase','cold_repeat','only_mode','engine','impl_binary_sha','reference_sha256',
    'distorted_sha256','cell_id','attempt','round_index','time_ns','iterations','elapsed_total_ns',
    'compensated_total_ns','gate_clean','qualified_timing','preferred','peak_rss_gib_rounded',
    'capture_utc','protocol','measurement_kind']


def round_rows(public, raw):
    """Emit saved rounds, never split an aggregate into invented observations."""
    means=raw.get('round_means_ns') or []
    rounds=raw.get('zenbench_rounds') or []
    if not means and public['timing_phase']=='cold' and public['status']=='ok' and public.get('median_ns') is not None:
        means=[public['median_ns']]
    for i,mean in enumerate(means):
        row={k:public.get(k) for k in ROUND_FIELDS}
        sample=rounds[i] if i<len(rounds) else {}
        gate=sample.get('gate_clean')
        if gate is None and raw.get('warm_rounds_flagged')==0 and raw.get('warm_rounds_clean')==len(means):
            gate=True
        row.update(round_index=i,time_ns=mean,iterations=sample.get('iterations'),
            elapsed_total_ns=(sample.get('elapsed_ns') or [None])[0],
            compensated_total_ns=(sample.get('compensated_ns') or [None])[0],gate_clean=gate,
            measurement_kind='fresh_process_constructor_and_first_pair' if public['timing_phase']=='cold' else 'warm_zenbench_round_mean_per_call')
        yield row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('captures',type=Path)
    p.add_argument('private_aggregates',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    all_rows=[];all_rounds=[];provenance=[];coverage={}
    for host in ('mac','i265','r7900x'):
        root=a.captures/host;receipt=json.loads((root/'capture-receipt.json').read_text())
        campaigns=[];logical=[]
        for meta in receipt['campaigns']:
            campaign=meta['campaign'];rawdir=root/campaign
            sources=json.loads((rawdir/'sources.json').read_text());by_source={s['id']:s for s in sources}
            configs={};states={}
            for source in sources:
                r=json.loads((root/(campaign+'-reports')/(source['id']+'.json')).read_text())
                for cell in r['cells']:
                    c=cell['config'];configs[(source['id'],cell['cell_id'])]=c
                    states[(source['id'],cell['cell_id'])]=cell['status']
            included=[];excluded=Counter()
            for line in (rawdir/'observations.jsonl').read_text().splitlines():
                row=json.loads(line);c=configs[(row['source'],row['cell_id'])]
                if c['only_mode']!='score' or max(c['width'],c['height'])>=7680 or c['threads'] not in (1,8):
                    excluded['map_or_8k_or_other_threads']+=1;continue
                required=1 if c['timing_phase']=='cold' else 10
                if c.get('coverage_first') is not False or c.get('sample_target_ns')!=100_000_000 or c.get('required_samples')!=required:
                    excluded['noncanonical_protocol']+=1;continue
                if c['timing_phase']=='cold' and c['cold_repeat'] not in (0,1,2):
                    excluded['other_cold_repeat']+=1;continue
                out=public_row(row,meta,host,campaign)
                included.append(out);all_rows.append(out)
                for sample in round_rows(out,row):all_rounds.append((out,sample))
                states[(row['source'],row['cell_id'])]=row['cell_status']
            failures={(r['source'],r['cell_id']):r['failure_class'] for r in included if r['cell_status']=='failed'}
            for cell,state in states.items():
                c=configs[cell]
                if not in_scope(c):continue
                if state=='failed' and (failures.get(cell) or '').startswith('input_contract_'):state='refused_by_design'
                if state=='unsupported':state='refused_by_design'
                logical.append((tuple(c.get(k) for k in CELL_KEY),state if state in CELL_RANK else 'failed'))
            # Report snapshots and overlays are separate hashes, retained without paths.
            public_meta=public_provenance(meta,campaign)
            public_meta['capture_utc']=meta['capture']['captured_utc']
            public_meta['exporter_sha256']=meta['capture']['exporter_sha256']
            public_meta['report_hashes']=[s['id'] for s in sources]
            public_meta['sealed_result_hashes']={s['id']:list(s.get('sealed_result_overlays',{}).values()) for s in sources}
            campaigns.append(dict(campaign=campaign,manifest_cells=meta['manifest_cells'],saved_attempt_rows=len(included),
                qualified_attempt_rows=sum(r['qualified_timing'] is True for r in included),
                gate_flagged_attempt_rows=sum((r['warm_rounds_flagged'] or 0)>0 for r in included),
                observed_cell_statuses=dict(Counter(states.values())),excluded_attempts=dict(excluded),
                rejected_overlays=meta['capture']['rejected_overlay_cells'],
                failure_class_attempts=dict(Counter(r['failure_class'] for r in included if r['failure_class'])),
                exhausted_saved_cells=len({r['cell_id'] for r in included if r['cell_status']=='failed'})))
            provenance.append(dict(host=host,**public_meta))
        # Completed private classical timing aggregates were already allowlisted.
        path=a.private_aggregates/(host+'-observations.csv')
        private_protocol=json.loads((a.private_aggregates/(host+'-protocol.json')).read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=private_protocol['observations_sha256']:raise ValueError('private aggregate hash differs')
        for r in csv.DictReader(path.open()):
            if r['metric_name']!='cw-ssim-pyiqa':raise ValueError('unexpected private metric')
            out={k:r.get(k) for k in FIELDS};out.update(host=host,campaign='private-classical',
                source_commit=private_protocol['worker_source_commit'],build_receipt_sha256=private_protocol['build_receipt_sha256'],capture_utc=private_protocol['captured_utc'],
                protocol='canonical-cold3-warm10-100ms-scalar',preferred=False,status='ok',cell_status='ok')
            out['qualified_timing']=r['qualified_timing']=='True'
            for k in ('warm_rounds_clean','warm_rounds_flagged'):out[k]=int(r[k])
            all_rows.append(out)
            for sample in round_rows(out,{}):all_rounds.append((out,sample))
        protocol=a.private_aggregates/(host+'-protocol.json')
        # The existing allowlisted protocol is copied intact; it contains no raw assets.
        (a.output/(host+'-private-protocol.json')).write_bytes(protocol.read_bytes())
        coverage[host]=dict(campaigns=campaigns,cells=cell_coverage(logical),capture_window_utc=[min(c['capture']['captured_utc'] for c in receipt['campaigns']),max(c['capture']['captured_utc'] for c in receipt['campaigns'])])
    groups=defaultdict(list)
    for row in all_rows:groups[key(row)].append(row)
    for rows in groups.values():
        qualified=[r for r in rows if r['qualified_timing'] is True]
        if qualified:
            best=max(qualified,key=lambda r:(r['capture_utc'] or '',int(r['attempt'])))
            best['preferred']=True
    for public,sample in all_rounds:sample['preferred']=public['preferred']
    for host,meta in coverage.items():
        rows=[r for r in all_rows if r['host']==host]
        meta.update(saved_attempt_rows=len(rows),preferred_qualified_cells=sum(r['preferred'] for r in rows),
            exact_metric_names=sorted({r['metric_name'] for r in rows}),
            qualification_note='Campaign pending/status counts overlap retained parents and successors; never sum them. Preferred rows count each exact binary/input/protocol/config series once. No zero imputation.')
        host_rounds=[sample for public,sample in all_rounds if public['host']==host]
        meta['individual_measurement_rows']=len(host_rounds)
        meta['round_note']='Warm points are per-round per-call means; iterations are not separate measured points. Missing round arrays remain absent, including private warm aggregates.'
        with (a.output/(host+'-rounds.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=ROUND_FIELDS,lineterminator='\n');w.writeheader();w.writerows(host_rounds)
        with (a.output/(host+'-observations.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');w.writeheader();w.writerows(rows)
    (a.output/'sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (a.output/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    protocol=dict(protocol='canonical-cold3-warm10-100ms-scalar',cold='Three fresh processes; scorer constructor plus first full-pair evaluation timed; RGB loading outside clocks',warm='Ten clean zenbench rounds, 100ms target; full-pair evaluation after initialization',threads=[1,8],memory='12G guard, jobs8; Linux hard cgroup; macOS monitored process-tree RSS guard',qualification='status ok plus ten gate-clean warm rounds; numerical reproduction is separate',preferred_key=list(('host','metric_name','width','height','threads','timing_phase','cold_repeat','only_mode','engine','impl_binary_sha','reference_sha256','distorted_sha256')),privacy='No per-image scores, paths, weights, graphs, model URLs or private source',zensim='Exact worker route zensim only; no separately identified W-C or other bake is claimed',active_scope='Mac/i265 changed27 successors; r7900x full non-NN; older saved ordinary observations retained',version_policy='Different frozen binary identities remain separate series; no pooling',portable='5/3 portable panels excluded',completion='Partial saved-results capture, not campaign completion')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (a.output/'SHA256SUMS.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.iterdir()},indent=2)+'\n')
    print(json.dumps({h:{k:v[k] for k in ('saved_attempt_rows','preferred_qualified_cells','capture_window_utc')} for h,v in coverage.items()}))

if __name__=='__main__':main()
