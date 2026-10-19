"""Focused changed-piece controls against retained and new nonmetric receipts."""
import argparse,copy,hashlib,json
from pathlib import Path
import gate_trace,monitor_join

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def retained(directory):
    value=json.loads((directory/'attribution.json').read_text());outer=json.loads((directory/'outer.json').read_text());config=json.loads((directory/'config.json').read_text())
    paths=dict(binary=Path(outer['launch']['executable']),config=directory/'config.json',source=Path(config['source']),trace=directory/'trace.jsonl',outer_receipt=directory/'outer.json')
    gate_trace.verify_artifact_join(value,paths)
    # Historical bytes remain untouched. Derive a stronger receipt from the
    # separately retained held/launch records; do not claim fields were emitted.
    held=json.loads((directory/'ownership/held.json').read_text());launch=json.loads((directory/'ownership/launch.json').read_text())
    if launch!=outer['launch']:raise ValueError('retained launch mismatch')
    outer['held']=held;outer['controller']=held['owner']
    return value,outer,paths

def run(old_dirs,current,out):
    out.mkdir(exist_ok=False,parents=True);results={}
    for index,directory in enumerate(old_dirs):
        v,o,paths=retained(directory);interval=monitor_join.verify_outer(v,o)
        (out/f'retained-{index}-derived-outer.json').write_text(json.dumps(o,indent=2))
        results[str(directory)]=dict(common_offset=interval,original_outer_sha256=sha(paths['outer_receipt']),source='derived held/controller fields from verified original receipts')
        for role in ('controller','child'):
            wrong=copy.deepcopy(o)
            if role=='controller':
                # Change all three claimed owner records consistently; the
                # independently observed snapshot must still contradict them.
                for record in (wrong['controller'],wrong['held']['owner'],wrong['launch']['owner']):record['start_ticks']+=1
            else:wrong['launch']['child']['start_ticks']+=1
            try:monitor_join.verify_outer(v,wrong)
            except ValueError:pass
            else:raise AssertionError('retained identity control accepted '+role)
    value=json.loads((current/'attribution.json').read_text());outer=json.loads((current/'outer.json').read_text());config=json.loads((current/'config.json').read_text())
    paths=dict(binary=Path(outer['launch']['executable']),config=current/'config.json',source=Path(config['source']),trace=current/'trace.jsonl',outer_receipt=current/'outer.json',held_receipt=current/'ownership/held.json',launch_receipt=current/'ownership/launch.json')
    results['current']=monitor_join.verify_artifacts(value,paths)|dict(common_offset=monitor_join.verify_outer(value,outer))
    # Reviewer's exact pairwise-overlap counterexample, through real file/hash
    # and schema/event joins. Values here are expressly SYNTHETIC clock inputs.
    snapshot=copy.deepcopy(value['snapshots'][0]);snapshot['sample_monotonic_ns']=100;snapshot['counter_interval_start_ns']=100
    base=copy.deepcopy(value);base.pop('monitor_events');base['snapshots']=[snapshot];base['checks']=copy.deepcopy(value['checks'][:2])
    for c,t in zip(base['checks'],(100,300)):
        c.update(snapshot_id=snapshot['snapshot_id'],snapshot_sha256=gate_trace.snapshot_hash(snapshot),checked_monotonic_ns=t,observation_age_ns=t-100)
    synthetic_outer=copy.deepcopy(outer);synthetic_outer.update(launch_before_ns=100,ready_received_ns=200,go_ns=300,exit_observed_ns=400,finally_ns=401)
    synthetic_outer['held']['observed_monotonic_ns']=99;synthetic_outer['launch']['observed_monotonic_ns']=250
    op=out/'counterexample-outer.json';op.write_text(json.dumps(synthetic_outer,indent=2));tp=out/'counterexample-trace.jsonl'
    events=[dict(kind='snapshot',value=snapshot)]+[dict(kind='check',value=c) for c in base['checks']]
    tp.write_text('\n'.join(json.dumps(e) for e in events)+'\n');base.update(trace_sha256=sha(tp),outer_receipt_sha256=sha(op))
    simple={k:p for k,p in paths.items() if k not in ('held_receipt','launch_receipt')};simple.update(trace=tp,outer_receipt=op)
    gate_trace.verify_artifact_join(base,simple)
    # Both old pairwise overlap predicates are true, but no common offset exists.
    assert all(100+c['checked_monotonic_ns']<=400 and 200+c['checked_monotonic_ns']>=300 for c in base['checks'])
    try:monitor_join.verify_outer(base,synthetic_outer)
    except ValueError as error:results['pairwise_counterexample']=str(error)
    else:raise AssertionError('pairwise counterexample accepted')
    (out/'counterexample-attribution.json').write_text(json.dumps(base,indent=2))
    positive=copy.deepcopy(base);positive['checks'][1].update(checked_monotonic_ns=100,observation_age_ns=0)
    # Same schema for a feasible shared boundary offset.
    pt=out/'feasible-trace.jsonl'
    pt.write_text('\n'.join(json.dumps(e) for e in [dict(kind='snapshot',value=snapshot)]+[dict(kind='check',value=c) for c in positive['checks']])+'\n')
    positive['trace_sha256']=sha(pt)
    gate_trace.verify_artifact_join(positive,simple|{'trace':pt})
    assert monitor_join.verify_outer(positive,synthetic_outer)==dict(min_ns=200,max_ns=200,exact=False)
    bad=copy.deepcopy(positive);bad['snapshots'][0]['sample_monotonic_ns']=301
    try:monitor_join.common_offset(bad,synthetic_outer)
    except ValueError:results['unused_snapshot_clock_refusal']=True
    else:raise AssertionError('unused snapshot interval ignored')
    negatives=[]
    for mutation in ('gate-age','read-age','publication','clean-reason','decision-count','missing-decision','stale-fresh','gate-order'):
        bad=copy.deepcopy(value);gates=[e['value'] for e in bad['monitor_events'] if e['kind']=='gate'];g=gates[0]
        if mutation=='gate-age':g['gate_observation_age_ns']+=1
        elif mutation=='read-age':g['read']['age_ns']+=1
        elif mutation=='publication':g['read']['publication_monotonic_ns']+=1
        elif mutation=='clean-reason':g['clean']=not g['clean']
        elif mutation=='decision-count':g['state']['heavy_process_count']+=1
        elif mutation=='missing-decision':next(x for x in gates if x['read']['status']=='missing')['clean']=True
        elif mutation=='stale-fresh':next(x for x in gates if x['read']['status']=='stale')['read']['status']='fresh'
        else:
            positions=[i for i,e in enumerate(bad['monitor_events']) if e['kind']=='gate'][:2]
            a,b=positions;bad['monitor_events'][a],bad['monitor_events'][b]=bad['monitor_events'][b],bad['monitor_events'][a]
        trace=out/(mutation+'.jsonl');events=[dict(kind='snapshot',value=s) for s in bad['snapshots']]+[dict(kind='check',value=c) for c in bad['checks']]+bad['monitor_events']
        trace.write_text('\n'.join(json.dumps(e) for e in events)+'\n');bad['trace_sha256']=sha(trace)
        try:monitor_join.verify_artifacts(bad,paths|{'trace':trace})
        except ValueError:negatives.append(mutation)
        else:raise AssertionError('semantic trace negative accepted '+mutation)
    results['rehashed_gate_negatives']=negatives
    (out/'result.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('current',type=Path);p.add_argument('output',type=Path);p.add_argument('retained',nargs='+',type=Path);a=p.parse_args();run(a.retained,a.current,a.output)
