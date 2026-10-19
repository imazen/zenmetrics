#!/usr/bin/env python3
"""Join a frozen VIF correction matrix to retained unchanged-author scores."""
import argparse
import csv
import io
import json
from pathlib import Path
import struct

from octave_primary import sha


def value(directory, arm, metric, stored):
    path=directory/(metric+'-'+arm)/'stdout.log'
    probes=[json.loads(s.removeprefix('IQA_PROBE ')) for s in path.read_text().splitlines() if s.startswith('IQA_PROBE ')]
    assert stored['returncode']==0 and len(probes)==1 and probes[0]['metric_name']==metric
    assert probes[0]['scores']==stored['scores'] and len(stored['scores'])==1
    result=struct.unpack('>d',bytes.fromhex(stored['scores'][0][1]))[0]
    assert stored['values']==[result]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['matrix','primary','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    identity=json.loads((a.matrix/'identity.json').read_text())
    original=json.loads((a.primary/'identity.json').read_text())
    assert identity['vif_family'] and identity['metrics']==['vif','vifvec']
    assert identity['inputs_sha256']==original['inputs_sha256'] and original['absolute_bound']==1e-9
    assert identity['binary_sha256']['baseline']==original['binary_sha256']
    done=json.loads((a.matrix/'completion.json').read_text())
    assert len(done['groups'])==24 and all(r['returncode']==0 for r in done['groups'])
    primary={ (r['pair'],r['size'],r['threads']):r for r in json.loads((a.primary/'completion.json').read_text())['cases'] }
    assert len(primary)==24
    rows=[]
    for group in done['groups']:
        directory=a.matrix/group['group'];plan=json.loads((directory/'plan.json').read_text())
        key=(plan['input']['pair'],plan['input']['size_name'],plan['threads']);prior=primary.pop(key)
        results=json.loads((directory/'results.json').read_text())['rows'];assert len(results)==2
        vif,vec=results;assert vif['metric']=='vif' and vec['metric']=='vifvec'
        before=value(directory,'baseline','vif',vif['baseline']);after=value(directory,'candidate','vif',vif['candidate'])
        assert before==prior['ours'] and prior['reference_values'][0]==prior['reference_values'][1]
        canon=prior['reference_values'][0]
        primary_case=a.primary/f"vif-{key[0]}-{key[1]}-t{key[2]}"
        canon_actual=[float(s.split()[2]) for s in (primary_case/'canonical-stdout.log').read_text().splitlines() if s.startswith(('COLD ','SAMPLE '))]
        assert canon_actual==prior['reference_values']
        if vec['baseline']['returncode']==vec['candidate']['returncode']==0:
            value(directory,'baseline','vifvec',vec['baseline']);value(directory,'candidate','vifvec',vec['candidate'])
            assert vec['baseline']['scores']==vec['candidate']['scores']
            vec_status='bit-identical'
        else:
            assert min(plan['input']['width'],plan['input']['height'])<72
            assert vec['baseline']['returncode']==vec['candidate']['returncode']==1 and vec['both_refused']
            assert vec['baseline']['stderr']==vec['candidate']['stderr'] and '72' in vec['candidate']['stderr']
            vec_status='unchanged-size-refusal'
        for role in ['reference','distorted']:
            config=json.loads((primary_case/'config.json').read_text())
            assert plan['input'][role+'_sha256']==sha(Path(config[role]))
        row=dict(metric='vif',pair=key[0],size=key[1],threads=key[2],baseline=before,candidate=after,canonical=canon,
            baseline_abs_delta=abs(before-canon),candidate_abs_delta=abs(after-canon),before_after_abs_delta=abs(before-after),bound=1e-9,
            baseline_status='pass' if abs(before-canon)<=1e-9 else 'numeric-finding',candidate_status='pass' if abs(after-canon)<=1e-9 else 'numeric-finding',
            vifvec_control=vec_status,qualified_ratio=False)
        rows.append(row)
    assert not primary
    a.output.mkdir();shards=[]
    for pair in ['pairA','pairB','pairC']:
        selected=[r for r in rows if r['pair']==pair];assert len(selected)==8
        stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(selected[0]),delimiter='\t',lineterminator='\n')
        w.writeheader();w.writerows(selected);data=stream.getvalue().encode();assert len(data)<=30000
        name='candidate-vif-'+pair+'.tsv';(a.output/name).write_bytes(data)
        shards.append(dict(path=name,sha256=sha(a.output/name),bytes=len(data)))
    index=dict(cases=len(rows),baseline_passes=sum(r['baseline_status']=='pass' for r in rows),candidate_passes=sum(r['candidate_status']=='pass' for r in rows),
        candidate_max_abs_delta=max(r['candidate_abs_delta'] for r in rows),before_after_max_abs_delta=max(r['before_after_abs_delta'] for r in rows),
        vifvec_bit_identical=sum(r['vifvec_control']=='bit-identical' for r in rows),vifvec_matching_refusals=sum(r['vifvec_control']=='unchanged-size-refusal' for r in rows),
        matrix_identity=identity,matrix_identity_sha256=sha(a.matrix/'identity.json'),matrix_completion_sha256=sha(a.matrix/'completion.json'),
        primary_identity_sha256=sha(a.primary/'identity.json'),primary_completion_sha256=sha(a.primary/'completion.json'),shards=shards,
        worker_sha256=sha(Path(__file__)),qualified_timing_cells=0)
    (a.output/'candidate-vif.json').write_text(json.dumps(index,indent=2)+'\n');print(json.dumps(index))


if __name__=='__main__':main()
