#!/usr/bin/env python3
"""Check retained NR reference-image scores against native auxiliary columns."""
import argparse
import json
import math
import os
from pathlib import Path
import struct

from compare_canonical import compare, sha
from haarpsi_primary import execute


def reference_scores(text):
    values=[]
    for line in text.splitlines():
        if line.startswith(('PAIR_REFERENCE ', 'REFERENCE_SCORE ')):
            value=float(line.split()[1])
            if not math.isfinite(value):
                raise ValueError('nonfinite canonical auxiliary score')
            values.append(value)
    if not values:
        raise ValueError('canonical auxiliary scores missing')
    return values


def validate_bindings(plan, relocations):
    for asset,expected in plan['bindings'].items():
        replacement=relocations.get(asset,asset)
        if replacement!=asset:
            if not Path(asset).name.startswith('prepare_') or any(asset in part for case in plan['cases'] for part in case['canonical']['command']):
                raise ValueError('Only nonexecuted preparation provenance may be relocated')
        if sha(replacement)!=expected:
            raise ValueError('bound source changed: '+asset)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args();manifest=json.loads(a.manifest.read_text())
    a.output.mkdir();rows=[]
    for item in manifest['cases']:
        source=Path(item['results']);plan=json.loads((source/'plan.json').read_text())
        case=next(c for c in plan['cases'] if c['id']==item['case'])
        relocations=item.get('binding_relocations',{})
        validate_bindings(plan,relocations)
        summary=source/case['id']/'summary.json'
        value=json.loads(summary.read_text())
        if value['passed'] is not True:raise ValueError('headline check has not passed')
        logs=sorted((source/case['id']).glob('block-*/canonical/stdout.log'))
        references=[v for log in logs for v in reference_scores(log.read_text())]
        directory=a.output/case['id'];directory.mkdir()
        config={k:case[k] for k in ['width','height','reference','distorted','threads']}
        config.update(metric_name=case['metric'],probe_only=True,only_mode='score',timing_phase='warm',locked_process_accounting=True)
        config_path=directory/'config.json';config_path.write_text(json.dumps(config)+'\n')
        env=dict(os.environ,TMPDIR=str(directory),LC_ALL='C',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1')
        for key in ['RAYON_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[key]=str(case['threads'])
        command=[plan['binary'],str(config_path.resolve())]
        rc,_=execute(command,env,directory,'native',1100)
        result={'metric':case['metric'],'case':case['id'],'width':case['width'],'height':case['height'],'threads':case['threads'],
                'plan_sha256':sha(source/'plan.json'),'headline_summary_sha256':sha(summary),
                'canonical_logs':{str(log):sha(log) for log in logs},'canonical_reference_scores':references,
                'absolute_bound':case['absolute_bound'],'relative_bound':case.get('relative_bound',0),
                'process_exit':rc,'qualification':'Accuracy-only auxiliary-column comparison; no timing ratio'}
        if rc:
            result.update(passed=False,error='Native probe failed; retained stdout and stderr')
        else:
            probes=[json.loads(line.removeprefix('IQA_PROBE ')) for line in (directory/'native-stdout.log').read_text().splitlines() if line.startswith('IQA_PROBE ')]
            if len(probes)!=1:raise ValueError('expected one native probe')
            auxiliary=[(name,struct.unpack('>d',bytes.fromhex(bits))[0]) for name,bits in probes[0]['scores'] if name.endswith('_ref')]
            if len(auxiliary)!=1:raise ValueError('expected one reference-image auxiliary column')
            name,score=auxiliary[0]
            comparisons=[compare(score,reference,case['absolute_bound'],case.get('relative_bound',0)) for reference in references]
            result.update(column=name,native_reference_score=score,max_abs_delta=max(delta for delta,_ in comparisons),
                passed=False if any(ok is False for _,ok in comparisons) else None if any(ok is None for _,ok in comparisons) else True)
        validate_bindings(plan,relocations)
        result['binding_relocations']=relocations
        result['files']={path.name:sha(path) for path in directory.iterdir() if path.is_file()}
        (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');rows.append(result)
        print('AUXILIARY_CASE',case['id'],result['passed'],result.get('max_abs_delta'),flush=True)
    (a.output/'completion.json').write_text(json.dumps({'cases':rows,'manifest_sha256':sha(a.manifest),'worker_sha256':sha(__file__)},indent=2)+'\n')


if __name__=='__main__':main()
