#!/usr/bin/env python3
"""Bitwise gate: original parser vs warm cached parameters, including every native VMAF-v1 map."""
import argparse,hashlib,json,os,pathlib,subprocess
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('reference',type=P);p.add_argument('candidate',type=P);p.add_argument('pairs',type=P);p.add_argument('output',type=P)
a=p.parse_args();a.output.mkdir(exist_ok=False)
metrics=['vmaf','vmaf-neg','vmaf-4k','vmaf-v1','vmaf-studio601','vmaf-neg-studio601']
records=[]
for pair in json.loads(a.pairs.read_text()):
    if pair['size_name'] not in ['cid22','2k']:continue
    for threads in [1,20]:
        for metric in metrics:
            results=[];key=f"{metric}-{pair['size_name']}-{threads}t"
            for role,binary in [('reference',a.reference),('candidate',a.candidate)]:
                root=a.output/f'{key}-{role}';root.mkdir()
                config=dict(pair,metric_name=metric,probe_only=True,reference=str(a.pairs.parent/pair['reference']),distorted=str(a.pairs.parent/pair['distorted']))
                if metric=='vmaf-v1':config['map_dump_dir']=str(root/'maps')
                path=root/'probe.json';path.write_text(json.dumps(config)+'\n')
                command=['env',f'RAYON_NUM_THREADS={threads}',f'OMP_NUM_THREADS={threads}',f'RTEN_NUM_THREADS={threads}',str(binary),str(path)]
                result=subprocess.run(command,capture_output=True,text=True,check=True)
                (root/'stdout.log').write_text(result.stdout);(root/'stderr.log').write_text(result.stderr)
                rows=[json.loads(s[len('IQA_PROBE '):]) for s in result.stdout.splitlines() if s.startswith('IQA_PROBE ')];assert len(rows)==1
                value=rows[0];assert value.pop('model_preload_adapter')==(role=='candidate')
                if 'maps' in value:
                    for m in value['maps']['maps']:
                        m['sha256']=hashlib.sha256((root/'maps'/m['file']).read_bytes()).hexdigest()
                results.append(value)
            assert results[0]==results[1],(key,results)
            record=dict(key=key,scores=results[0]['scores'],native_map_count=len(results[0].get('maps',{}).get('maps',[])),identical=True)
            records.append(record);print(json.dumps(record),flush=True)
receipt=dict(reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),cases=records,all_identical=True)
(a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
