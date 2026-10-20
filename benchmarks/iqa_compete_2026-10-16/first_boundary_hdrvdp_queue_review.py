#!/usr/bin/env python3
"""File-only source-review controls; caller must provide shared admission.

Uses synthetic receipts with the unchanged frozen queue validator. No scorer,
Octave, worker, preflight or subprocess is invoked. Findings characterize the
retained-summary validator, not native/reference numerical agreement.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def synthetic_case(root,queue,bits):
    root.mkdir();(root/'cases').mkdir()
    case={'id':'synthetic-review-case'}
    observations=[]
    for ordinal,role in enumerate(queue.ORDER):
        directory=root/'arms'/case['id']/f'{ordinal}-{role}';directory.mkdir(parents=True)
        log=directory/'control-stdout.log';log.write_text('synthetic source-review fixture; no metric executed\n')
        result=directory/'result.json'
        write(result,dict(case=case['id'],ordinal=ordinal,role=role,source_runtime_verified=True,
                          cohort_sha256='c'*64,seal_sha256='d'*64,passed=True,score_bits=bits,
                          artifacts={str(log):sha(log)}))
        observations.append(dict(ordinal=ordinal,role=role,result=str(result),result_sha256=sha(result),
                                 passed=True,score_bits=bits))
    write(root/'cases'/(case['id']+'.json'),dict(case=case['id'],cohort_sha256='c'*64,
        seal_sha256='d'*64,observations=observations,passed=True))
    return case


def review(frozen,manifest,output):
    cohort=json.loads((frozen/'cohort.json').read_text())
    module=frozen/'helpers/first_boundary_hdrvdp_arm_queue.py'
    if sha(module)!=cohort['bindings'][str(module)]:raise ValueError('frozen queue source differs')
    sys.path.insert(0,str(module.parent))
    queue=importlib.import_module('first_boundary_hdrvdp_arm_queue')
    output.mkdir()
    case=synthetic_case(output/'artifact-control',queue,'3ff0000000000000')
    baseline=queue.checked_case(output/'artifact-control',case,'c'*64,'d'*64)['passed']
    log=output/'artifact-control/arms/synthetic-review-case/0-default/control-stdout.log'
    row=json.loads(log.with_name('result.json').read_text())
    log.write_text('changed synthetic retained artifact\n')
    recorded_hash_differs=sha(log)!=row['artifacts'][str(log)]
    after=queue.checked_case(output/'artifact-control',case,'c'*64,'d'*64)['passed']
    case=synthetic_case(output/'missing-bits-control',queue,None)
    absent=queue.checked_case(output/'missing-bits-control',case,'c'*64,'d'*64)['passed']
    source_manifest=json.loads(manifest.read_text())
    thread_rows=[]
    for metric,path in source_manifest['metric_plans'].items():
        plan=json.loads(Path(path).read_text())
        selected=min((c for c in plan['cases'] if c['metric']==metric and c['matched_scope']),
                     key=lambda c:c['threads'])
        command=selected['canonical']['command']
        thread_rows.append(dict(metric=metric,source_threads=selected['threads'],
            explicit_thread_argument=command.count('{threads}'),
            formatted={str(t):[part.format_map(dict(selected,threads=t,samples=1,output=str(output)))
                              for part in command] for t in (1,8)}))
    result=dict(schema=1,metric_inference_executed=False,subprocesses_executed=False,
        controller_sha256=sha(__file__),frozen_queue_sha256=sha(module),
        manifest_sha256=sha(manifest),synthetic_receipts=True,
        findings=dict(retained_artifact=dict(baseline_accepted=baseline,artifact_hash_differs=recorded_hash_differs,
                                             accepted_after_artifact_change=after),
                      absent_bits=dict(all_four_score_bits=None,accepted=absent)),
        comparative_thread_rows=thread_rows,selected_metrics=len(thread_rows),
        explicit_thread_templates=sum(row['explicit_thread_argument']==1 for row in thread_rows),
        qualification='Source/protocol control only; no numerical, timing or native admission qualification')
    write(output/'result.json',result)
    print(json.dumps(dict(findings=result['findings'],selected_metrics=len(thread_rows),
                          explicit_thread_templates=result['explicit_thread_templates'],
                          metric_inference_executed=False)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen','manifest','output'):parser.add_argument(name,type=Path)
    args=parser.parse_args();review(args.frozen.resolve(strict=True),args.manifest.resolve(strict=True),args.output)


if __name__=='__main__':main()
