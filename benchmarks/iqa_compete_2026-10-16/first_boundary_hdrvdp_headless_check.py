#!/usr/bin/env python3
"""Admitted source-only headless preparation; no Octave/scorer execution."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','output','results','old_helpers'):parser.add_argument(name,type=Path)
    parser.add_argument('--parent-sha',required=True)
    parser.add_argument('--heavy-wrapper',type=Path,required=True)
    parser.add_argument('--lock',type=Path,action='append',required=True)
    args=parser.parse_args()
    helpers=Path(__file__).parent
    for path in helpers.glob('*.py'):ast.parse(path.read_text())
    sys.path.insert(0,str(helpers))
    import first_boundary_hdrvdp_headless as headless
    import first_boundary_hdrvdp_arm_worker as worker
    receipt=headless.prepare(args.parent,args.parent_sha,args.output,args.heavy_wrapper,args.lock)
    plan=headless.load(receipt['parent_plan']);old=headless.load(args.parent)
    headless.verify(plan['bindings'])
    exact_bodies=0;exact_sources=0
    old_root=args.parent.resolve().parent;root=args.output.resolve()
    for name,expected in old['bindings'].items():
        source=Path(name)
        if source.is_relative_to(old_root) and not source.is_relative_to(old_root/'variants'):
            copied=root/source.relative_to(old_root)
            if headless.sha(copied)!=expected:raise AssertionError('original source bytes changed')
            exact_sources+=1
    for before,after in zip(old['cases'],plan['cases']):
        for key in before:
            if key!='arms' and before[key]!=after[key]:raise AssertionError('parent case contract changed')
        for role in ('default','resident'):
            previous=before['arms'][role]['command'];command=after['arms'][role]['command']
            if command!=headless.headless(previous,old_root,root):raise AssertionError('headless argv differs')
            if headless.body(Path(previous[-1]).read_text())!=headless.body(Path(command[-1]).read_text()):
                raise AssertionError('original evaluator body changed')
            exact_bodies+=1
    if len(plan['preflight'])!=9 or any(c['command'].count('--no-window-system')!=1 for c in plan['preflight']):
        raise AssertionError('nine headless preflights required')
    queue=Path(receipt['cohort']).parent
    cohort,parent=worker.context(queue,receipt['cohort_sha256'])
    selected=worker.selected_cases(cohort,parent)
    identities={(c['metric'],c['phase'],c['threads']) for c in selected}
    if identities!={(m,'warm',t) for m in ('hdrvdp','hdrvdp3') for t in (1,8)}:
        raise AssertionError('four warm one/eight cases required')
    for case in selected:
        if case['absolute_bound']!=(0.005 if case['metric']=='hdrvdp' else None):
            raise AssertionError('existing bounds changed')
    args.results.mkdir()
    # Rerun the actual admitted-audit file controls with this additive source.
    spec=importlib.util.spec_from_file_location('bound_audit_checks',helpers/'first_boundary_hdrvdp_queue_audit_check.py')
    controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls)
    controls.run(args.old_helpers.resolve(),helpers,args.results/'retained-audit')
    headless.write(args.results/'result.json',dict(receipt,passed=True,original_source_files_exact=exact_sources,
        original_timed_bodies_exact=exact_bodies,preflights_prepared=9,warm_cases_prepared=4,
        retained_audit_static_controls=6,metric_inference_executed=False,matched_scope=False,
        qualification='source/static only; fresh headless preflights/ABBA/native controls pending'))
    print(json.dumps(dict(passed=True,original_timed_bodies_exact=exact_bodies,
        warm_cases=4,metric_inference_executed=False,cohort_sha256=receipt['cohort_sha256'])),flush=True)


if __name__=='__main__':main()
