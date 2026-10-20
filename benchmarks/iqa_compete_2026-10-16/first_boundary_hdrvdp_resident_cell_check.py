#!/usr/bin/env python3
"""File-only resident-cell preparation/source controls; no evaluator execution."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','parent','seal','bundle','results'):parser.add_argument(name,type=Path)
    parser.add_argument('--parent-sha',required=True)
    parser.add_argument('--seal-sha',required=True)
    args=parser.parse_args()
    ast.parse(args.source.read_text())
    loader=importlib.util.spec_from_file_location('source_hdr_cell',args.source)
    cell=importlib.util.module_from_spec(loader);loader.loader.exec_module(cell)
    prepared=cell.prepare(args.parent,args.parent_sha,args.seal,args.seal_sha,args.bundle)
    cfg=cell.load(prepared['config']);cell.verify(cfg['bindings'])
    controls=[]
    for metric,spec in cfg['routes'].items():
        original=cell.split_original(Path(spec['original_wrapper']).read_text(),metric)
        for name,path in spec['parts'].items():
            if Path(path).read_text()!=original[name]:raise AssertionError('source slice changed')
        for samples in (1,3):
            script=cell.script_for(spec,samples)
            start=script.index('clock=tic();\n')+len('clock=tic();\n')
            end=script.index('elapsed=toc(clock)',start)
            if (script[start:end]!=original['body'] or script.count('clock=tic();')!=1
                    or not script.startswith(original['setup']+original['inputs'])
                    or not script.endswith(original['suffix'])
                    or f'for call_index=0:{samples}\n' not in script):
                raise AssertionError('timing/input/cache boundary differs')
            controls.append(dict(metric=metric,samples=samples,exact_body_sha256=hashlib.sha256(
                script[start:end].encode()).hexdigest(),passed=True))
        for bad in (0,-1,True):
            try:cell.script_for(spec,bad)
            except ValueError:pass
            else:raise AssertionError('invalid explicit sample budget accepted')
        plan=cell.load(prepared['metric_plans'][metric]);cell.verify(plan['bindings'])
        if {c['threads'] for c in plan['cases']}!={1,8}:raise AssertionError('thread inventory differs')
        for row in plan['cases']:
            command=row['canonical']['command']
            if (row['matched_scope'] is not False or command.count('{threads}')!=1
                    or command.count('{samples}')!=1):raise AssertionError('scope/template differs')
            replacements=dict(row,samples=3)
            argv=[part.format_map(replacements) for part in command]
            # Check formatted positional budgets without entering cell().
            if argv[-2:]!=[str(row['threads']),'3']:raise AssertionError('formatted budget differs')
            expected_bound=0.005 if metric=='hdrvdp' else None
            if row['absolute_bound']!=expected_bound:raise AssertionError('existing bound changed')
    args.results.mkdir()
    cell.write(args.results/'result.json',dict(passed=True,controls=controls,
        source_sha256=cell.sha(args.source),config_sha256=prepared['config_sha256'],
        metric_inference_executed=False,qualification='source/static only; ABBA/native parity pending'))
    print(json.dumps(dict(passed=True,source_controls=len(controls),**prepared)),flush=True)


if __name__=='__main__':main()
