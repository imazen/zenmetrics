#!/usr/bin/env python3
"""Materialize one compatible study-cell A/B plan; never run a comparison."""
import argparse
from pathlib import Path
import shutil
import sys

from analyze_benchmark_study import ingest, read_cells
from build_benchmark_study import ROOT, canonical, file_sha, load


def build(left_study,left_packaged,left_cell,right_study,right_packaged,right_cell,output,blocks,deadline):
    if type(blocks) is not int or not 2<=blocks<=1000 or type(deadline) is not int or not 1<=deadline<=86400:
        raise ValueError('declare 2..1000 blocks and positive bounded deadline')
    rows=[]
    identities=[]
    for study,packaged,sha in [(left_study,left_packaged,left_cell),(right_study,right_packaged,right_cell)]:
        # Reuse the complete materialization/identity checks, without any observed packets.
        ingest(study,packaged,[])
        summary,cells=read_cells(study)
        row=cells[sha]
        if row['status']!='ready' or row['measurement']!='wall':
            raise ValueError('paired latency requires qualified ordinary wall cells')
        build_meta=load(packaged/sha/'build.json')
        if build_meta['instrumented'] is not False or build_meta['profile']!='release':
            raise ValueError('paired latency requires ordinary release builds')
        rows.append((summary,row))
        identities.append(load(packaged/'identities.json'))
    (left_summary,left),(right_summary,right)=rows
    if (left['case']!=right['case'] or left_summary['operations_per_launch']!=right_summary['operations_per_launch']
            or any(left[key]!=right[key] for key in ['machine','domain','testbed','target'])
            or any(left['sources']['inputs'][side]['sha256']!=right['sources']['inputs'][side]['sha256'] for side in ['reference','distorted'])
            or left['expected_correctness']!=right['expected_correctness']):
        raise ValueError('pair cells differ in metric/workload/input/testbed/correctness; hardware and unlike metrics cannot be paired')
    if identities[0]['runner_sha']!=identities[1]['runner_sha']:
        raise ValueError('pair cells require the same selected native runner')
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    checker=output/'checker.py';shutil.copyfile(ROOT/'scripts/jobsys/check_benchmark_pair_output.py',checker)
    expected=output/'expected.json'
    expected.write_bytes(canonical(dict(operations=left_summary['operations_per_launch'],correctness=left['expected_correctness']))+b'\n')
    inputs={'checker':str(checker),'expected':str(expected)}
    arms=[]
    for label,packaged,row in [('baseline',left_packaged,left),('candidate',right_packaged,right)]:
        source=packaged/row['cell_sha'];target=output/label
        # Copy only the verified listed assets; do not capture unrelated directory contents.
        manifest=load(source/'bundle.json');target.mkdir()
        shutil.copyfile(source/'bundle.json',target/'bundle.json')
        for name,artifact in manifest['files'].items():
            path=source/name
            if file_sha(path)!=artifact['sha256']: raise ValueError('changed packaged pair asset')
            dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
            if file_sha(dest)!=artifact['sha256']: raise ValueError('changed pair asset during copy')
            inputs[f'{label}_{name.replace(".","_")}']=str(dest)
        inputs[f'{label}_bundle']=str(target/'bundle.json')
        arms.append(dict(id=label,program=str(target/'driver'),args=['fleet',str(target/'case.json')],cwd=str(target),
            env={'RAYON_NUM_THREADS':str(row['case']['threads']),'ZENBENCH_OPERATIONS':str(left_summary['operations_per_launch']),
                 'ZENBENCH_RESULT_FILE':'${output}/driver.json','ZENMETRICS_PAIR_TIMING':'1'},
            revision=row['sources']['artifact']['build']['sha256'],timeout_seconds=left_summary['timeout_seconds']))
    plan=dict(schema=1,baseline=arms[0],candidates=[arms[1]],cases=[dict(id='compatible-cell',inputs=inputs)],
        metric=dict(extract=dict(kind='prefix',prefix='METRIC_NS='),unit='ns',
                    expected=left_summary['operations_per_launch'],divide_by=1),
        check=dict(kind='command',program=str(Path(sys.executable).resolve()),
                   args=[str(checker),'${baseline}','${candidate}',str(expected)],timeout_seconds=30),
        warmup='Declared driver warmup runs inside each fresh child; operation_ns excludes preparation and warmup',
        blocks=blocks,aa=True,deadline_seconds=deadline)
    (output/'pair.json').write_bytes(canonical(plan)+b'\n')
    record=dict(schema=1,left_cell=left['cell_sha'],right_cell=right['cell_sha'],runner_sha=identities[0]['runner_sha'],
                pair_sha=file_sha(output/'pair.json'),invocations=blocks*3,
                mode='exploratory external pair; no confirmation acceptance',executed=False)
    (output/'pair-identity.json').write_bytes(canonical(record)+b'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['left_study','left_packaged','left_cell','right_study','right_packaged','right_cell','output']:
        parser.add_argument(name,type=str if name.endswith('cell') else Path)
    parser.add_argument('--blocks',type=int,required=True);parser.add_argument('--deadline',type=int,required=True)
    args=parser.parse_args()
    try: print(canonical(build(**vars(args))).decode())
    except (ValueError,OSError,KeyError,TypeError) as error: parser.exit(2,f'pair plan refused: {error}\n')

if __name__=='__main__': main()
