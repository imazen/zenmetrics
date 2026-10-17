#!/usr/bin/env python3
"""Measure native ORT worker affinities for default/explicit session budgets."""
import argparse,json,os,pathlib,subprocess,sys
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=P);p.add_argument('--child',choices=['default','1','20']);a=p.parse_args()
if a.child:
    import numpy as np,onnx,onnxruntime as ort
    from onnx import helper,TensorProto
    graph=helper.make_graph([helper.make_node('Mul',['x','x'],['y'])],'thread-control',
        [helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,16])],
        [helper.make_tensor_value_info('y',TensorProto.FLOAT,[1,16])])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)],ir_version=10)
    options=ort.SessionOptions()
    if a.child!='default':options.intra_op_num_threads=int(a.child);options.inter_op_num_threads=1
    baseline=[]
    for status in sorted(P('/proc/self/task').glob('*/status')):
        fields=dict(line.split(':',1) for line in status.read_text().splitlines() if ':' in line)
        baseline.append(dict(tid=int(status.parent.name),name=fields['Name'].strip(),affinity=fields['Cpus_allowed_list'].strip()))
    session=ort.InferenceSession(model.SerializeToString(),options,providers=['CPUExecutionProvider'])
    session.run(None,{'x':np.ones((1,16),dtype=np.float32)})
    masks=[];tasks=[]
    for status in sorted(P('/proc/self/task').glob('*/status')):
        line=next(line for line in status.read_text().splitlines() if line.startswith('Cpus_allowed_list:'))
        masks.append(line.split(':',1)[1].strip())
        fields=dict(line.split(':',1) for line in status.read_text().splitlines() if ':' in line)
        tasks.append(dict(tid=int(status.parent.name),name=fields['Name'].strip(),affinity=fields['Cpus_allowed_list'].strip(),stat=(status.parent/'stat').read_text(),wchan=(status.parent/'wchan').read_text()))
    print(json.dumps(dict(configuration=a.child,package_version=ort.__version__,
                         requested_intra_threads=options.intra_op_num_threads,
                         requested_inter_threads=options.inter_op_num_threads,
                         worker_affinity_masks=masks,baseline_threads=baseline,native_threads=tasks,providers=session.get_providers())))
else:
    assert not a.output.exists(),'Preserve earlier observations'
    rows=[]
    for setting in ['default','1','20']:
        command=[sys.executable,str(P(__file__).resolve()),str(a.output),'--child',setting]
        if setting!='20':command=['taskset','-c','2',*command]
        # Isolate the native ORT pool from numpy's independent BLAS pool.
        env=os.environ|{key:'1' for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}
        result=subprocess.run(command,env=env,check=True,capture_output=True,text=True)
        row=json.loads(result.stdout);row['stderr']=result.stderr;rows.append(row)
        a.output.with_name(a.output.stem+'-cases.json').write_text(json.dumps(dict(cases=rows),indent=2)+'\n')
    assert all(mask=='2' for mask in rows[1]['worker_affinity_masks']),'Explicit one-thread session escaped the caller affinity'
    assert len(rows[1]['worker_affinity_masks'])==1,'Explicit one-thread control created additional native workers'
    assert len(rows[2]['worker_affinity_masks'])==20,'Explicit twenty-thread control did not create the requested native pool'
    a.output.write_text(json.dumps(dict(cases=rows,scope='Correctness/configuration probe, CPU execution provider of the installed ORT distribution; not a performance measurement'),indent=2)+'\n')
    print(json.dumps(dict(cases=[dict(configuration=row['configuration'],native_threads=len(row['worker_affinity_masks']),affinity_masks=sorted(set(row['worker_affinity_masks']))) for row in rows])))
