#!/usr/bin/env python3
"""Read-only graph/session capability audit; CPU fallback stays disabled."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def probe(a):
    import onnxruntime as ort
    ort.disable_telemetry_events()
    ort.set_default_logger_severity(0)
    ort.set_default_logger_verbosity(1)
    core=a.bundle/'runtime/libonnxruntime.1.31.0.dylib'
    plugin=a.bundle/'runtime/libonnxruntime_providers_webgpu.dylib'
    installed=Path(ort.__file__).parent/'capi/libonnxruntime.1.31.0.dylib'
    if ort.__version__!='1.31.0' or sha(core)!=sha(installed):
        raise ValueError('matching-wheel core differs from the frozen ORT core')
    ort.register_execution_provider_library('webgpu',str(plugin))
    devices=list(ort.get_ep_devices())
    if a.info:
        print('ORT_BUILD',ort.get_build_info())
        print('REGISTER',ort.register_execution_provider_library.__doc__)
        print('ADD_DEVICES',ort.SessionOptions.add_provider_for_devices.__doc__)
        for device in devices:
            print('DEVICE',[(name,str(getattr(device,name))) for name in dir(device) if not name.startswith('_')])
        return
    gpu=[d for d in devices if d.ep_name=='WebGpuExecutionProvider' and str(d.device.type).lower().endswith('gpu')]
    if len(gpu)!=1:
        raise ValueError('require exactly one WebGPU GPU: '+repr(devices))
    model=a.bundle/'models'/a.model/(a.model+'_full.onnx')
    identity=dict(model_sha256=sha(model),frozen_core_sha256=sha(core),wheel_core_sha256=sha(installed),plugin_sha256=sha(plugin),
        ort_version=ort.__version__,ort_build=ort.get_build_info(),fallback_disabled=True,python_wrapper_fallback=False,
        device=str(gpu[0]),session_only=True,intra_op=1,inter_op=1,deterministic=True,script_sha256=sha(Path(__file__)))
    options=ort.SessionOptions()
    options.log_severity_level=0
    options.log_verbosity_level=1
    options.logid=a.model+'-strict-webgpu-audit'
    options.intra_op_num_threads=1
    options.inter_op_num_threads=1
    options.use_deterministic_compute=True
    options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
    options.add_provider_for_devices([gpu[0]],{})
    try:
        session=ort.InferenceSession(str(model),sess_options=options,enable_fallback=False)
        identity.update(status='session_created',providers=session.get_providers())
    except Exception as error:
        identity.update(status='session_refused',error=str(error),error_type=type(error).__name__)
    import onnx
    graph=onnx.load_model(str(model),load_external_data=False).graph
    value_types={v.name:onnx.helper.printable_type(v.type) for v in [*graph.input,*graph.output,*graph.value_info]}
    initializers={v.name:dict(dtype=onnx.TensorProto.DataType.Name(v.data_type),dims=list(v.dims)) for v in graph.initializer}
    nodes=[dict(index=i,name=n.name,op=n.op_type,domain=n.domain,inputs=list(n.input),outputs=list(n.output),
        input_types=[value_types.get(name,initializers.get(name)) for name in n.input]) for i,n in enumerate(graph.node)]
    identity.update(onnx_version=onnx.__version__,nodes=nodes,model_sha256_after=sha(model))
    if identity['model_sha256']!=identity['model_sha256_after']:raise ValueError('frozen model changed')
    a.output.write_text(json.dumps(identity,indent=2)+'\n')
    print('AUDIT_RESULT',json.dumps({k:v for k,v in identity.items() if k!='nodes'}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=['afine','arniqa','ckdn','deepdc'])
    p.add_argument('--info',action='store_true')
    p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.worker:
        return probe(a)
    import fcntl
    a.output.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','RTEN_NUM_THREADS','ORT_NUM_THREADS','RAYON_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        env[key]='1'
    env.update(WGPU_BACKEND='metal',TMPDIR=str(a.output.resolve()))
    with (Path.home()/'tmp/mac-heavy.lock').open('a') as lock:
        print('waiting shared mac-heavy.lock',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        print('acquired shared mac-heavy.lock',flush=True)
        for model in ([a.model] if a.model else ['afine','arniqa','ckdn','deepdc']):
            command=[str(Path.home()/'work/claudehints/scripts/run-heavy'),'--mem','12G','--jobs','8','--',sys.executable,str(Path(__file__).resolve()),'--worker','--bundle',str(a.bundle.resolve()),'--output',str(a.output.resolve()/(model+'.json')),'--model',model]
            if a.info:command.append('--info')
            with (a.output/(model+'.log')).open('xb') as log:
                rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            print(json.dumps(dict(model=model,returncode=rc)),flush=True)
            if rc:raise RuntimeError('diagnostic failed; preserve output '+str(a.output))
            if a.info:break


if __name__=='__main__':
    main()
