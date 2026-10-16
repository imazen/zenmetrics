#!/usr/bin/env python3
"""Run bounded, separately admitted IQA cells and retain exact raw JSONL."""
import argparse, datetime, hashlib, json, os, pathlib, platform, socket, subprocess, time
P=pathlib.Path

def sha(p):
    h=hashlib.sha256()
    with P(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def append(p,row):
    with p.open('a') as f:f.write(json.dumps(row,allow_nan=False,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def metadata(binary):
    cpu=platform.processor()
    if P('/proc/cpuinfo').exists():
        text=P('/proc/cpuinfo').read_text();cpu=next(s.split(':',1)[1].strip() for s in text.splitlines() if s.startswith('model name'))
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True).stdout.strip() if __import__('shutil').which('nvidia-smi') else None
    return dict(host=socket.gethostname().split('.')[0],cpu_model=cpu,os=platform.platform(),arch=platform.machine(),gpu=gpu,impl='zenmetrics',impl_version='source',impl_binary_sha=sha(binary),device='cpu',simd_tier='pending runtime dispatch probe')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('binary',type=P);p.add_argument('pairs',type=P);p.add_argument('root',type=P)
    p.add_argument('--sizes',default='cid22,2k');p.add_argument('--metrics');p.add_argument('--threads',default='1,20');p.add_argument('--model-root',type=P);p.add_argument('--models-json',type=P);a=p.parse_args()
    a.root.mkdir(exist_ok=True,parents=True);(a.root/'cells').mkdir(exist_ok=True);(a.root/'logs').mkdir(exist_ok=True)
    models=json.loads(a.models_json.read_text()) if a.models_json else {}
    meta=metadata(a.binary);out=a.root/f"{meta['host']}.jsonl"
    inventory=json.loads(subprocess.check_output([a.binary,'inventory'],text=True));pairs=json.loads(a.pairs.read_text());(a.root/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    done=set()
    if out.exists():
        for s in out.read_text().splitlines():
            r=json.loads(s);done.add((r['metric_name'],r['size_name'],r['threads'],r['mode'],r['device']))
    selected=set(a.metrics.split(',')) if a.metrics else {r['metric_name'] for r in inventory}
    assert selected<={r['metric_name'] for r in inventory}
    for pair in pairs:
        if pair['size_name'] not in a.sizes.split(','):continue
        paths={k:str((a.pairs.parent/pair[k]).resolve()) for k in ['reference','distorted']}
        hashes={k:sha(v) for k,v in paths.items()};pair_sha=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
        for count in map(int,a.threads.split(',')):
            for metric in inventory:
                name=metric['metric_name']
                if name not in selected:continue
                row=dict(meta | pair | paths,metric_name=name,metric_family=name.split('-')[0],threads=count,mode='score',pair_sha256=pair_sha,input_sha256=hashes,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                row.update({k:None for k in ['n','median_ns','p10_ns','p90_ns','mean_ns','score','ref_score','abs_delta','relative_delta','load_before','load_after']})
                key=(name,pair['size_name'],count,'score','cpu')
                if key in done:continue
                config=a.root/'cells'/f"{name}-{pair['size_name']}-{count}t.json";config.write_text(json.dumps(row)+'\n');log=a.root/'logs'/f'{config.stem}.log'
                env=os.environ.copy()
                for k in ['RAYON_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RTEN_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]=str(count)
                env['ZENBENCH_CPU_POLICY']='unrestricted'
                env['ZENBENCH_NO_SAVE']='1'
                env['TMPDIR']=str(a.root/'scratch');P(env['TMPDIR']).mkdir(exist_ok=True)
                if a.model_root:env['ZENMETRICS_NN_MODEL_DIR']=str(a.model_root)
                if name in models:env['ZENMETRICS_NN_MODEL_DIR']=models[name]
                env['ZENMETRICS_NN_ACCEPT_WEIGHTS']='ckdn,afine,dreamsim,liqe,hyperiqa,paq2piq'
                env['ZENMETRICS_NN_ENGINE']='rten'
                command=['env',*[f'{k}={env[k]}' for k in ['RAYON_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RTEN_NUM_THREADS','OPENBLAS_NUM_THREADS']],'ZENBENCH_CPU_POLICY=unrestricted','ZENBENCH_NO_SAVE=1',str(a.binary.resolve()),str(config.resolve())]
                if platform.system()=='Linux':
                    cpus=sorted(os.sched_getaffinity(0));chosen=[2 if 2 in cpus else cpus[0]] if count==1 else cpus
                    row['cpu_affinity']=chosen;row['placement_policy']='unrestricted descriptive hybrid/all-core study; explicit OS thread affinity'
                    config.write_text(json.dumps(row)+'\n');command=['taskset','-c',','.join(map(str,chosen)),*command]
                if platform.system()=='Linux':command=['flock',str(P.home()/'tmp/i265-heavy.lock'),str(P.home()/'work/claudehints/scripts/run-heavy'),'--mem','16G','--jobs','16','--','timeout','--kill-after=20s','1100s',*command]
                print('RUN',config.stem,flush=True)
                with log.open('x') as f:r=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
                text=log.read_text();rows=[json.loads(s[8:]) for s in text.splitlines() if s.startswith('IQA_ROW ')]
                if r.returncode or len(rows) not in (1,2):
                    if 'busy load' in text:raise RuntimeError(f'busy host refused; retain {log}; retry after load drops')
                    rows=[dict(row,status=f'skipped:process_exit_{r.returncode}',n=0,notes=f'No completed timing; full log {log}',load_before=None,load_after=None)]
                for result in rows:
                    import re
                    match=re.search(r'run-heavy: done rc=(\d+) (\d+)s \| peak-RSS ([0-9.]+)GiB \| min-avail (\d+)MiB \| peak-load ([0-9.]+)',text)
                    if match:result['resource']=dict(exit_code=int(match[1]),seconds=int(match[2]),peak_rss_gib=float(match[3]),min_available_mib=int(match[4]),peak_load=float(match[5]))
                    result['log_sha256']=sha(log);append(out,result)
                print('RESULT',config.stem,[(r['mode'],r['status']) for r in rows],flush=True)
                # Map capability will be emitted by the native-map arm once qualified.
if __name__=='__main__':main()
