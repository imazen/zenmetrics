#!/usr/bin/env python3
# declare_bf944_repair.py — CPU-VENDOR-class repair wave for bf944 (SOTA-944 P1 bigcodec).
#
# The tier-matched wave assumed AVX2 boxes are interchangeable; MEASURED FALSE:
# 22 append-block slots (2 per 17-slot group) use vendor-specific approximation
# instructions (AMD vs Intel rsqrt/rcp tables), so G-BF1 bitwise requires the
# re-extraction to run on the SAME CPU VENDOR (x tier) as the bf924 extractor.
# Most cells already have a class-matched blob from the wave's multi-worker
# re-scoring; this declares ONLY the remainder (repair_cells.json census):
#   bf944amd-<x>  -> jobs/_pool944amdv4/runlist.tsv   (tower, ian)
#   bf944int-<x>  -> jobs/_pool944intelv4/runlist.tsv (i265, jason)
# Cells are matched inside the bf924 manifests by the python replication of
# JobId::of (sorted-keys serde encoding — proven exact vs zenfleet-ctl ids).
import gzip, hashlib, io, json, os, subprocess, sys

BUCKET='zentrain'
E = dict(os.environ)
for line in open(os.path.expanduser('~/.config/cloudflare/r2-credentials')):
    line=line.strip()
    if line.startswith('R2_') and '=' in line:
        k,v=line.split('=',1); E[k]=v.strip().strip('"').strip("'")
E['AWS_ACCESS_KEY_ID']=E['R2_ACCESS_KEY_ID']; E['AWS_SECRET_ACCESS_KEY']=E['R2_SECRET_ACCESS_KEY']; E['AWS_REGION']='auto'
EP='https://%s.r2.cloudflarestorage.com'%E['R2_ACCOUNT_ID']

def s5(*a, stdin=None):
    r=subprocess.run(['s5cmd','--endpoint-url',EP,*a],env=E,capture_output=True,input=stdin)
    r.err=r.stderr.decode(errors='replace')[:200]; return r

def jid_sorted(kind, inputs):
    obj={"inputs":sorted(set(inputs)),"kind":kind}
    return hashlib.sha256(json.dumps(obj,separators=(',',':')).encode()).hexdigest()

SRC={ }
TAR_RUNS=[("zavif","mandfix4-zenavif-1782593621",8),("zjxll","jxl-lossy-vardct-1782609551",24),("zwebp","mandfix2-zenwebp-1782584881",9),("zjxlm","jxl-modular-1782596759",10),("zpng","mandfix2-zenpng-1782584881",2)]
for t,s,n in TAR_RUNS:
    for i in range(n): SRC[f'{t}-t{i}']=(f's3://{BUCKET}/jxl-lossy/runs/{s}/variants/box-{i}.tar','tar')
SRC['zjl2']=('canonical/2026-06-27/zenjpeg_lossy/encodes','enc')

def main():
    dry='--dry-run' in sys.argv
    rep=json.load(open(os.path.expanduser('~/tmp/bigcodec944/repair_cells.json')))
    # class -> [(run944, image_path, id944, id944po)]
    by_run={}  # (cls, x) -> set(id944)
    for cls, items in rep.items():
        tag='amd' if cls=='amdv4' else 'int'
        for r944, ip, i4, i4p in items:
            x=r944.split('-',1)[1]
            by_run.setdefault((tag,x),set()).add(i4)
    runlists={'amd':[],'int':[]}; totals={'amd':0,'int':0}; fail=0
    for (tag,x), wanted in sorted(by_run.items()):
        old=f'bf924-{x}'
        r=s5('cat',f's3://{BUCKET}/jobs/{old}/manifest.json.gz')
        if r.returncode!=0: print(f'FAIL fetch {old}: {r.err}'); fail+=1; continue
        cells=json.load(gzip.open(io.BytesIO(r.stdout)))
        sub=[]
        for c in cells:
            k44={"kind":"score_file","metrics":["zensim-foldapp2"]}
            if jid_sorted(k44,c['inputs']) in wanted:
                cc=json.loads(json.dumps(c)); cc['kind']['metrics']=['zensim-foldapp2']
                sub.append(cc)
        if len(sub)!=len(wanted):
            print(f'FAIL {tag}-{x}: matched {len(sub)} != wanted {len(wanted)}'); fail+=1; continue
        new=f'bf944{tag}-{x}'; src,mode=SRC[x]
        totals[tag]+=len(sub)
        if not dry:
            buf=io.BytesIO()
            with gzip.GzipFile(fileobj=buf,mode='wb') as gz: gz.write(json.dumps(sub).encode())
            u=s5('pipe',f's3://{BUCKET}/jobs/{new}/manifest.json.gz',stdin=buf.getvalue())
            if u.returncode!=0: print(f'FAIL upload {new}: {u.err}'); fail+=1; continue
            if mode=='tar':
                u=s5('cp',f's3://{BUCKET}/jobs/{old}/variant_index.tsv',f's3://{BUCKET}/jobs/{new}/variant_index.tsv')
                if u.returncode!=0: print(f'FAIL idx {new}: {u.err}'); fail+=1; continue
        runlists[tag].append(f'{new}\t{src}\t{mode}')
        print(f'ok {new}: {len(sub)} cells',flush=True)
    if fail: print(f'ABORT: {fail} failures'); sys.exit(1)
    exp={'amd':44679,'int':39291}
    assert totals==exp, (totals,exp)
    for tag,pool in (('amd','_pool944amdv4'),('int','_pool944intelv4')):
        rl='\n'.join(runlists[tag])+'\n'
        if dry: print(f'[dry] {pool}: {len(runlists[tag])} runs {totals[tag]} cells')
        else:
            u=s5('pipe',f's3://{BUCKET}/jobs/{pool}/runlist.tsv',stdin=rl.encode())
            assert u.returncode==0, u.err
            print(f'runlist: {len(runlists[tag])} runs -> s3://{BUCKET}/jobs/{pool}/runlist.tsv ({totals[tag]} cells)')
    print('DONE')

if __name__=='__main__':
    main()
