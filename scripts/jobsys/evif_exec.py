#!/usr/bin/env python3
"""zenfleet EVIF Encode/Metric executor; worker atomically publishes both CAS objects.

The metric output is a labelled EVIF, not the ordinary JSON metric row. Only
source-built frozen SafeSyn scores are targets. Butteraugli/CVVDP are diagnostics.
All binaries are pinned and preinstalled; no downloads or fallback on a worker.
"""
import csv
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

TEACHER = 'ssim2-safesyn-b7c2b4b3'
PINS = {'evif':'232434619478f3ce076e895f3b4342b00f86e8b633a0da5123867d2e45dab800',
        'teacher':'2e6d5f77f54062c9a08c6390668eb075474ff8772926720b5475d8ca179eb3a6',
        'diagnostics':'37c3bc112d8bdc86b949005f8fcafd419510b88b7dd7c7a8938ba59251c4e50f'}
CAPABILITIES=['evif-cas-v1',TEACHER,'evif-view-batch-v1','evif-diagnostics-v1']

def sha(data): return hashlib.sha256(data).hexdigest()
def validate(job):
    cell=job['cell']; kind=job['kind']; codec=cell['codec']
    if codec not in ['evif:zenjpeg-expert','evif:zenjpeg-color']:
        raise ValueError('unsupported EVIF family')
    params=json.loads(cell['knob_tuple_json'])
    allowed = {'quality','subsampling','quant_scale','zero_bias_scale','adaptive_quant','deringing','trellis','xyb'} if codec.endswith('expert') else {'quality','color','tables','downsampling','deringing','adaptive_quant','auto_optimize'}
    if not isinstance(params,dict) or not params or set(params)-allowed:
        raise ValueError('unknown/empty EVIF params')
    if params.get('quality')!=cell['q']:raise ValueError('cell quality and EVIF quality differ')
    if kind['kind']=='encode':
        if kind['codec']!=codec or kind['q']!=cell['q'] or json.loads(kind['knobs'])!=params or kind.get('hdr',False):
            raise ValueError('encode identity differs from cell')
    elif kind['kind']=='metric':
        if kind['metric']!=TEACHER:raise ValueError('unknown frozen teacher')
    else:raise ValueError('EVIF executor serves Encode/Metric')
    data=Path(cell['image_path']).read_bytes()
    if data[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('EVIF SDR source must be prepared PNG')
    w,h=struct.unpack('>II',data[16:24])
    if not w or not h or w*h>=1_000_000:raise ValueError('source must be below 1MP')
    inputs=set(job['inputs'])
    if sha(data) not in inputs or sha(json.dumps(params,sort_keys=True,separators=(',',':')).encode()) not in inputs:
        raise ValueError('job inputs must bind reference bytes and exact canonical params')
    return codec.split(':',1)[1],params

@lru_cache(maxsize=1)
def binaries():
    root=Path(os.environ['ZEN_EVIF_BIN_DIR']); result={}
    for key,name in [('evif','evif'),('teacher','chroma-frozen-teacher'),('diagnostics','zenmetrics-diagnostics')]:
        path=root/name
        if sha(path.read_bytes())!=PINS[key]:raise ValueError('binary pin mismatch: '+key)
        result[key]=path
    return result

def execute(job):
    if job.get('cell',{}).get('codec')=='evif:diagnostics':
        from evif_diagnostics import execute as diagnostic_execute
        return diagnostic_execute(job,PINS)
    if job.get('cell',{}).get('codec')=='evif:batch':
        from evif_batch import execute as batch_execute
        return batch_execute(job,PINS)
    family,params=validate(job);bins=binaries()
    scratch=Path(os.environ['TMPDIR'])
    if not scratch.is_absolute() or str(scratch).startswith(('/tmp','/run','/dev/shm')):
        raise ValueError('disk-backed explicit TMPDIR required')
    # Transient working files are bounded by a single <1MP cell. CAS outputs
    # are emitted together; no artifact is deleted from a prior attempt.
    with tempfile.TemporaryDirectory(prefix='evif-cell-',dir=scratch) as td:
        root=Path(td); logs=[]
        def run(args,stdin=None):
            p=subprocess.run(list(map(str,args)),input=stdin,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
            return p.stdout
        paramfile=root/'params.jsonl';paramfile.write_text(json.dumps(params)+'\n')
        out=root/'render'
        run([bins['evif'],'render','--family',family,'--params',paramfile,'--format','evif','--rendition','png','--out',out,job['cell']['image_path']])
        rows=list(csv.DictReader((out/'render.tsv').open(),delimiter='\t'))
        if len(rows)!=1:raise ValueError('EVIF refused the declared cell')
        container=Path(rows[0]['path']); extracted=root/'extracted'
        run([bins['evif'],'extract',container,extracted])
        envelope_path=next(extracted.glob('*.digests.json')); envelope=json.loads(envelope_path.read_text())
        jpeg=next(extracted.glob('*.jpeg')).read_bytes();candidate=next(extracted.glob('*.png'))
        if job['kind']['kind']=='metric':
            pair=dict(reference=job['cell']['image_path'],distorted=str(candidate))
            label=json.loads(run([bins['teacher']],(json.dumps(pair)+'\n').encode()))
            value=float(label['value'])
            if struct.pack('>d',value).hex()!=label['value_bits']:raise ValueError('teacher float transport differs')
            scores=[dict(metric='ssim2',variant='safesyn-teacher',implementation=label['implementation']+';source-helper-sha256='+PINS['teacher'],value=value,better='higher')]
            # These ports remain diagnostics and never enter the target/admission.
            pairs=root/'pairs.tsv'
            with pairs.open('w') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['ref_path','dist_path']);w.writerow([pair['reference'],pair['distorted']])
            import pyarrow.parquet as pq
            for metric in ['butteraugli','cvvdp']:
                diag=root/(metric+'.parquet')
                run([bins['diagnostics'],'score-pairs','--metric',metric,'--gpu-runtime','cpu','--pairs-tsv',pairs,'--out-parquet',diag,*(['--display-model','standard_4k'] if metric=='cvvdp' else [])])
                diagnostic=pq.read_table(diag).to_pylist()
                if len(diagnostic)!=1:raise ValueError('diagnostic transport row count differs')
                columns=['butteraugli_max','butteraugli_pnorm3'] if metric=='butteraugli' else [k for k in diagnostic[0] if k.startswith('cvvdp_cpu_imazen_v')]
                if not columns:raise ValueError('missing CVVDP port identity')
                for column in columns:
                    if column not in diagnostic[0] or diagnostic[0][column] is None:raise ValueError('missing diagnostic '+column)
                    scores.append(dict(metric=column,variant='diagnostic',implementation='zenmetrics@0a61830db89a768ab9462f166d1412e948e4efa9;display=standard_4k;exe-sha256='+PINS['diagnostics'],value=diagnostic[0][column],better='lower' if metric=='butteraugli' else 'higher'))
            envelope['scores']=scores
            scorefile=root/'scores.json';scorefile.write_text(json.dumps(envelope))
            scored=root/'scored.evif'
            run([bins['evif'],'score-import',container,scorefile,scored]);container=scored
        data=container.read_bytes()
        if len(data)+len(jpeg)+24>32*1024*1024:raise ValueError('EVIF output envelope exceeds worker limit')
        return b'ZFEVIF1\n'+struct.pack('<QQ',len(data),len(jpeg))+data+jpeg

def main():
    if sys.argv[1:]==['capabilities']:
        print('\n'.join(CAPABILITIES));return
    if sys.argv[1:]==['--serve']:
        source=sys.stdin.buffer; target=sys.stdout.buffer
        while True:
            length=source.read(4)
            if not length:return
            if len(length)!=4:raise ValueError('truncated job frame')
            count=struct.unpack('<I',length)[0]
            if count>1024*1024:raise ValueError('oversized job frame')
            data=source.read(count)
            if len(data)!=count:raise ValueError('truncated job frame')
            try:payload=execute(json.loads(data));status=0
            except Exception as error:payload=str(error).encode();status=1
            target.write(bytes([status])+struct.pack('<I',len(payload))+payload);target.flush()
        return
    sys.stdout.buffer.write(execute(json.load(sys.stdin)))
if __name__=='__main__':main()
