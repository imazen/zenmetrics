"""Later zenfleet ScoreFile pass: read persisted EVIF PNGs, never encode."""
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import struct
import tempfile
from evif_batch import chunks, sha

CAPABILITY = 'evif-diagnostics-v1'


def execute(job, pins):
    cell=job['cell'];knobs=json.loads(cell['knob_tuple_json'])
    if cell['codec']!='evif:diagnostics' or job['kind']!={'kind':'score_file','metrics':[CAPABILITY]}:
        raise ValueError('unsupported EVIF diagnostic job')
    if set(knobs)!={'source_sha256','reference_pixels_sha256','width','height','evif_shas'}:
        raise ValueError('unknown diagnostic controls')
    reference=Path(cell['image_path']);reference_raw=reference.read_bytes()
    if sha(reference_raw)!=knobs['source_sha256'] or sha(reference_raw) not in job['inputs']:
        raise ValueError('unbound diagnostic reference')
    archive=Path(os.environ['ZEN_ARCHIVE_BLOBS_DIR'])
    if not archive.is_absolute() or not 0<knobs['width']*knobs['height']<1_000_000:
        raise ValueError('invalid archive/geometry')
    selected=knobs['evif_shas']
    if not selected or len(selected)>128 or len(set(selected))!=len(selected):
        raise ValueError('invalid diagnostic selection')
    bins=Path(os.environ['ZEN_EVIF_BIN_DIR']);binary=bins/'zenmetrics-diagnostics'
    if sha(binary.read_bytes())!=pins['diagnostics'] or pins['diagnostics'] not in job['inputs']:
        raise ValueError('unbound diagnostic implementation')
    stage_base=archive/'.staging';stage_base.mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='evif-diagnostics-',dir=stage_base))
    catalog=[]
    for index,digest in enumerate(selected):
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest) or digest not in job['inputs']:
            raise ValueError('invalid/unbound diagnostic container')
        data=(archive/digest[:2]/digest[2:4]/digest).read_bytes()
        if sha(data)!=digest:raise ValueError('diagnostic CAS bytes differ')
        parts=chunks(data);target=json.loads(parts['scor'])['scores']
        if len(target)!=1 or target[0]['metric']!='ssim2':raise ValueError('diagnostics require frozen target-only container')
        png_sha=sha(parts['PNGF']);png=stage/(png_sha+'.png')
        if not png.exists():png.write_bytes(parts['PNGF'])
        catalog.append(dict(index=str(index),evif_sha256=digest,jpeg_sha256=sha(parts['BITS'][4:]),candidate=png,png_sha256=png_sha,teacher=target[0],scores={},diffmaps={}))
    pairs=stage/'pairs.tsv'
    with pairs.open('w') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['ref_path','dist_path','image_path','codec','q','knob_tuple_json'])
        for row in catalog:writer.writerow([reference,row['candidate'],row['index'],'evif:diagnostics',95,'{}'])
    import pyarrow.parquet as pq
    for metric in ['butteraugli','cvvdp']:
        output=stage/(metric+'.parquet')
        args=[str(binary),'score-pairs','--metric',metric,'--gpu-runtime','cpu','--group-by-ref','--pairs-tsv',str(pairs),'--out-parquet',str(output)]
        if metric=='cvvdp':args+=['--display-model','standard_4k']
        with (stage/(metric+'.log')).open('wb') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
        actual={r['image_path']:r for r in pq.read_table(output).to_pylist()}
        if len(actual)!=len(catalog):raise ValueError('diagnostic row count differs')
        for row in catalog:
            record=actual[row['index']]
            columns=['butteraugli_max','butteraugli_pnorm3'] if metric=='butteraugli' else [k for k in record if k.startswith('cvvdp_cpu_imazen_v')]
            if not columns or any(record[c] is None for c in columns):raise ValueError('diagnostic score missing')
            row['scores'].update({c:record[c] for c in columns})
    # This pinned binary predates file:// variants. Its existing local-shard
    # transport reads authenticated PNGs through a tiny content-addressed index.
    # Only that index travels through S3; the rasters stay on the archive mount.
    unique={r['png_sha256']:r['candidate'] for r in catalog}
    index_file=stage/'variant-index.tsv'
    index_file.write_text(''.join(f'{digest}\t0\t{path.stat().st_size}\t{path.name}\n' for digest,path in sorted(unique.items())))
    index_sha=sha(index_file.read_bytes())
    uri='s3://'+os.environ['ZEN_BUCKET']+'/'+os.environ['ZEN_RUN']+'/diagnostic-indices/'+index_sha+'.tsv'
    with (stage/'index-upload.log').open('wb') as log:
        subprocess.run(['s5cmd','--endpoint-url',os.environ['ZEN_R2_ENDPOINT'],'cp',str(index_file),uri],stdout=log,stderr=subprocess.STDOUT,check=True)
    env=os.environ.copy();env.update(ZEN_VARIANT_INDEX_URI=uri,ZEN_VARIANTS_LOCAL_DIR=str(stage))
    cache={}
    import numpy as np
    with (stage/'native-maps.log').open('wb') as log:
        child=subprocess.Popen([str(binary),'jobexec','--serve'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env=env)
        try:
            for row in catalog:
                for metric in ['butteraugli','cvvdp@standard_4k']:
                    key=(row['png_sha256'],metric)
                    if key not in cache:
                        request=dict(kind=dict(kind='diffmap',metric=metric),inputs=[row['png_sha256']],cell=dict(image_path=str(reference),codec='zenpng',q=95,knob_tuple_json='{}'))
                        message=json.dumps(request).encode();child.stdin.write(struct.pack('<I',len(message))+message);child.stdin.flush()
                        header=child.stdout.read(5)
                        if len(header)!=5:raise ValueError('native map process ended; inspect retained native-maps.log')
                        status,count=header[0],struct.unpack('<I',header[1:])[0]
                        if count>32*1024*1024:raise ValueError('native map exceeds bounded object size')
                        payload=child.stdout.read(count)
                        if len(payload)!=count:raise ValueError('truncated native map response')
                        if status:raise ValueError('native map refused: '+payload.decode(errors='replace'))
                        raw=gzip.decompress(payload);parts=raw.split(b'\n',3)
                        w,h=knobs['width'],knobs['height']
                        if len(parts)!=4 or parts[:3]!=[b'Pf',f'{w} {h}'.encode(),b'-1.0'] or len(parts[3])!=w*h*4:
                            raise ValueError('native map geometry/format differs')
                        if not np.isfinite(np.frombuffer(parts[3],dtype='<f4')).all():raise ValueError('nonfinite native map')
                        file=stage/(row['png_sha256']+'.'+metric.replace('@','-')+'.pfm.gz');file.write_bytes(payload)
                        cache[key]=dict(sha256=sha(payload),width=w,height=h,format='gzip-PFM-f32le',input_binding=sha((knobs['reference_pixels_sha256']+'|'+row['png_sha256']+'|'+metric+'|'+pins['diagnostics']).encode()))
                    row['diffmaps'][metric]=cache[key]
            child.stdin.close()
            if child.wait(timeout=30):raise ValueError('native map process failed after EOF')
        finally:
            if child.poll() is None:child.kill();child.wait()
    output=[]
    for row in catalog:output.append({k:v for k,v in row.items() if k not in ['candidate','index']})
    sidecars={f.name:sha(f.read_bytes()) for f in stage.iterdir() if f.suffix in ['.parquet','.log','.tsv'] or f.name.endswith('.pfm.gz')}
    result=dict(schema=CAPABILITY,image_path=str(reference),knob_tuple_json=cell['knob_tuple_json'],
                source_sha256=sha(reference_raw),reference_pixels_sha256=knobs['reference_pixels_sha256'],variant_index_uri=uri,variant_index_sha256=index_sha,sidecars=sidecars,
                implementation='zenmetrics@0a61830db89a768ab9462f166d1412e948e4efa9;display=standard_4k;exe-sha256='+pins['diagnostics'],cells=output)
    (stage/'result.json').write_text(json.dumps(result,sort_keys=True,allow_nan=False))
    # Preserve the raw port tables/logs as CAS objects alongside the joined rows.
    objects=[dict(path=name,sha256=digest) for name,digest in sidecars.items()]
    return b'ZFEVIFSTAGE1\n'+json.dumps(dict(root=str(stage),primary='result.json',objects=objects)).encode()
