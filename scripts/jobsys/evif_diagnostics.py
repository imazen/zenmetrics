"""Later zenfleet ScoreFile pass: read persisted EVIF PNGs, never encode."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
from evif_batch import chunks, sha

CAPABILITY = 'evif-diagnostics-v1'


def execute(job, pins):
    cell=job['cell'];knobs=json.loads(cell['knob_tuple_json'])
    if cell['codec']!='evif:diagnostics' or job['kind']!={'kind':'score_file','metrics':[CAPABILITY]}:
        raise ValueError('unsupported EVIF diagnostic job')
    if set(knobs)!={'source_sha256','reference_pixels_sha256','width','height','evif_shas'}:
        raise ValueError('unknown diagnostic controls')
    reference=Path(cell['image_path']);raw=reference.read_bytes()
    if sha(raw)!=knobs['source_sha256'] or sha(raw) not in job['inputs']:
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
        png=stage/(str(index)+'.png');png.write_bytes(parts['PNGF'])
        catalog.append(dict(index=str(index),evif_sha256=digest,jpeg_sha256=sha(parts['BITS'][4:]),candidate=png,teacher=target[0],scores={}))
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
    output=[]
    for row in catalog:output.append({k:v for k,v in row.items() if k not in ['candidate','index']})
    result=dict(schema=CAPABILITY,image_path=str(reference),knob_tuple_json=cell['knob_tuple_json'],
                source_sha256=sha(raw),reference_pixels_sha256=knobs['reference_pixels_sha256'],
                implementation='zenmetrics@0a61830db89a768ab9462f166d1412e948e4efa9;display=standard_4k;exe-sha256='+pins['diagnostics'],cells=output)
    (stage/'result.json').write_text(json.dumps(result,sort_keys=True,allow_nan=False))
    # Preserve the raw port tables/logs as CAS objects alongside the joined rows.
    objects=[dict(path=f.name,sha256=sha(f.read_bytes())) for f in stage.iterdir() if f.suffix in ['.parquet','.log']]
    return b'ZFEVIFSTAGE1\n'+json.dumps(dict(root=str(stage),primary='result.json',objects=objects)).encode()
