"""One zenfleet view job: one union render, frozen teacher, and score import.

Canonical target identity is the replayed B0 metric; receipts bind the actual
raw-reference adapter separately. Full-container parity is checked against the
unbatched path before declaration. Diagnostics can run after target generation.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

# Filled from the prebuilt, verified tools before baking this executor image.
BATCH_PINS = {'evif-batch':'6ec626d6c13ae45a68516a77676eb3da477322f9b5ca70adc637e09019ef7476','chroma-frozen-batch-teacher':'062d96f9fd881d82c279f1dfbbb2dee1783f968cec7316d757088bad0ef87758'}
CAPABILITY='evif-view-batch-v1'

def sha(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def chunks(data):
    if data[:12]!=b'\x8bEVIF\r\n\x1a\x00\x01\x00\x00':raise ValueError('unexpected EVIF framing')
    offset=12;result={}
    while offset<len(data):
        if offset+8>len(data):raise ValueError('truncated EVIF')
        tag=data[offset:offset+4].decode('ascii');n=struct.unpack_from('<I',data,offset+4)[0];offset+=8
        if tag in result or offset+n>len(data):raise ValueError('duplicate/truncated EVIF chunk')
        result[tag]=data[offset:offset+n];offset+=n
    if not {'HEAD','BITS','PNGF','DIGS','SEAL'}<=result.keys():raise ValueError('EVIF sample incomplete')
    return result

def execute(job,pins):
    cell=job['cell'];knobs=json.loads(cell['knob_tuple_json'])
    if cell['codec']!='evif:batch' or job['kind']!={'kind':'metric','metric':'evif-chroma-batch-v1'}:raise ValueError('unsupported EVIF batch job')
    if set(knobs)!={'grid_path','grid_sha256','reference_pixels_sha256','width','height','diagnostics'}:raise ValueError('unknown batch controls')
    if knobs['diagnostics'] not in ['none','all']:raise ValueError('unknown diagnostics mode')
    reference=Path(cell['image_path']);raw=reference.read_bytes();w,h=struct.unpack('>II',raw[16:24])
    if raw[:8]!=b'\x89PNG\r\n\x1a\n' or w*h>=1_000_000 or not w*h or (w,h)!=(knobs['width'],knobs['height']):raise ValueError('invalid prepared view')
    grid=Path(knobs['grid_path']).read_bytes()
    if sha(grid)!=knobs['grid_sha256'] or sha(raw) not in job['inputs'] or sha(grid) not in job['inputs']:raise ValueError('unbound view/grid bytes')
    entries=[json.loads(line) for line in grid.splitlines() if line.strip()]
    if not entries or len(entries)>128:raise ValueError('invalid rung count')
    bins=Path(os.environ['ZEN_EVIF_BIN_DIR'])
    for name,digest in BATCH_PINS.items():
        if sha((bins/name).read_bytes())!=digest:raise ValueError('batch binary pin mismatch: '+name)
    if knobs['diagnostics']=='all' and sha((bins/'zenmetrics-diagnostics').read_bytes())!=pins['diagnostics']:raise ValueError('diagnostic binary pin mismatch')
    stage_base=Path(os.environ.get('ZEN_ARCHIVE_BLOBS_DIR',os.environ['TMPDIR']))/'.staging'
    stage_base.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='evif-view-',dir=stage_base))
    def run(name,args,stdin=None):
        with (stage/(name+'.stderr')).open('wb') as stderr:
            p=subprocess.run(list(map(str,args)),input=stdin,stdout=subprocess.PIPE,stderr=stderr)
        if p.returncode:raise RuntimeError(f'{name} failed; retained staging: {stage}; '+(stage/(name+'.stderr')).read_text(errors='replace'))
        return p.stdout
    union=[]
    for row in entries:
        if row['family'] not in ['zenjpeg-expert','zenjpeg-color']:raise ValueError('unsupported union family')
        union.append(dict(family=row['family'],params=json.loads(row['params_json'])))
    paramfile=stage/'params.jsonl';paramfile.write_text(''.join(canonical(p)+'\n' for p in union))
    out=stage/'render';run('render',[bins/'evif-batch','render','--family','union','--params',paramfile,'--format','evif','--rendition','png','--emit-reference-rgb8','--out',out,reference])
    rows=list(csv.DictReader((out/'render.tsv').open(),delimiter='\t'))
    if len(rows)!=len(entries):raise ValueError('union render refused declared rungs')
    reference_rgb=out/(reference.stem+'.reference.rgb8')
    if sha(reference_rgb.read_bytes())!=knobs['reference_pixels_sha256']:raise ValueError('union reference differs from frozen RGB projection')
    header=dict(reference=str(reference),reference_rgb8=str(reference_rgb),width=w,height=h,reference_pixels_sha256=knobs['reference_pixels_sha256'])
    requests=[header];catalog=[]
    for row in rows:
        index=int(row['tag'][1:]);entry=entries[index]
        if json.loads(row['params'])!=json.loads(entry['params_json']):raise ValueError('union recipe differs')
        original=Path(row['path']);parts=chunks(original.read_bytes());candidate=stage/(row['tag']+'.png');candidate.write_bytes(parts['PNGF'])
        native=stage/(row['tag']+'.jpeg');native.write_bytes(parts['BITS'][4:])
        digests=parts['DIGS'];count=digests[0]
        if len(digests)!=1+9*count:raise ValueError('invalid pixel digests')
        digest={digests[1+9*i]:struct.unpack_from('<Q',digests,2+9*i)[0] for i in range(count)}
        if set(digest)!={0,1}:raise ValueError('both pixel digests required')
        requests.append(dict(distorted=str(candidate)))
        catalog.append(dict(tag=row['tag'],entry=entry,original=original,candidate=candidate,native=native,reference_digest=f'{digest[1]:016x}',distorted_digest=f'{digest[0]:016x}'))
    result=run('teacher',[bins/'chroma-frozen-batch-teacher'],''.join(json.dumps(r)+'\n' for r in requests).encode())
    (stage/'teacher.jsonl').write_bytes(result)
    scores=[json.loads(line) for line in result.splitlines()];assert len(scores)==len(catalog)
    for item,label in zip(catalog,scores):
        if label['reference']!=str(reference) or label['distorted']!=str(item['candidate']) or label['reference_pixels_sha256']!=knobs['reference_pixels_sha256']:raise ValueError('teacher pair identity differs')
        if struct.pack('>d',label['value']).hex()!=label['value_bits']:raise ValueError('teacher float transport differs')
        item['teacher']=label
        item['scores']=[dict(metric='ssim2',variant='safesyn-teacher',implementation=label['implementation']+';source-helper-sha256='+pins['teacher'],value=label['value'],better='higher')]
    if knobs['diagnostics']=='all':
        import pyarrow.parquet as pq
        pairs=stage/'diagnostic-pairs.tsv'
        with pairs.open('w') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['ref_path','dist_path','image_path','codec','q','knob_tuple_json'])
            for item in catalog:writer.writerow([reference,item['candidate'],item['tag'],item['entry']['family'],95,item['entry']['params_json']])
        for metric in ['butteraugli','cvvdp']:
            output=stage/(metric+'.parquet')
            run(metric,[bins/'zenmetrics-diagnostics','score-pairs','--metric',metric,'--gpu-runtime','cpu','--group-by-ref','--pairs-tsv',pairs,'--out-parquet',output,*(['--display-model','standard_4k'] if metric=='cvvdp' else [])])
            actual={r['image_path']:r for r in pq.read_table(output).to_pylist()};assert len(actual)==len(catalog)
            for item in catalog:
                row=actual[item['tag']];columns=['butteraugli_max','butteraugli_pnorm3'] if metric=='butteraugli' else [k for k in row if k.startswith('cvvdp_cpu_imazen_v')]
                assert columns
                for column in columns:item['scores'].append(dict(metric=column,variant='diagnostic',implementation='zenmetrics@0a61830db89a768ab9462f166d1412e948e4efa9;display=standard_4k;exe-sha256='+pins['diagnostics'],value=row[column],better='lower' if metric=='butteraugli' else 'higher'))
    imports=[]
    for item in catalog:
        envelope=stage/(item['tag']+'.scores.json');envelope.write_text(json.dumps({k:item[k] for k in ['reference_digest','distorted_digest','scores']}))
        item['scored']=stage/(item['tag']+'.evif');imports.append(dict(input=str(item['original']),scores=str(envelope),output=str(item['scored'])))
    imports_path=stage/'imports.json';imports_path.write_text(json.dumps(imports));run('import',[bins/'evif-batch','score-import-many',imports_path])
    objects=[];output=[]
    for item in catalog:
        evif_data=item['scored'].read_bytes();native_data=item['native'].read_bytes();parts=chunks(evif_data)
        if parts['BITS'][4:]!=native_data:raise ValueError('import changed bitstream')
        evif_sha=sha(evif_data);native_sha=sha(native_data)
        objects.extend([dict(path=item['scored'].name,sha256=evif_sha),dict(path=item['native'].name,sha256=native_sha)])
        entry=item['entry'];label=item['teacher']
        output.append(dict(family=entry['label'],rung=entry['rung'],params_json=entry['params_json'],evif_sha256=evif_sha,jpeg_sha256=native_sha,teacher_ssim2=label['value'],teacher_bits=label['value_bits'],reference_pixels_sha256=label['reference_pixels_sha256'],distorted_pixels_sha256=label['distorted_pixels_sha256']))
    manifest=dict(schema='evif-view-batch-v1',image_path=str(reference),source_sha256=sha(raw),reference_pixels_sha256=knobs['reference_pixels_sha256'],knob_tuple_json=cell['knob_tuple_json'],cells=output,runtime=dict(batch_binaries=BATCH_PINS,canonical_teacher_helper=pins['teacher'],grid_sha256=sha(grid),diagnostics=knobs['diagnostics']))
    (stage/'result.json').write_text(json.dumps(manifest,sort_keys=True,allow_nan=False))
    return b'ZFEVIFSTAGE1\n'+json.dumps(dict(root=str(stage),primary='result.json',objects=objects)).encode()
