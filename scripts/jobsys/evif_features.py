"""Frozen Mlic feature transport over persisted renditions; no encoder or metric substitution."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from evif_batch import chunks, sha

CAPABILITY='evif-rev7-features-v1'
EXTRACTOR_SHA='ed4957417339779dd8ed25fe59a6166326ba9db86d5c1530b1e1c788c9d614d8'
FEATURE_ID='basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af'
IDS=list(range(228))+list(range(372,720))


def execute(job):
    cell=job['cell'];knobs=json.loads(cell['knob_tuple_json'])
    if cell['codec']!='evif:features' or job['kind']!={'kind':'score_file','metrics':[CAPABILITY]}:
        raise ValueError('unsupported frozen feature job')
    if set(knobs)!={'source_sha256','reference_pixels_sha256','width','height','evif_shas'}:
        raise ValueError('unknown feature controls')
    reference=Path(cell['image_path']);reference_raw=reference.read_bytes()
    if sha(reference_raw)!=knobs['source_sha256'] or sha(reference_raw) not in job['inputs']:
        raise ValueError('unbound feature reference')
    if not 0<knobs['width']*knobs['height']<1_000_000:
        raise ValueError('invalid feature geometry')
    selected=knobs['evif_shas']
    if not selected or len(selected)>128 or len(set(selected))!=len(selected):
        raise ValueError('invalid feature selection')
    binary=Path(os.environ['ZEN_EVIF_BIN_DIR'])/'extract_features_372col-rev7'
    if EXTRACTOR_SHA not in job['inputs'] or sha(binary.read_bytes())!=EXTRACTOR_SHA:
        raise ValueError('unbound frozen Mlic extractor')
    archive=Path(os.environ['ZEN_ARCHIVE_BLOBS_DIR']);stage_base=archive/'.staging';stage_base.mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='evif-features-',dir=stage_base));catalog=[];unique={}
    for index,digest in enumerate(selected):
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest) or digest not in job['inputs']:
            raise ValueError('unbound feature container')
        data=(archive/digest[:2]/digest[2:4]/digest).read_bytes()
        if sha(data)!=digest:raise ValueError('feature CAS bytes differ')
        parts=chunks(data);png_sha=sha(parts['PNGF']);candidate=stage/(png_sha+'.png')
        if not candidate.exists():candidate.write_bytes(parts['PNGF'])
        row_id=unique.setdefault(png_sha,len(unique))
        catalog.append(dict(row_id=row_id,evif_sha256=digest,png_sha256=png_sha,candidate=candidate,
                            jpeg_sha256=sha(parts['BITS'][4:])))
    pairs=stage/'pairs.tsv'
    with pairs.open('w') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['ref_path','dist_path','human_score','row_id'])
        for digest,row_id in unique.items():writer.writerow([reference,stage/(digest+'.png'),0,row_id])
    output=stage/'features.raw.csv';audit=stage/'features.audit.jsonl'
    env=os.environ.copy();env.update(ZENSIM_FORMULA_REV='7',ZENSIM_ROOT_FORM='sqrt',RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
    env.pop('ZENSIM_MAX_TIER',None)
    args=[str(binary),'--corpus','pairs-tsv','--path',str(pairs),'--out',str(output),'--restore-cuts','basic,peaks,v2',
          '--input-contract','legacy-rgb8','--era-label','rev7_rowlanes','--force-tier','native','--audit-jsonl',str(audit)]
    with (stage/'extract.log').open('wb') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest=json.loads(Path(str(output)+'.manifest.json').read_text())
    if manifest['feature_set_id']!=FEATURE_ID or str(manifest['formula_revision'])!='7':raise ValueError('frozen extractor manifest differs')
    with output.open() as f:reader=csv.reader(f);header=next(reader);rows=list(reader)
    row_index=header.index('row_id');rows.sort(key=lambda r:int(r[row_index]))
    if [int(r[row_index]) for r in rows]!=list(range(len(unique))):raise ValueError('frozen feature row identities differ')
    if len([h for h in header if h.startswith('f') and h[1:].isdigit()])!=1825:raise ValueError('frozen feature width differs')
    positions=[header.index('f'+str(i)) for i in IDS]
    if any(not math.isfinite(float(r[p])) for r in rows for p in positions):raise ValueError('nonfinite measured feature')
    normalized=stage/'features.csv'
    with normalized.open('w') as f:writer=csv.writer(f,lineterminator='\n');writer.writerow(header);writer.writerows(rows)
    audits=[json.loads(line) for line in audit.read_text().splitlines()];by_path={}
    for r in audits:
        if r['reference']!=str(reference) or r['reference_pixels_sha256']!=knobs['reference_pixels_sha256']:
            raise ValueError('frozen extractor reference pixels differ')
        if r['distorted'] in by_path and by_path[r['distorted']]['distorted_pixels_sha256']!=r['distorted_pixels_sha256']:
            raise ValueError('inconsistent frozen candidate pixels')
        by_path[r['distorted']]=r
    if len(audits)!=len(unique):raise ValueError('frozen audit coverage differs')
    for r in catalog:r['distorted_pixels_sha256']=by_path[str(r.pop('candidate'))]['distorted_pixels_sha256']
    sidecars={p.name:sha(p.read_bytes()) for p in stage.iterdir() if p.is_file() and not p.name.endswith('.png')}
    result=dict(schema=CAPABILITY,image_path=str(reference),source_sha256=sha(reference_raw),reference_pixels_sha256=knobs['reference_pixels_sha256'],
                knob_tuple_json=cell['knob_tuple_json'],extractor_sha256=EXTRACTOR_SHA,feature_set_id=FEATURE_ID,
                build_commit='b77883be18db5933a9db4b9e2670a2ba88a9729b',feature_width=1825,measured_feature_ids=IDS,
                features_csv_sha256=sidecars[normalized.name],sidecars=sidecars,cells=catalog)
    (stage/'result.json').write_text(json.dumps(result,sort_keys=True,allow_nan=False))
    return b'ZFEVIFSTAGE1\n'+json.dumps(dict(root=str(stage),primary='result.json',objects=[dict(path=k,sha256=v) for k,v in sidecars.items()])).encode()
