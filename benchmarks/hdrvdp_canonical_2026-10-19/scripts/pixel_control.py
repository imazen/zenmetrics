"""Exact packed RGB8 parity: FFmpeg PNG oracle, Octave imread, Pillow.
Optional native directory adds image::open(...).to_rgb8() from the score probe.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, __version__ as pillow_version

pairs, output = map(Path, sys.argv[1:3])
native = Path(sys.argv[3]) if len(sys.argv) == 4 else None
assert not output.exists(), 'retain every attempt'
output.mkdir(parents=True)
rows = list(csv.reader(pairs.open(), delimiter='\t'))[1:]
paths = sorted({Path(p) for row in rows for p in row[:2]})
assert len(rows) == 61 and len(paths) == 122
assert len({p.name for p in paths}) == len(paths)
listing = output/'images.txt'
listing.write_text(''.join(str(p)+'\n' for p in paths))
driver=Path(__file__).with_name('pixel_dump.m')
shutil.copy2(driver,output/'pixel_dump.m')
subprocess.run(['octave','--quiet',str(output/'pixel_dump.m'),str(listing),str(output/'octave')],check=True)
octave={r['name']:r for r in csv.DictReader((output/'octave/manifest.tsv').open(),delimiter='\t')}
native_shapes=None
if native:
    native_shapes={r['name']:r for r in csv.DictReader((native/'manifest.tsv').open(),delimiter='\t')}
    assert set(native_shapes)==set(octave)
ffmpeg=shutil.which('ffmpeg'); assert ffmpeg
summary=dict(n=0, exact_octave=0, exact_pillow=0, exact_native=0,
             unequal_octave_bytes=0, unequal_pillow_bytes=0, unequal_native_bytes=0,
             max_octave_code_delta=0, max_pillow_code_delta=0, max_native_code_delta=0,
             compared_bytes=0, failures=[], pillow=pillow_version,
             pairs_sha256=hashlib.sha256(pairs.read_bytes()).hexdigest(),
             script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             octave_driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
             ffmpeg_sha256=hashlib.sha256(Path(ffmpeg).read_bytes()).hexdigest())
summary['ffmpeg_version']=subprocess.check_output([ffmpeg,'-version'],text=True).splitlines()[0]
with (output/'pixels.tsv').open('w') as table:
    fields=['name','width','height','input_sha256','oracle_sha256','octave_sha256','pillow_sha256','native_sha256']
    writer=csv.DictWriter(table,fieldnames=fields,delimiter='\t');writer.writeheader()
    for p in paths:
        r=octave[p.name]; w,h=int(r['width']),int(r['height'])
        cmd=[ffmpeg,'-nostdin','-v','error','-threads','1','-i',str(p),'-frames:v','1',
             '-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
        oracle=subprocess.check_output(cmd)
        assert len(oracle)==w*h*3==int(r['bytes'])
        with Image.open(p) as im:
            assert im.mode=='RGB' and im.size==(w,h)
            pillow=im.tobytes()
        images=dict(octave=(output/'octave'/(p.name+'.rgb')).read_bytes(),pillow=pillow)
        if native:
            assert native_shapes[p.name]==r, 'native decoded geometry differs'
            images['native']=(native/(p.name+'.rgb')).read_bytes()
        record=dict(name=p.name,width=w,height=h,input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    oracle_sha256=hashlib.sha256(oracle).hexdigest(),native_sha256='unmeasured')
        for key, data in images.items():
            assert len(data)==len(oracle)
            unequal=int(np.count_nonzero(np.frombuffer(data,dtype=np.uint8)!=np.frombuffer(oracle,dtype=np.uint8)))
            maximum=int(np.max(np.abs(np.frombuffer(data,dtype=np.uint8).astype(np.int16)-np.frombuffer(oracle,dtype=np.uint8))))
            summary['exact_'+key]+=int(unequal==0)
            summary['unequal_'+key+'_bytes']+=unequal
            summary['max_'+key+'_code_delta']=max(summary['max_'+key+'_code_delta'],maximum)
            record[key+'_sha256']=hashlib.sha256(data).hexdigest()
            if unequal: summary['failures'].append(dict(name=p.name,decoder=key,unequal_bytes=unequal,max_delta=maximum))
        writer.writerow(record)
        summary['n']+=1;summary['compared_bytes']+=len(oracle)
summary['native_measured']=native is not None
(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
assert summary['n']==122 and not summary['failures'], 'decoded pixels disagree; inspect retained evidence'
