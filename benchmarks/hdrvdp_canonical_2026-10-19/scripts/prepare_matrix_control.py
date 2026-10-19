"""Inject the 12 measured native spectral integrals into an Octave oracle copy.
This controls only the spectral mixing stage; no score correction is fitted.
"""
import csv
import hashlib
from pathlib import Path
import shutil
import sys

source,native,target=map(Path,sys.argv[1:]);assert not target.exists()
rows=[r for r in csv.DictReader(native.open(),delimiter='\t') if r['stage']=='mixing']
values={(int(r['row']),int(r['column'])):r['value'] for r in rows}
assert len(rows)==len(values)==12 and set(values)=={(r,c) for r in range(3) for c in range(4)}
shutil.copytree(source,target)
(target/'native_mixing.tsv').write_text(''.join('\t'.join(values[r,c] for c in range(4))+'\n' for r in range(3)))
(target/'native_mixing.source.sha256').write_text(hashlib.sha256(native.read_bytes()).hexdigest()+'\n')
p=target/'hdrvdp_visual_pathway.m';text=p.read_text();marker='% Color space conversion'
assert text.count(marker)==1
text=text.replace(marker,"M_img_lmsr = dlmread(fullfile(fileparts(mfilename('fullpath')), 'native_mixing.tsv'));\n\n"+marker)
p.write_text(text)
