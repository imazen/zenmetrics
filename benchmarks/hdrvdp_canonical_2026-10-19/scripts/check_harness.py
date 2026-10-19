"""Exercise complete-panel refusal and mixed-attempt collection using retained data.
Usage: check_harness.py SCRATCH PUBLISHED CONFIG
"""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

scratch,pub,cfg=map(Path,sys.argv[1:]);scripts=Path(__file__).parent
baseline=scratch/'hdr2-retained.tsv'
args=[sys.executable,str(scripts/'compare.py'),str(pub),str(cfg),'HDR_VDP_2',str(baseline),'q']
expected=json.loads(subprocess.check_output(args,text=True));assert expected['n']==61
with tempfile.TemporaryDirectory(prefix='hdrvdp-harness-',dir=Path.home()/'tmp') as tmp:
    tmp=Path(tmp);text=baseline.read_text().splitlines()
    for label,rows in [('incomplete',text[:-1]),('duplicate',text[:-1]+[text[1]])]:
        bad=tmp/(label+'.tsv');bad.write_text('\n'.join(rows)+'\n')
        bad_args=args.copy();bad_args[-2]=str(bad)
        result=subprocess.run(bad_args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        assert result.returncode!=0,label+' was accepted'
    table=tmp/'trials.tsv'
    subprocess.run([sys.executable,str(scripts/'collect.py'),str(scratch),str(pub),str(cfg),str(table)],check=True)
    rows={r['trial']:r for r in csv.DictReader(table.open(),delimiter='\t')}
    assert len(rows)==len(list(scratch.glob('*.meta.json'))),'attempt lost'
    for key in ['median_raw','max_raw','median_jnd','max_jnd']:
        assert float(rows['hdr2-retained'][key])==expected[key],'attempt state leaked: '+key
    assert rows['hdr2-retained']['n']=='61'
    assert rows['hdr2-203-screen6']['n']=='6'
    assert rows['hdr2-203-screen6']['qualification']=='screening-only'
    assert rows['hdr3-200-10bit']['rc']=='1'
    assert rows['hdr3-200-10bit']['qualification']=='failed-or-incomplete'
print('PASS: incomplete/duplicate refusal; independent mixed-attempt rows; failed attempts retained')
