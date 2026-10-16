"""Summarize completed review measurements without rerunning inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('workflow_log', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
summary = {'schema': 1, 'metric': 'pieapp', 'host': 'i265', 'cap_geometry': [3000, 3000]}
caps = {}
for engine in ('rten', 'ort-cuda'):
    path = a.directory / f'cap-{engine}.log'
    text = path.read_text()
    calls = re.findall(r'call=(\d+) seconds=([0-9.]+) score=([-+0-9.eE]+) bits=([0-9a-f]+)', text)
    if engine == 'ort-cuda':
        assert len(calls) == 2 and calls[0][3] == calls[1][3], (engine, calls)
        assert re.search(r'Exit status: 0\s*$', text)
    else:
        assert len(calls) == 1 and 'Command terminated by signal 15' in text

    caps[engine] = dict(load_seconds=float(re.search(r'load_seconds=([0-9.]+)',text)[1]),
                        calls=[dict(call=int(i),seconds=float(t),score=float(v),bits=b) for i,t,v,b in calls],
                        complete=(engine == 'ort-cuda'),
                        total_seconds=(float(re.search(r'total_seconds=([0-9.]+)',text)[1]) if engine == 'ort-cuda' else None),
                        interruption_note=(None if engine == 'ort-cuda' else 'Coordinator stopped second evaluation; first call completed; no rerun'),
                        maximum_resident_kib=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',text)[1]),
                        log_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
summary['historical_revision_2_caps'] = caps
summary['score_revision'] = 3
summary['export'] = json.loads((a.directory/'export.json').read_text())
summary['weight_identification'] = json.loads((a.directory/'weight-identify.json').read_text())
summary['reconstruction'] = json.loads((a.directory/'reconstruction.json').read_text())
summary['calibration_comparison'] = json.loads((a.directory/'konjnd-comparison.json').read_text())
head = json.loads((a.directory/'head.json').read_text())
summary['weight_sign'] = json.loads((a.directory/'weight-sign.json').read_text())
summary['head'] = dict(cases=len(head),negative_raw_weights=sum(r['raw_weight']<0 for r in head),
                      maximum_error_by_hypothesis={key:max(row['hypotheses'][key]['absolute_error'] for row in head) for key in head[0]['hypotheses']})
cli = json.loads((a.directory/'cli.json').read_text())
assert cli['pairs'] == 24
summary['demo_cli'] = dict(pairs=cli['pairs'],worst=cli['worst'],precision='Demo CUDA CLI printed decimal values; retained exactly as printed')
phases = ['cli','inventory','debug','release','fmt','clippy','nn-clippy','gpu-features','nn-features','rten-1415','rten-2207','rten-2406','cuda','cuda-cap','konjnd-0','konjnd-1','konjnd-2','konjnd-3','routing','cli-build','cli-smoke']
resources={}
for phase in phases:
    log=a.directory/'steps'/f'{phase}.log'
    assert (a.directory/'steps'/f'{phase}.exit').read_text().strip() == '0', phase
    receipt = re.search(r'run-heavy: done rc=(\d+) (\d+)s \| peak-RSS ([0-9.]+)GiB \| min-avail (\d+)MiB \| peak-load ([0-9.]+)',log.read_text())
    assert receipt and int(receipt[1]) == 0 and int(receipt[2]) < 1200, phase
    resources[phase]=dict(rc=int(receipt[1]),seconds=int(receipt[2]),peak_rss_gib=float(receipt[3]),min_available_mib=int(receipt[4]),peak_load=float(receipt[5]),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
parity={}
manifest=json.loads((a.directory/'shards'/'manifest.json').read_text())
assert manifest['total_pairs'] == 24
for phase,expected in [(f'rten-{name}',row['pairs']) for name,row in manifest['shards'].items()]+[('cuda',24)]:
    text=(a.directory/'steps'/f'{phase}.log').read_text()
    found=re.findall(r'pieapp: (\d+) pairs, worst \|d\| ([0-9.eE+-]+) = ([0-9.eE+-]+) JND',text)
    assert len(found)==1 and int(found[0][0])==expected,(phase,found,expected)
    parity[phase]=dict(pairs=int(found[0][0]),reported_worst_absolute=float(found[0][1]),reported_worst_jnd=float(found[0][2]))
summary['native_parity']=parity
path=a.directory/'cap-ort-cuda-v3.log'
text=path.read_text()
calls=re.findall(r'call=(\d+) seconds=([0-9.]+) score=([-+0-9.eE]+) bits=([0-9a-f]+)',text)
assert len(calls)==2 and calls[0][3]==calls[1][3]
assert re.search(r'Exit status: 0\s*$',text)
summary['revision_3_cuda_cap']=dict(calls=[dict(call=int(i),seconds=float(t),score=float(v),bits=b) for i,t,v,b in calls],maximum_resident_kib=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',text)[1]),log_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
summary['workflow'] = dict(steps=resources,orchestrator_log_sha256=hashlib.sha256(a.workflow_log.read_bytes()).hexdigest())
with a.output.open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps(summary,indent=2))
