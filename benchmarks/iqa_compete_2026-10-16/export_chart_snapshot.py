#!/usr/bin/env python3
"""Export observed durable attempts for charts; never promote failed/flagged rows."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

FIELDS = ('metric_name width height size_name threads timing_phase cold_repeat only_mode device impl '
          'impl_binary_sha reference_sha256 distorted_sha256 median_ns mean_ns p10_ns p90_ns n '
          'score score_bits ref_score simd_tier input_method notes round_means_ns').split()


def export(root, output, host):
    output.mkdir(parents=True, exist_ok=False)
    observations, sources = [], []
    for path in sorted(root.glob('**/report.json')):
        if not (path.parent/'identity.json').exists():
            continue
        raw = path.read_bytes()
        report = json.loads(raw)
        identity = json.loads((path.parent/'identity.json').read_bytes())
        source_id = hashlib.sha256(raw).hexdigest()
        sources.append(dict(id=source_id, report=str(path.relative_to(root)),
                            identity=report.get('identity'), frozen=identity.get('frozen', {}).get('manifest_sha256'),
                            policy=identity.get('policy'), complete=report.get('complete'), qualified=report.get('qualified')))
        for cell in report['cells']:
            config = cell['config']
            for attempt in cell.get('attempts', []):
                row = attempt.get('row') or {}
                value = dict(host=host, source=source_id, cell_id=config['durable_cell_id'],
                             attempt=attempt['attempt'], status=attempt['status'],
                             cell_status=cell['status'], reason=attempt.get('reason'),
                             config_sha256=attempt.get('config_sha256'), log_sha256=attempt.get('log_sha256'))
                value.update({k:row.get(k,config.get(k)) for k in FIELDS})
                samples = row.get('zenbench_samples', [])
                value['warm_rounds_clean'] = sum(s.get('gate_clean') is True for s in samples)
                value['warm_rounds_flagged'] = sum(s.get('gate_clean') is not True for s in samples)
                value['qualified_timing'] = attempt['status']=='ok' and (config['timing_phase']=='cold' or
                    (row.get('n',0)>=10 and len(samples)>=10 and all(s.get('gate_clean') is True for s in samples)))
                observations.append(value)
    def save(name, value):
        (output/name).write_text(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n')
    save('sources.json', sources)
    with (output/'observations.jsonl').open('x') as stream:
        for value in observations:
            stream.write(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n')
    save('snapshot.json',dict(schema=1,host=host,captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         sources=len(sources),attempt_rows=len(observations),qualified_attempt_rows=sum(r['qualified_timing'] for r in observations),
         files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--host',required=True)
    args=parser.parse_args()
    export(args.root,args.output,args.host)
