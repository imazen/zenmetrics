#!/usr/bin/env python3
"""Correctness smoke of native shims; these one-call probes are not benchmarks."""
import argparse,json,pathlib,subprocess,struct
from native_run import ADAPTERS
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('vmaf',type=P);p.add_argument('jxl',type=P);p.add_argument('rust',type=P);p.add_argument('pairs',type=P);p.add_argument('planes',type=P);p.add_argument('output',type=P);a=p.parse_args();a.output.mkdir(exist_ok=False)
pair=next(v for v in json.loads(a.pairs.read_text()) if v['size_name']=='cid22');planes=json.loads(a.planes.read_text());rows=[]
for engine,adapters in ADAPTERS.items():
 for metric,adapter,convention,bound in adapters:
  native=pair if convention=='rgb' else next(v for v in planes if v['size_name']=='cid22' and v['convention']==convention);base=a.pairs.parent if convention=='rgb' else a.planes.parent
  binary=a.vmaf if engine=='libvmaf' else a.jxl
  command=[str(binary),adapter,str(native['width']),str(native['height']),str(base/native['reference']),str(base/native['distorted']),'1']+(['score','1'] if engine=='libjxl' else ['1','400' if convention.endswith('-y') else '420'])
  result=subprocess.run(command,capture_output=True,text=True);(a.output/f'{engine}-{metric}.log').write_text(result.stdout+result.stderr)
  row=dict(engine=engine,metric=metric,rc=result.returncode,parity_abs_bound=bound)
  if result.returncode:rows.append(row);print(json.dumps(row),flush=True);continue
  native_score=float(next(s.split()[2] for s in result.stdout.splitlines() if s.startswith('SAMPLE ')))
  config=a.output/f'{metric}-rust.json';config.write_text(json.dumps(dict(pair,metric_name=metric,probe_only=True,reference=str(a.pairs.parent/pair['reference']),distorted=str(a.pairs.parent/pair['distorted']))))
  rust=subprocess.run([str(a.rust),str(config)],capture_output=True,text=True,check=True);(a.output/f'{engine}-{metric}-rust.log').write_text(rust.stdout+rust.stderr)
  probe=json.loads(next(s[len('IQA_PROBE '):] for s in rust.stdout.splitlines() if s.startswith('IQA_PROBE ')));score=struct.unpack('>d',bytes.fromhex(probe['scores'][0][1]))[0]
  row.update(score=native_score,ref_score=score,abs_delta=abs(score-native_score),within_bound=abs(score-native_score)<=bound);rows.append(row);print(json.dumps(row),flush=True)
(a.output/'receipt.json').write_text(json.dumps(dict(not_a_timing_panel=True,cases=rows),indent=2)+'\n')
assert all(r['rc']==0 for r in rows),'native adapter invocation failed; inspect retained logs'
