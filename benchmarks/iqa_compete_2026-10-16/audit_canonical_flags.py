#!/usr/bin/env python3
"""Retain newly completed >1.08 comparisons and prepare fixed follow-ups.

Incomplete acquisition directories are reported, never interpreted as passes.
This is a read-only reader of the original cohorts; only its new audit directory
and the explicitly supplied local regression log are written.
"""
import argparse
import copy
import datetime
import json
from pathlib import Path
import statistics

from compare_canonical import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    p.add_argument('regression_log',type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=True)
    index=a.output/'index.json'
    state=json.loads(index.read_text()) if index.exists() else dict(flags={},followups=[],profiles=[])
    pending=[]
    for job in json.loads(a.manifest.read_text())['jobs']:
        root=Path(job['output'])
        plan_path=root/'plan.json'
        if not plan_path.exists():
            pending.append(str(root));continue
        plan=json.loads(plan_path.read_text());digest=sha(plan_path)
        for case in plan['cases']:
            path=root/case['id']/'summary.json'
            if not path.exists():
                pending.append(str(path));continue
            row=json.loads(path.read_text())
            if row['plan_sha256']!=digest or row['rust_source']!=plan['rust_source']:
                raise ValueError('completed summary identity differs')
            key=digest+':'+case['id']
            if row['ratio'] is None or row['ratio']<=1.08 or key in state['flags']:continue
            blocks=[json.loads(p.read_text()) for p in path.parent.glob('block-*/result.json')]
            ours=statistics.median(b['rust_ns'] for b in blocks)/1e6
            canonical=statistics.median(b['canonical_ns'] for b in blocks)/1e6
            stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
            with a.regression_log.open('a') as stream:
                stream.write(f'{stamp} | {case["metric"]} | {case["width"]}x{case["height"]} | {case["threads"]} | ours {ours:.17g} ms | canonical {canonical:.17g} ms | {row["ratio"]:.17g} | measured-here {path}; descriptive alternating-block median\n')
            directory=a.output/(case['id']+'-'+digest[:8]);directory.mkdir()
            followup=copy.deepcopy(plan);followup['cases']=[case]
            followup['preceding_plan_sha256']=digest
            followup['confirmation_stage']='Fixed 20-block follow-up; retain every preceding observation and resource flag'
            followup['bindings'][str(Path(__file__).resolve())]=sha(__file__)
            successor=directory/'followup-plan.json'
            with successor.open('x') as stream:json.dump(followup,stream,indent=2)
            config=json.loads(next(path.parent.glob('block-*/rust/config.json')).read_text())
            config.update(required_samples=10,sample_target_ns=200_000_000,chunk_budget_ms=850_000)
            profile_config=directory/'profile-config.json'
            with profile_config.open('x') as stream:json.dump(config,stream,indent=2)
            state['followups'].append(dict(plan=str(successor.resolve()),output=str((directory/'confirmation-run').resolve()),options=['--blocks','20','--samples','5']))
            state['profiles'].append(dict(binary=plan['binary'],config=str(profile_config.resolve()),output=str((directory/'profile').resolve()),threads=case['threads']))
            state['flags'][key]=dict(summary_sha256=sha(path),ratio=row['ratio'],source=plan['rust_source'],path=str(path))
            print(json.dumps(dict(flag=case['id'],ratio=row['ratio'],followup=str(successor))),flush=True)
    index.write_text(json.dumps(state,indent=2)+'\n')
    (a.output/'queue.json').write_text(json.dumps(dict(jobs=state['followups']),indent=2)+'\n')
    print(json.dumps(dict(completed_flags=len(state['flags']),pending_cases_or_cohorts=len(pending))),flush=True)


if __name__=='__main__':main()
