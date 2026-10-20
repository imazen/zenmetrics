#!/usr/bin/env python3
"""Run explicit comparison plans sequentially after completed prerequisites.

The queue has no inference worker of its own. Each comparison retains the
existing two-lock resource admission, asset checks and native numerical gates.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def incomplete(prerequisites):
    pending=[]
    for item in prerequisites:
        root=Path(item['results'])
        plan=json.loads((root/'plan.json').read_text())
        for case in plan['cases']:
            directory=root/case['id']
            if not (directory/'summary.json').exists() or len(list(directory.glob('block-*/result.json')))!=item['blocks']:
                pending.append(str(directory))
    return pending


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path)
    p.add_argument('--wait',action='store_true',help='Keep a visible controller waiting for the listed fixed prerequisites')
    a=p.parse_args();manifest=json.loads(a.manifest.read_text())
    for job in manifest['jobs']:
        if Path(job['output']).exists():
            raise ValueError('queue refuses an existing output: '+job['output'])
    previous=None
    while True:
        pending=incomplete(manifest.get('prerequisites',[]))
        if not pending:break
        if not a.wait:raise ValueError('unfinished prerequisites: '+str(len(pending)))
        if len(pending)!=previous:
            print(json.dumps(dict(pending_prerequisite_cases=len(pending))),flush=True)
            previous=len(pending)
        time.sleep(30)
    failures=[]
    runner=Path(__file__).with_name('compare_canonical_budgeted.py')
    for job in manifest['jobs']:
        command=[sys.executable,'-B',str(runner),job['plan'],job['output'],*job.get('options',[])]
        result=subprocess.run(command)
        print(json.dumps(dict(plan=job['plan'],output=job['output'],exit_code=result.returncode)),flush=True)
        if result.returncode:failures.append(job['output'])
    if failures:
        print(json.dumps(dict(incomplete_or_failed_comparisons=failures)),flush=True)
        raise SystemExit(1)


if __name__=='__main__':main()
