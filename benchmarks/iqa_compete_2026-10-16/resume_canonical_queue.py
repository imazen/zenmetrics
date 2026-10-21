#!/usr/bin/env python3
"""Resume frozen original comparison queues without repeating completed blocks."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from resume_canonical_campaign import resume
from compare_canonical_queue import incomplete


def options(job):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--metrics')
    parser.add_argument('--blocks',type=int,default=3)
    parser.add_argument('--samples',type=int,default=5)
    selected = parser.parse_args(job.get('options',[]))
    if selected.blocks != 3 or selected.samples != 5:
        raise ValueError('original confirmation or nonstandard budget needs its own controller')
    return selected


def pending(manifest):
    result = []
    for job in manifest['jobs']:
        plan = json.loads(Path(job['plan']).read_text())
        selected = options(job)
        for case in plan['cases']:
            if selected.metrics and case['metric'] not in set(selected.metrics.split(',')):
                continue
            directory = Path(job['output'])/case['id']
            if not (directory/'summary.json').exists() or any(len(list(directory.glob(f'block-{number:02d}*/result.json'))) != 1 for number in range(3)):
                result.append(str(directory))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--after-queue',type=Path)
    parser.add_argument('--flags',type=Path)
    parser.add_argument('--regression-log',type=Path)
    parser.add_argument('--original-helper',type=Path)
    parser.add_argument('--original-helper-sha256')
    args = parser.parse_args()
    if bool(args.flags) != bool(args.regression_log):
        parser.error('--flags and --regression-log are required together')
    manifest = json.loads(args.manifest.read_text())
    for job in manifest['jobs']:
        options(job)
    previous = None
    while True:
        missing = incomplete(manifest.get('prerequisites',[]))
        if args.after_queue:
            missing += pending(json.loads(args.after_queue.read_text()))
        if not missing:
            break
        if len(missing) != previous:
            print('WAIT_PREREQUISITE_CASES',len(missing),flush=True)
            previous = len(missing)
        time.sleep(30)
    def emit(line):
        print(line,flush=True)
        if args.flags:
            subprocess.run(['just','bench-iqa-canonical-flags',str(args.manifest),str(args.flags),str(args.regression_log)],check=True)
    results = []
    for job in manifest['jobs']:
        result = resume(job['plan'],job['output'],emitter=emit,original_helper=args.original_helper,original_helper_sha256=args.original_helper_sha256,metrics=options(job).metrics)
        results.append(result)
        print('RECOVERY_JOB_DONE',json.dumps(dict(job=job,result=result)),flush=True)
    print('RECOVERY_QUEUE_DONE',json.dumps(results),flush=True)
    raise SystemExit(1 if any(result['failed'] for result in results) else 0)


if __name__ == '__main__':
    main()
