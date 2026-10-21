#!/usr/bin/env python3
"""Resume original blocks in place; preserve all completed and interrupted data."""
import argparse
import csv
import datetime
import fcntl
import io
import json
from pathlib import Path
import shutil
import re
import statistics
import subprocess
import sys

from compare_canonical import sha, summarize, compare
from compare_canonical_budgeted import compare_samples


def _attempt(directory, number, mode):
    base = directory/f'block-{number:02d}'
    candidate = base
    index = 0
    while candidate.exists() or _log(candidate, mode).exists():
        index += 1
        candidate = directory/f'{base.name}-recovery-{index}'
    return candidate


def _log(attempt, mode):
    return Path(str(attempt)+('-wrapper.log' if mode == 'warm' else '.log'))


def _completed(directory, number):
    paths = sorted(directory.glob(f'block-{number:02d}*/result.json'))
    if len(paths) > 1:
        raise ValueError('multiple completed attempts; do not select a winner: '+str(paths))
    return paths[0] if paths else None


def _validate(block, case, number, mode, phase):
    roles = ['canonical','rust'] if mode == 'warm' else ['canonical','native']
    if mode == 'warm':
        if block['order'] != (roles if number % 2 == 0 else roles[::-1]):
            raise ValueError('original alternating order differs')
    elif block['case'] != case['id'] or block['phase'] != phase:
        raise ValueError('first-evaluation case or phase differs')
    if set(block['arms']) == set(roles):
        native = block['arms'][roles[1]]['score']
        canonical = block['arms']['canonical']
        if mode == 'warm':
            delta, passed = compare_samples(native,canonical.get('extra',{}).get('sample_scores',[canonical['score']]),case['absolute_bound'],case.get('relative_bound',0))
        else:
            delta, passed = compare(native,canonical['score'],case['absolute_bound'],case.get('relative_bound',0))
        if block.get('delta') != delta or block['passed'] is not passed:
            raise ValueError('retained numerical classification differs')
    elif not block.get('errors'):
        raise ValueError('missing arm without an explicit retained error')


def _summary(plan, case, blocks, digest, mode):
    if mode == 'first':
        return dict(metric=case['metric'],case=case['id'],width=case['width'],height=case['height'],threads=case['threads'],phase=plan['phase'],plan_sha256=digest,
            passed=all(block['passed'] is True for block in blocks),
            ratio=statistics.median(block['ratio'] for block in blocks) if all('ratio' in block for block in blocks) else None,
            blocks=blocks,qualification='Three first evaluations after construction, descriptive only; not a warm timing result or full startup measurement')
    return dict(metric=case['metric'],case=case['id'],width=case['width'],height=case['height'],threads=case['threads'],canonical=case['canonical']['label'],
        passed=False if any(block.get('errors') or block['passed'] is False for block in blocks) else None if any(block['passed'] is None for block in blocks) else True,
        absolute_bound=case['absolute_bound'],max_abs_delta=max((block['delta'] for block in blocks if 'delta' in block),default=None),
        ratio=summarize(blocks,case['matched_scope']),matched_scope=case['matched_scope'],scope=case['scope'],
        qualification='descriptive; no confidence interval or regression acceptance',rust_source=plan['rust_source'],plan_sha256=digest)


def _csv_strings(row):
    out = io.StringIO()
    writer = csv.DictWriter(out,fieldnames=list(row),delimiter='\t',lineterminator='\n')
    writer.writeheader()
    writer.writerow(row)
    return next(csv.DictReader(io.StringIO(out.getvalue()),delimiter='\t'))


def _append_csv(root, row):
    target = root/'summary.tsv'
    if target.exists() and target.stat().st_size:
        with target.open(newline='') as stream:
            reader = csv.DictReader(stream,delimiter='\t')
            if reader.fieldnames != list(row):
                raise ValueError('retained TSV header differs; preserve it')
            found = [old for old in reader if old['case'] == row['case']]
        if len(found) > 1 or (found and found[0] != _csv_strings(row)):
            raise ValueError('retained TSV row differs; preserve it')
        if found:
            return False
    with target.open('a',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(row),delimiter='\t',lineterminator='\n')
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    return True


def _helper(plan, mode, explicit=None, expected=None):
    name = 'compare_canonical_budgeted.py' if mode == 'warm' else 'compare_first_evaluation.py'
    if bool(explicit) != bool(expected):
        raise ValueError('explicit helper path and SHA256 are required together')
    if explicit:
        preferred = Path(explicit).resolve(strict=True)
        if preferred.name != name or sha(preferred) != expected:
            raise ValueError('explicit original helper identity differs')
        if str(preferred) in plan['bindings'] and plan['bindings'][str(preferred)] != expected:
            raise ValueError('explicit helper contradicts original plan binding')
        return preferred
    preferred = Path(__file__).with_name(name).resolve()
    if str(preferred) not in plan['bindings']:
        candidates = [Path(path) for path in plan['bindings'] if Path(path).name == name]
        if len(candidates) != 1:
            raise ValueError('original bound block helper is ambiguous: '+name)
        preferred = candidates[0]
    if sha(preferred) != plan['bindings'][str(preferred)]:
        raise ValueError('original block helper changed')
    return preferred


def _execute(plan_path, case, number, output, mode, samples, helper):
    arguments = ['--block-case',case['id'],'--block-number',str(number),'--samples',str(samples)] if mode == 'warm' else ['--case',case['id'],'--block',str(number)]
    command = ['flock',str(Path.home()/'tmp/lilith-heavy.lock'),'flock',str(Path.home()/'tmp/lilith-wsl-heavy.lock'),
        str(Path.home()/'work/claudehints/scripts/run-heavy'),'--mem','12G','--jobs',str(case['threads']),'--',
        sys.executable,'-B',str(helper),str(plan_path),str(output),*arguments]
    helper_digest = sha(helper)
    with _log(output,mode).open('x') as stream:
        result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    if sha(helper) != helper_digest:
        raise ValueError('original helper changed during recovered block')
    if result.returncode or not (output/'result.json').exists():
        raise RuntimeError('unfinished recovery block retained: '+str(output))


def resume(plan_path, root, mode='warm', blocks=3, samples=5, executor=None, emitter=print, original_helper=None, original_helper_sha256=None, metrics=None):
    if blocks != 3 or samples != 5:
        raise ValueError('this recovery command retains original three-block/five-warm-sample campaigns only')
    plan_path, root = Path(plan_path).resolve(strict=True), Path(root).absolute()
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    digest = sha(plan_path)
    requested = set(metrics.split(',')) if metrics else {case['metric'] for case in plan['cases']}
    if not requested <= {case['metric'] for case in plan['cases']}:
        raise ValueError('selected metric absent from frozen original plan')
    root.mkdir(exist_ok=True)
    frozen = root/'plan.json'
    if frozen.exists():
        if frozen.read_bytes() != plan_bytes:
            raise ValueError('original plan differs; do not replace it')
    else:
        frozen.write_bytes(plan_bytes)
    helper = _helper(plan,mode,original_helper,original_helper_sha256) if executor is None else None
    execute = executor or _execute
    with (root/'recovery-coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal = root/('recovery-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        journal.mkdir()
        if (root/'summary.tsv').exists():
            shutil.copyfile(root/'summary.tsv',journal/'original-summary.tsv')
        with (journal/'request.json').open('x') as stream:
            json.dump(dict(plan_sha256=digest,mode=mode,blocks=blocks,samples=samples,selected_metrics=sorted(requested),source_sha256=sha(__file__),helper=str(helper) if helper else 'injected validation executor',helper_sha256=sha(helper) if helper else None,helper_in_original_plan=str(helper) in plan['bindings'] if helper else False,qualification='Original completed blocks and failed outcomes retained; only missing block outcomes acquire new named attempts.'),stream,indent=2)
        rows = []
        for case in plan['cases']:
            if case['metric'] not in requested:
                continue
            directory = root/case['id']
            directory.mkdir(exist_ok=True)
            for retained in directory.glob('block-*'):
                match = re.match(r'block-(\d+)(?:$|[-.])',retained.name)
                if match and int(match[1]) >= blocks:
                    raise ValueError('confirmation or expanded campaign needs its original controller: '+str(directory))
            retained_summary = (directory/'summary.json').exists()
            acquired = []
            paths = []
            for number in range(blocks):
                path = _completed(directory,number)
                if path is None:
                    output = _attempt(directory,number,mode)
                    execute(frozen,case,number,output,mode,samples,helper)
                    path = output/'result.json'
                block = json.loads(path.read_text())
                _validate(block,case,number,mode,plan.get('phase'))
                acquired.append(block)
                paths.append(dict(path=str(path),sha256=sha(path)))
            row = _summary(plan,case,acquired,digest,mode)
            summary = directory/'summary.json'
            if summary.exists():
                if json.loads(summary.read_text()) != row:
                    raise ValueError('completed case summary differs; preserve it: '+str(summary))
            else:
                with summary.open('x') as stream:
                    json.dump(row,stream,indent=2)
                    stream.write('\n')
            appended = _append_csv(root,row) if mode == 'warm' else False
            with (journal/'cases.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(case=case['id'],retained_summary=retained_summary,blocks=paths))+'\n')
            if not retained_summary or appended:
                emitter(('CASE_DONE ' if mode == 'warm' else 'FIRST_CASE_DONE ')+json.dumps(row))
            rows.append(row)
            if frozen.read_bytes() != plan_bytes:
                raise ValueError('frozen plan changed during recovery')
        result = dict(expected=sum(case['metric'] in requested for case in plan['cases']),completed=len(rows),failed=sum(row['passed'] is False for row in rows),plan_sha256=digest,mode=mode)
        with (journal/'result.json').open('x') as stream:
            json.dump(result,stream,indent=2)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--mode',choices=['warm','first'],default='warm')
    parser.add_argument('--metrics')
    parser.add_argument('--original-helper',type=Path)
    parser.add_argument('--original-helper-sha256')
    args = parser.parse_args()
    print(json.dumps(resume(args.plan,args.output,args.mode,original_helper=args.original_helper,original_helper_sha256=args.original_helper_sha256,metrics=args.metrics)),flush=True)


if __name__ == '__main__':
    main()
