#!/usr/bin/env python3
"""Freeze scalar-only priority queues while retaining original evidence and budgets."""
import argparse
from collections import defaultdict
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import shlex
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def frozen_module(bundle):
    # Each retained attempt is validated by its own original controller source.
    spec = importlib.util.spec_from_file_location('frozen_durable', Path(bundle) / 'durable.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(bundle))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def processes():
    rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,args='], text=True)
    result = []
    for line in rows.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3:
            try:
                argv = shlex.split(parts[2])
            except ValueError:
                argv = parts[2].split()
            result.append((int(parts[0]), int(parts[1]), argv))
    return result


def boundary(config, receipt):
    """Caller owns the shared lock: no worker may still be measuring."""
    sources = config['sources'] + config.get('nn', [])
    outputs = {str(Path(s['output']).resolve()) for s in sources}
    snapshot = processes()
    active = [r for r in snapshot if '_watchdog' in r[2] and
              any(any(arg.startswith(o + '/') for arg in r[2]) for o in outputs)]
    if active:
        raise ValueError('watchdog still exists despite shared lock; preserve worker: ' + repr(active))
    stopped = []
    for service in config['stop_services']:
        if sys.platform == 'darwin':
            subprocess.run(['launchctl', 'disable', service], check=True)
            before = subprocess.run(['launchctl', 'print', service], text=True, capture_output=True)
            if before.returncode == 0:
                subprocess.run(['launchctl', 'bootout', service], check=True)
        else:
            before = subprocess.run(['systemctl', '--user', 'show', service], text=True, capture_output=True)
            if 'LoadState=not-found' not in before.stdout:
                if before.returncode:
                    raise ValueError('cannot resolve service state: ' + service)
                subprocess.run(['systemctl', '--user', 'stop', service], check=True)
        stopped.append(dict(service=service, before=before.stdout, before_rc=before.returncode))
    # LaunchAgent removal can leave an already-created durable child. Terminate
    # only exact output-bound controllers, after proving no watchdog can be live.
    owned = []
    for pid, parent, argv in processes():
        if len(argv) >= 5 and Path(argv[1]).name == 'durable.py' and argv[2] == 'run' and argv[4] in outputs:
            os.kill(pid, signal.SIGTERM)
            owned.append(dict(pid=pid, argv=argv))
    deadline = time.monotonic() + 20
    while True:
        live = [r for r in processes() if any(r[0] == p['pid'] and r[2] == p['argv'] for p in owned)]
        if not live:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('idle controller did not exit; do not kill a worker: ' + repr(live))
        time.sleep(.2)
    for source in sources:
        lock = Path(source['output']) / 'coordinator.lock'
        if lock.exists():
            with lock.open('rb') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(receipt, dict(checked_utc=now(), shared_lock=config['lock'], no_owned_watchdogs=True,
                        stopped_services=stopped, stopped_idle_controllers=owned))


def retained(source):
    """Read only; never recover attempts, rewrite reports, or reset failure budgets."""
    bundle, output = Path(source['bundle']), Path(source['output'])
    module = frozen_module(bundle)
    _, configs, frozen = module.manifest(bundle / 'manifest.json')
    if source.get('manifest_sha256', frozen['manifest_sha256']) != frozen['manifest_sha256']:
        raise ValueError('parent phase manifest differs')
    if not output.exists():
        return module, frozen, configs, {}
    saved = load(output / 'identity.json')
    if source.get('max_attempts', 2) not in (1, 2) or saved['policy']['max_attempts'] != source.get('max_attempts', 2):
        raise ValueError('unexpected retained attempt budget')
    if saved['frozen'] != frozen:
        raise ValueError('retained identity differs: ' + str(output))
    identity = module.digest(module.encoded(saved))
    evidence = {}
    for config in configs:
        attempts = sorted((output / 'cells' / config['durable_cell_id']).glob('attempt-*'))
        records = []
        for number, path in enumerate(attempts, 1):
            if path.name != f'attempt-{number:04d}' or not (path / 'result.json').exists():
                raise ValueError('nonfinal retained attempt; preserve for recovery: ' + str(path))
            intent = load(path / 'intent.json')
            record = module.read_result(path / 'result.json')
            if (load(path / 'config.json') != config or
                    any(record.get(k) != v for k, v in dict(identity=identity, cell_id=config['durable_cell_id'], attempt=number).items()) or
                    any(intent.get(k) != record.get(k) for k in ('identity', 'cell_id', 'attempt'))):
                raise ValueError('retained attempt identity mismatch: ' + str(path))
            for key, name in [('intent_sha256', 'intent.json'), ('config_sha256', 'config.json'), ('log_sha256', 'worker.log')]:
                expected = sha(path / name) if (path / name).exists() else None
                if record.get(key) != expected:
                    raise ValueError('retained evidence hash differs: ' + str(path))
            if record.get('cleanup_verified') is False:
                raise ValueError('retained cleanup unverified: ' + str(path))
            if record['status'] in ('ok', 'unsupported'):
                if module.validate_row(path / 'worker.log', config) != record['row']:
                    raise ValueError('retained row differs: ' + str(path))
            records.append(dict(path=str(path / 'result.json'), sha256=sha(path / 'result.json'), record=record))
        evidence[config['durable_cell_id']] = records
    return module, frozen, configs, evidence


def decision(records, cap=2):
    def usable(record):
        if record['status'] == 'unsupported':
            return True
        row = record.get('row') or {}
        if record['status'] != 'ok':
            return False
        return row.get('timing_phase') != 'warm' or (row.get('n', 0) >= 10 and
            len(row.get('zenbench_samples', [])) >= 10 and all(
            sample.get('gate_clean') is True for sample in row['zenbench_samples']))
    if any(usable(r['record']) for r in records):
        return 'retained', 0
    if len(records) >= cap:
        return 'exhausted', 0
    return 'pending', cap - len(records)


def clean_warm(row):
    return (row.get('status') == 'ok' and row.get('width') == row.get('height') == 512
            and row.get('threads') == 8 and row.get('mode') == 'score' and row.get('timing_phase') == 'warm'
            and row.get('n', 0) >= 10 and len(row.get('zenbench_samples', [])) >= 10
            and all(s.get('gate_clean') is True for s in row['zenbench_samples']))


def freeze(source, target, cells, lineage):
    # Existing clone/freeze preserves original binary, runners, inputs and policy.
    from prepare_coverage import freeze as existing_freeze
    phase = existing_freeze(Path(source), target, cells)
    manifest = load(target / 'manifest.json')
    hashes = load(target / 'SHA256SUMS.json')
    if (target / 'scalar-lineage.json').exists():
        parent_hash = sha(target / 'scalar-lineage.json')
        parent_name = 'scalar-parent-lineage-' + parent_hash + '.json'
        (target / 'scalar-lineage.json').rename(target / parent_name)
        manifest['input_sha256'].pop('scalar-lineage.json')
        manifest['input_sha256'][parent_name] = parent_hash
        hashes.pop('scalar-lineage.json', None)
        hashes[parent_name] = parent_hash
        lineage = dict(lineage, parent_lineage=dict(path=parent_name, sha256=parent_hash))
    write(target / 'scalar-lineage.json', lineage)
    manifest['input_sha256']['scalar-lineage.json'] = sha(target / 'scalar-lineage.json')
    manifest['metadata'].update(scalar_only=True, scalar_priority_source_sha256=sha(Path(__file__)),
                                scalar_lineage_sha256=sha(target / 'scalar-lineage.json'))
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    hashes.update({'manifest.json': sha(target / 'manifest.json'), 'scalar-lineage.json': sha(target / 'scalar-lineage.json')})
    (target / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2) + '\n')
    phase['manifest_sha256'] = hashes['manifest.json']
    return phase


def remeasure(config, root, outer_fd):
    source = Path(config['ranking_corrected_bundle'])
    template = next(c for c in load(source / 'manifest.json')['cells'] if c['only_mode'] == 'score' and
                    c['timing_phase'] == 'warm' and c['threads'] == 8 and c['width'] == c['height'] == 512)
    cells = [dict(template, metric_name=name) for name in ['psnr', 'ssim']]
    bundle, output = root / 'ranking-bundle', root / 'ranking-results'
    freeze(source, bundle, cells, dict(reason='Corrected protected replacement for two flagged ranking rows; originals retained'))
    time.sleep(5)  # Recorded quiet settle outside all measurement clocks.
    command = [sys.executable, str(bundle / 'durable.py'), 'run', str(bundle / 'manifest.json'), str(output),
               '--lock', str(root / 'ranking-inner.lock'), '--run-heavy', str(bundle / 'tools/run-heavy'),
               '--memory', '12G', '--jobs', '8', '--timeout-seconds', '1100', '--max-attempts', '2']
    with (root / 'ranking-control.log').open('xb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, pass_fds=(outer_fd,))
    report = load(output / 'report.json')
    if not report['complete'] or len(report['warm']) != 2 or not all(clean_warm(r) for r in report['warm']):
        raise ValueError('ranking controls not clean; preserve original gates and all attempts')
    write(root / 'ranking-control.json', dict(command=command, protected_lock=config['lock'], settle_seconds=5,
          report_sha256=sha(output / 'report.json'), binary_sha256=load(bundle / 'manifest.json')['binary_sha256']))
    return report['warm']


def rank(config, replacement):
    rows = {}
    if config.get('ranking_provisional'):
        provisional = Path(config['ranking_provisional'])
        if sha(provisional) != config['ranking_provisional_sha256']:
            raise ValueError('ranking source changed')
        value = load(provisional)
        # Full clean observed ranking is required, not just a truncated top ten.
        candidates = value.get('ranking', value.get('rows', value.get('top10', [])))
        for row in candidates:
            if row.get('flagged_rounds') == 0 and row.get('rounds', 0) >= 10:
                rows[row['metric']] = dict(row)
    else:
        for source in config['ranking_sources']:
            _, _, _, evidence = retained(source)
            for records in evidence.values():
                for record in records:
                    row = record['record'].get('row') or {}
                    if clean_warm(row):
                        rows[row['metric_name']] = dict(metric=row['metric_name'], median_ns=row['median_ns'],
                            rounds=row['n'], flagged_rounds=0, source=record['path'], source_sha256=record['sha256'])
    for row in replacement:
        if not clean_warm(row):
            raise ValueError('flagged ranking replacement')
        rows[row['metric_name']] = dict(metric=row['metric_name'], median_ns=row['median_ns'], rounds=row['n'],
                                        flagged_rounds=0, binary=row['impl_binary_sha'], replacement=True)
    if len(rows) < 10:
        raise ValueError('fewer than ten clean scalar ranking candidates')
    return sorted(rows.values(), key=lambda row: (row['median_ns'], row['metric']))


def requested_family(name):
    return (name in ('haarpsi', 'haarpsi-y', 'ssim2', 'dssim', 'psnrhvs-daala') or name.startswith('fsim') or
            name.startswith(('butteraugli', 'cvvdp')) or name == 'vmaf' or name.startswith('vmaf-'))


QUEUE_NAMES = ['fastest', 'dvifmish', 'requested-first', 'requested-family', 'remaining-scalar', 'slow-tail']


def queue_tier(name, top):
    if name in ('hdrvdp', 'hdrvdp3', 'mdctpsnr'):
        return 5
    if name.startswith('dvifmish-'):
        return 1
    if name in top:
        return 0
    if name in ('vmaf-neg', 'vmaf-v1', 'psnrhvs-daala') or name.startswith('fsim'):
        return 2
    return 3 if requested_family(name) else 4


def priority_order(name):
    return {'vmaf-neg': 0, 'vmaf-v1': 1, 'psnrhvs-daala': 2}.get(name, 3 if name.startswith('fsim') else 4)


def coverage_key(cell):
    return tuple(cell[k] for k in ('metric_name', 'width', 'height', 'size_name', 'threads', 'timing_phase'))


def disabled_geometry(cell, config):
    """Match dimensions in either orientation; labels do not control admission."""
    return bool(config.get("disabled_geometries")) and sorted((cell["width"], cell["height"])) in [sorted(g) for g in config.get("disabled_geometries", [])]


def deadline_selection(entries, qualified=()):
    """One smallest eligible resolution, one existing cold + warm, eight workers."""
    qualified = set(qualified)
    by_method = defaultdict(list)
    for entry in entries:
        c = entry['original_config']
        if entry['status'] == 'pending' and entry.get('queue') not in ('slow-tail', 'remaining-scalar') and c['threads'] == 8 and c['only_mode'] == 'score':
            by_method[c['metric_name']].append(entry)
    selected = set()
    choices = []
    for method, candidates in sorted(by_method.items()):
        geometries = sorted({(e['original_config']['width'], e['original_config']['height'], e['original_config']['size_name'])
                             for e in candidates}, key=lambda g:(g[0]*g[1], g[2]))
        geometries = [g for g in geometries if any((method, g[0], g[1], g[2], 8, phase) not in qualified for phase in ('cold', 'warm'))]
        if not geometries:
            continue
        geometry = geometries[0]
        group = [e for e in candidates if (e['original_config']['width'], e['original_config']['height'], e['original_config']['size_name']) == geometry]
        picked = []
        for phase in ('cold', 'warm'):
            if (method, geometry[0], geometry[1], geometry[2], 8, phase) in qualified:
                continue
            matching = sorted([e for e in group if e['original_config']['timing_phase'] == phase],
                              key=lambda e:e['original_config']['cold_repeat'])
            if matching:
                chosen = matching[0]
                selected.add((chosen['source'], chosen['original_cell_id']))
                picked.append(chosen['original_cell_id'])
        choices.append(dict(metric=method, geometry=geometry, selected_original_cell_ids=picked))
    return selected, choices


def prepare(config, root, ranking):
    if config.get('parent_sequence') and sha(config['parent_sequence']) != config['parent_sequence_sha256']:
        raise ValueError('immutable parent sequence differs')
    top = [row['metric'] for row in ranking[:10]]
    groups = defaultdict(list)
    coverage = []
    qualified = set()
    inventory_names = set()
    for source in config.get('coverage_sources', []):
        _, old_frozen, old_configs, old_evidence = retained(source)
        inventory_names.update(c['metric_name'] for c in old_configs)
        for cell in old_configs:
            if cell['only_mode'] == 'score' and any(r['record']['status'] == 'ok' and decision([r])[0] == 'retained' for r in old_evidence.get(cell['durable_cell_id'], [])):
                qualified.add(coverage_key(cell))
    for source_index, source in enumerate(config['sources']):
        _, frozen, configs, evidence = retained(source)
        original = frozen['manifest']
        inventory_names.update(c['metric_name'] for c in configs)
        for raw, cell in zip(original['cells'], configs):
            records = evidence.get(cell['durable_cell_id'], [])
            status, remaining = decision(records, source.get('max_attempts', 2))
            if raw['only_mode'] == 'score' and any(r['record']['status'] == 'ok' and decision([r])[0] == 'retained' for r in records):
                qualified.add(coverage_key(raw))
            entry = dict(source=source_index, original_cell_id=cell['durable_cell_id'], original_config=raw,
                         evidence=[{k: r[k] for k in ('path', 'sha256')} for r in records], status=status,
                         remaining_attempts=remaining)
            if raw['only_mode'] != 'score':
                entry['status'] = 'retained-map' if records else 'maps-not-scheduled'
            elif remaining and disabled_geometry(raw, config):
                entry['status'] = 'disabled-by-user'
            elif remaining:
                name = raw['metric_name']
                tier = queue_tier(name, top)
                entry['queue'] = QUEUE_NAMES[tier]
                groups[(tier, source_index, remaining)].append(entry)
            coverage.append(entry)
    selection, choices = deadline_selection(coverage, qualified) if config.get('deadline_unix') else (set(), [])
    expanded = defaultdict(list)
    for (tier, index, remaining), entries in groups.items():
        for entry in entries:
            primary = (entry['source'], entry['original_cell_id']) in selection
            c = entry['original_config']
            method_tag = (f'{priority_order(c["metric_name"]):02d}-' + c['metric_name'] + '-' + c['timing_phase'] + '-' + str(c['cold_repeat'])) if primary else ''
            key = (0 if primary else 1, tier, method_tag, index, remaining)
            expanded[key].append(entry)
    for entry in coverage:
        entry['deadline_selected'] = (entry['source'], entry['original_cell_id']) in selection
    phases, deferred = [], []
    for (secondary, tier, method, index, remaining), entries in sorted(expanded.items()):
        entries.sort(key=lambda e: (e['original_config']['width'] * e['original_config']['height'],
                     priority_order(e['original_config']['metric_name']), e['original_config']['metric_name'], e['original_config']['threads'],
                     e['original_config']['timing_phase'], e['original_config']['cold_repeat']))
        name = ('deadline-' + method + '-' if not secondary else '') + QUEUE_NAMES[tier] + f'-source{index}-attempts{remaining}'
        source = config['sources'][index]
        execution_bundle = Path(source.get('execution_bundle', source['bundle']))
        target_manifest = load(execution_bundle / 'manifest.json')
        original_manifest = load(Path(source['bundle']) / 'manifest.json')
        for entry in entries:
            raw = entry['original_config']
            if raw['threads'] not in (1, 8) or (raw['timing_phase'] == 'cold' and raw['cold_repeat'] not in (0, 1, 2)):
                raise ValueError('unexpected frozen methodology')
            for role in ('reference', 'distorted'):
                if target_manifest['input_sha256'][raw[role]] != original_manifest['input_sha256'][raw[role]]:
                    raise ValueError('replacement worker input differs')
        phase = freeze(execution_bundle, root / name,
                       [e['original_config'] for e in entries], dict(source=config['sources'][index], cells=entries))
        phase.update(name=name, output=str(Path(config['results']) / name), max_attempts=remaining)
        (deferred if config.get('deadline_unix') and secondary else phases).append(phase)
    for original in config.get('nn', []):
        if not config.get('disabled_geometries') and not original.get('execution_bundle') and not original.get('gpu_score_repeatability_policy'):
            phases.append(dict(original, manifest_sha256=sha(Path(original['bundle']) / 'manifest.json'), max_attempts=original.get('max_attempts', 2)))
            continue
        _, frozen, configs, evidence = retained(original)
        pending = defaultdict(list)
        for raw, cell in zip(frozen['manifest']['cells'], configs):
            records = evidence.get(cell['durable_cell_id'], [])
            state, remaining = decision(records, original.get('max_attempts', 2))
            entry = dict(source=original, original_cell_id=cell['durable_cell_id'], original_config=raw,
                         status=state, remaining_attempts=remaining,
                         evidence=[{k:r[k] for k in ('path', 'sha256')} for r in records])
            if remaining:
                if raw['only_mode'] != 'score':
                    entry['status'] = 'maps-not-scheduled'
                elif disabled_geometry(raw, config):
                    entry['status'] = 'disabled-by-user'
                else:
                    pending[remaining].append(entry)
            coverage.append(entry)
        for remaining, entries in sorted(pending.items()):
            entries.sort(key=lambda e:(e['original_config']['width']*e['original_config']['height'], e['original_config']['metric_name'], e['original_config']['threads'], e['original_config']['timing_phase'], e['original_config']['cold_repeat']))
            name = original['name'] + '-attempts' + str(remaining)
            execution_bundle = Path(original.get('execution_bundle', original['bundle']))
            target_manifest = load(execution_bundle / 'manifest.json') if original.get('execution_bundle') else frozen['manifest']
            execution = []
            for entry in entries:
                raw = dict(entry['original_config'])
                for role in ('reference', 'distorted'):
                    if original.get('execution_bundle') and target_manifest['input_sha256'][raw[role]] != frozen['manifest']['input_sha256'][raw[role]]:
                        raise ValueError('replacement NN worker input differs')
                if original.get('gpu_score_repeatability_policy') and raw['timing_phase'] == 'warm':
                    raw['gpu_score_repeatability_policy'] = original['gpu_score_repeatability_policy']
                    from durable import check_score_policy
                    check_score_policy(raw)
                execution.append(raw)
            phase = freeze(execution_bundle, root/name, execution, dict(source=original, cells=entries))
            phase.update(name=name, output=str(Path(config['results'])/name), max_attempts=remaining)
            phases.append(phase)
    if not phases and not deferred and not config.get('deadline_unix') and not config.get('allow_empty'):
        raise ValueError('no remaining work')
    write(root / 'retained-coverage.json', coverage)
    write(root / 'ranking-final.json', dict(criterion='clean 512 scalar eight-worker warm median; ten rounds minimum',
            top10=ranking[:10], ranking=ranking, requested_families=sorted(name for name in inventory_names if requested_family(name)), quality_claim='requested families; no formal quality Pareto claim'))
    plan = dict(schema=1, lock=config['lock'], phases=phases, source_sha256=sha(Path(__file__)),
                disabled_geometries=config.get('disabled_geometries', []), window_end_unix=config.get('window_end_unix') if config.get('stop_at_deadline') is True else None,
                stop_at_deadline=config.get('stop_at_deadline') is True, results_due_unix=config.get('results_due_unix', config.get('window_end_unix')), requested_start_unix=config.get('requested_start_unix'), ranking_sha256=sha(root / 'ranking-final.json'), coverage_sha256=sha(root / 'retained-coverage.json'),
                policy='scalar only; priority fastest ten plus all dvifmish and named families; then remaining scalar; then original NN')
    import shutil
    coordinator = root / 'coordinator'
    coordinator.mkdir()
    for name in ['scalar_priority.py', 'mac_service.py', 'durable.py', 'admission_recovery.py']:
        shutil.copy2(HERE / name, coordinator / name)
    plan['scheduler_files'] = {str(p.relative_to(root)): sha(p) for p in coordinator.iterdir()}
    if config.get('parent_sequence'):
        plan['parent_sequence'] = dict(path=config['parent_sequence'], sha256=config['parent_sequence_sha256'])
    if config.get('deadline_unix'):
        resume = dict(plan, phases=phases + deferred)
        write(root / 'resume-sequence.json', resume)
        plan.update(deadline_unix=config['deadline_unix'], requested_start_unix=config['requested_start_unix'],
                    deadline_choices=choices, deferred_phases=deferred, qualified_retained_keys=sorted(qualified),
                    priority_methods=sorted(name for name in inventory_names if name in top or name.startswith('dvifmish-') or requested_family(name)),
                    resume_plan_sha256=sha(root / 'resume-sequence.json'))
    write(root / 'sequence.json', plan)
    return plan


def stop_endpoint(plan):
    # A due time is reporting metadata, never implicit authorization to stop.
    return (plan.get('window_end_unix') or plan.get('deadline_unix')) if plan.get('stop_at_deadline') is True else None


def execute(plan_path, expected):
    if sha(plan_path) != expected:
        raise ValueError('sequence changed')
    plan = load(plan_path)
    for name, expected_hash in plan['scheduler_files'].items():
        if sha(plan_path.parent / name) != expected_hash:
            raise ValueError('frozen scheduler changed')
    if sha(Path(__file__)) != plan['source_sha256']:
        raise ValueError('executed scheduler differs')
    for phase in plan['phases']:
        if stop_endpoint(plan) and time.time() >= stop_endpoint(plan):
            write_path = plan_path.parent / 'deadline-paused.json'
            write_path.write_text(json.dumps(dict(checked_utc=now(), deadline_unix=plan.get('window_end_unix') or plan['deadline_unix'], next_phase=phase['name'], complete=False))+'\n')
            return 0
        bundle, output = Path(phase['bundle']), Path(phase['output'])
        if sha(bundle / 'manifest.json') != phase['manifest_sha256']:
            raise ValueError('phase manifest changed')
        if any(disabled_geometry(c, plan) for c in load(bundle / 'manifest.json')['cells']):
            raise ValueError('disabled geometry in upcoming phase')
        if any(c['only_mode'] != 'score' for c in load(bundle / 'manifest.json')['cells']):
            raise ValueError('upcoming map scopes refused')
        wait = output.with_name(output.name + '.service')
        wait.mkdir(parents=True, exist_ok=True)
        write_path = wait / 'priority-wait.json'
        write_path.write_text(json.dumps(dict(pending=[], active=True, phase=phase['name'])) + '\n')
        command = [sys.executable, str(bundle / 'durable.py'), 'run', str(bundle / 'manifest.json'), str(output),
                   '--lock', plan['lock'], '--run-heavy', str(bundle / 'tools/run-heavy'), '--memory', '12G', '--jobs', '8',
                   '--timeout-seconds', '1100', '--max-attempts', str(phase['max_attempts'])]
        # Frozen source handles watchdogs, recovery and the shared per-attempt lock.
        from mac_service import supervise
        print(json.dumps(dict(starting=phase['name'], command=command)), flush=True)
        rc = supervise(command, output)
        if rc:
            return rc
        report = load(output / 'report.json')
        if not report['complete']:
            raise ValueError('phase did not complete')
        write_path.write_text(json.dumps(dict(pending=[], active=False, complete=True, phase=phase['name'])) + '\n')
    return 0


def install(root):
    plan = root / 'sequence.json'
    token = hashlib.sha256(str(plan.resolve()).encode()).hexdigest()[:16]
    command = [sys.executable, str(root / 'coordinator/scalar_priority.py'), 'execute', str(plan), '--sha256', sha(plan)]
    previous = None
    value = load(plan)
    for phase in value['phases'] + value.get('deferred_phases', []):
        wait = Path(phase['output']).with_name(Path(phase['output']).name + '.service')
        wait.mkdir(parents=True, exist_ok=True)
        (wait / 'priority-wait.json').write_text(json.dumps(dict(pending=([dict(manual_resume_required=True)] if phase in value.get('deferred_phases', []) else []) + ([previous] if previous else []), phase=phase['name'])) + '\n')
        previous = dict(output=phase['output'], manifest_sha256=phase['manifest_sha256'])
    if sys.platform == 'darwin':
        label = 'io.imazen.iqa-scalar.' + token
        service = f'user/{os.getuid()}/' + label
        plist = Path.home() / 'Library/LaunchAgents' / (label + '.plist')
        payload = dict(Label=label, ProgramArguments=command, RunAtLoad=True, KeepAlive=dict(SuccessfulExit=False),
                       ThrottleInterval=30, ProcessType='Background', LimitLoadToSessionType='Background',
                       StandardOutPath=str(root / 'service.log'), StandardErrorPath=str(root / 'service.stderr.log'),
                       EnvironmentVariables=dict(TMPDIR=str(Path.home() / 'tmp')))
        with plist.open('xb') as stream:
            plistlib.dump(payload, stream)
        subprocess.run(['launchctl', 'bootstrap', f'user/{os.getuid()}', str(plist)], check=True)
    else:
        service = 'iqa-scalar-' + token + '.service'
        subprocess.run(['systemd-run', '--user', '--unit=' + service, '--property=Restart=on-failure',
                        '--property=RestartSec=10', '--property=StandardOutput=append:' + str(root / 'service.log'),
                        '--property=StandardError=append:' + str(root / 'service.stderr.log'),
                        '--setenv=TMPDIR=' + str(Path.home() / 'tmp'), *command], check=True)
    write(root / 'installed-service.json', dict(service=service, command=command, checked_utc=now()))
    return service


def install_window_guard(plan_path):
    """Stop the coordinator at the first shared worker boundary after the endpoint."""
    root = plan_path.parent
    plan = load(plan_path)
    if not stop_endpoint(plan):
        raise ValueError('window guard requires explicit stop_at_deadline authorization')
    service = load(root / 'installed-service.json')['service']
    config = dict(lock=plan['lock'], sources=plan['phases'], stop_services=[service],
                  output=str(root / 'window-boundary'), deadline_unix=stop_endpoint(plan),
                  parent_sequence=str(plan_path), parent_sequence_sha256=sha(plan_path))
    path = root / 'window-guard.json'
    write(path, config)
    command = [sys.executable, str(Path(__file__).resolve()), 'deadline-boundary', str(path), '--sha256', sha(path)]
    token = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if sys.platform == 'darwin':
        label = 'io.imazen.iqa-window.' + token
        native = f'user/{os.getuid()}/' + label
        plist = Path.home() / 'Library/LaunchAgents' / (label + '.plist')
        with plist.open('xb') as stream:
            plistlib.dump(dict(Label=label, ProgramArguments=command, RunAtLoad=True,
                ProcessType='Background', LimitLoadToSessionType='Background',
                StandardOutPath=str(root/'window-guard.log'), StandardErrorPath=str(root/'window-guard.stderr.log'),
                EnvironmentVariables=dict(TMPDIR=str(Path.home()/'tmp'))), stream)
        subprocess.run(['launchctl', 'bootstrap', f'user/{os.getuid()}', str(plist)], check=True)
    else:
        native = 'iqa-window-' + token + '.service'
        subprocess.run(['systemd-run', '--user', '--unit='+native,
            '--property=StandardOutput=append:'+str(root/'window-guard.log'),
            '--property=StandardError=append:'+str(root/'window-guard.stderr.log'),
            '--setenv=TMPDIR='+str(Path.home()/'tmp'), *command], check=True)
    write(root/'window-guard-install.json', dict(service=native, command=command, window_end_unix=stop_endpoint(plan)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['boundary', 'deadline-boundary', 'ranking', 'rollout', 'execute', 'status', 'window-guard'])
    p.add_argument('config', type=Path)
    p.add_argument('--sha256')
    a = p.parse_args()
    if a.action == 'window-guard':
        install_window_guard(a.config)
        return 0
    if a.action == 'status':
        plan = load(a.config)
        qualified = {tuple(key) for key in plan.get('qualified_retained_keys', [])}
        for phase in plan['phases']:
            report = Path(phase['output']) / 'report.json'
            if report.exists():
                for cell in load(report).get('cells', []):
                    row = cell.get('row')
                    if cell['status'] == 'ok' and row and decision([dict(record=dict(status='ok', row=row))])[0] == 'retained':
                        qualified.add(coverage_key(cell['config']))
        choices = plan.get('deadline_choices', [])
        complete_choices = [choice for choice in choices if all((choice['metric'], choice['geometry'][0], choice['geometry'][1], choice['geometry'][2], 8, phase) in qualified for phase in ('cold', 'warm'))]
        service = load(a.config.parent / 'installed-service.json')['service']
        native = subprocess.run(['launchctl', 'print', service] if sys.platform == 'darwin' else
                    ['systemctl', '--user', 'show', service, '--property=ActiveState,MainPID'], capture_output=True, text=True)
        phases = []
        for phase in plan['phases'] + plan.get('deferred_phases', []):
            output = Path(phase['output'])
            wait = output.with_name(output.name + '.service') / 'priority-wait.json'
            phases.append(dict(name=phase['name'], bundle=phase['bundle'], output=str(output),
                progress=load(output / 'progress.json') if (output / 'progress.json').exists() else 'queued',
                wait=load(wait) if wait.exists() else None))
        print(json.dumps(dict(service=service, native_rc=native.returncode, native=native.stdout, phases=phases,
            elapsed_seconds=time.time()-plan['requested_start_unix'] if plan.get('requested_start_unix') else None,
            remaining_seconds=max(0,plan['deadline_unix']-time.time()) if plan.get('deadline_unix') else None,
            resume_plan=str(a.config.parent/'resume-sequence.json') if plan.get('deadline_unix') else None,
            deadline_targets_total=len(choices), deadline_targets_complete=len(complete_choices),
            deadline_methods_pending=[c['metric'] for c in choices if c not in complete_choices],
            retained_priority_methods=sorted({key[0] for key in plan.get('qualified_retained_keys', []) if key[0] in plan.get('priority_methods', [])}))))
        return 0
    if a.action == 'execute':
        awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())]) if sys.platform == 'darwin' else None
        try:
            return execute(a.config, a.sha256)
        finally:
            if awake:
                awake.terminate()
                awake.wait()
    if a.sha256 and sha(a.config) != a.sha256:
        raise ValueError('boundary configuration changed')
    config = load(a.config)
    if a.action == 'deadline-boundary':
        while time.time() < config['deadline_unix']:
            time.sleep(min(30, config['deadline_unix'] - time.time()))
    if config.get('parent_sequence') and sha(config['parent_sequence']) != config['parent_sequence_sha256']:
        raise ValueError('immutable parent sequence differs before boundary')
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=False)
    with Path(config['lock']).open('a+b') as lock:
        print('waiting shared boundary lock', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print('acquired shared boundary lock', flush=True)
        boundary(config, root / 'boundary.json')
        if a.action in ('boundary', 'deadline-boundary'):
            return 0
        replacement = remeasure(config, root, lock.fileno()) if config.get('remeasure_ranking') else []
        if config.get('ranking_final'):
            if sha(config['ranking_final']) != config['ranking_final_sha256']:
                raise ValueError('final ranking changed')
            ranking = load(config['ranking_final'])['ranking']
        else:
            ranking = rank(config, replacement)
        if a.action == 'ranking':
            write(root / 'ranking-final.json', dict(top10=ranking[:10], ranking=ranking, provisional=False))
            return 0
        prepare(config, root, ranking)
        print(json.dumps(dict(service=install(root), plan=str(root / 'sequence.json'))), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
