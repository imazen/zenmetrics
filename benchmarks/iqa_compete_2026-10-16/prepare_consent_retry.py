#!/usr/bin/env python3
"""Separate owner-authorized retry budget for proven, exhausted consent refusals."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time
import scalar_priority as priority
from prepare_nn_durable import resolve_weight_consents


def eligible(raw, records, status, config):
    return (raw['only_mode'] == 'score' and not priority.disabled_geometry(raw, config)
            and status == 'exhausted' and bool(records)
            and all('WeightsConsent' in str((r['record'].get('row') or {}).get('error', '')) for r in records))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    a = p.parse_args()
    c = priority.load(a.config)
    if c.get('authorized_new_attempts') != 2 or not c.get('authorization') or c.get('accepted_weights') != 'all':
        raise ValueError('requires explicit authorization and two new consent-only attempts')
    root = Path(c['output'])
    root.mkdir(parents=True, exist_ok=False)
    with Path(c['lock']).open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        priority.boundary(dict(c, nn=[]), root/'boundary.json')
        phases, audit = [], []
        for index, source in enumerate(c['sources']):
            _, frozen, configs, evidence = priority.retained(source)
            selected, lineage = [], []
            for raw, config in zip(frozen['manifest']['cells'], configs):
                records = evidence.get(config['durable_cell_id'], [])
                status, remaining = priority.decision(records, source.get('max_attempts', 2))
                if not eligible(raw, records, status, c):
                    continue
                execution = dict(raw, worker_environment=dict(raw['worker_environment']))
                names = resolve_weight_consents(c['accepted_weights'], [raw['metric_name']])
                if not names:
                    raise ValueError('refusing unidentified consent terms')
                execution['worker_environment']['ZENMETRICS_NN_ACCEPT_WEIGHTS'] = names
                selected.append(execution)
                lineage.append(dict(original_config=raw, original_cell_id=config['durable_cell_id'],
                    original_remaining_attempts=remaining, original_status=status,
                    original_attempts=[dict(path=r['path'], sha256=r['sha256']) for r in records],
                    explicit_resolved_consent=names, new_attempt_budget=2))
            audit.append(dict(source=source, consent_cells=len(selected), original_binary_sha256=frozen['binary_sha256']))
            if not selected:
                continue
            selected.sort(key=lambda cell:(cell['width']*cell['height'], cell['metric_name'],cell['threads'],cell['timing_phase'],cell['cold_repeat']))
            device = selected[0]['device']
            name = ('gpu' if device!='cpu' else 'cpu') + '-consent-source' + str(index)
            phase = priority.freeze(source['bundle'], root/name, selected,
                dict(source=source, authorization=c['authorization'], old_attempts_reset=False, cells=lineage))
            phase.update(name=name, output=str(Path(c['results'])/name), max_attempts=2)
            phases.append(phase)
        if not phases:
            raise ValueError('no exhausted consent-only cells eligible')
        # The config orders all GPU sources before CPU sources.
        kinds=[priority.load(Path(p['bundle'])/'manifest.json')['cells'][0]['device']!='cpu' for p in phases]
        if kinds != sorted(kinds, reverse=True):
            raise ValueError('GPU sources must precede CPU')
        priority.write(root/'consent-audit.json', audit)
        priority.write(root/'authorization.json', dict(authorization=c['authorization'],
            new_attempts_per_cell=2, original_attempts_preserved=True, source_config_sha256=priority.sha(a.config)))
        coordinator=root/'coordinator'
        coordinator.mkdir()
        for name in ['scalar_priority.py','mac_service.py','durable.py','admission_recovery.py']:
            shutil.copy2(priority.HERE/name, coordinator/name)
        plan=dict(schema=1,lock=c['lock'],phases=phases,source_sha256=priority.sha(priority.HERE/'scalar_priority.py'),
            scheduler_files={str(p.relative_to(root)):priority.sha(p) for p in coordinator.iterdir()},
            disabled_geometries=c['disabled_geometries'],stop_at_deadline=False,window_end_unix=None,
            results_due_unix=c.get('results_due_unix'),requested_start_unix=time.time(),
            authorization_sha256=priority.sha(root/'authorization.json'),
            policy='Separate explicitly authorized consent-only retry campaign; old attempts/caps unchanged; GPU before CPU; no maps/8K')
        priority.write(root/'sequence.json', plan)
        print(json.dumps(dict(service=priority.install(root), cells=sum(p['cells'] for p in phases),
            plan_sha256=priority.sha(root/'sequence.json'))),flush=True)


if __name__ == '__main__': main()
