#!/usr/bin/env python3
"""Restrict a stopped canonical campaign to unfinished cells of an explicit route list.

--purpose changed (default): the approved 27 changed routes, one create-once reservation per parent.
--purpose remaining: any other routes the parent still owes, frozen with the parent's own worker so their
series stays one binary. A parent may have one successor of each purpose; a remaining successor refuses any
route an earlier successor of that parent already owns, so no cell is scheduled twice.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import durable
from mac_service import verify
from prepare_coverage import freeze
from prepare_gate_retry import frozen_controller


def select(cells, names, cap):
    groups=defaultdict(list)
    retained=[]
    excluded=[]
    exhausted=[]
    for i, cell in enumerate(cells):
        c=cell['config']
        if c['metric_name'] not in names:
            excluded.append(cell['cell_id'])
            continue
        if cell['status'] in ('ok','unsupported'):
            retained.append(cell['cell_id'])
            continue
        remaining=cap-len(cell['attempts'])
        if remaining<0:raise ValueError('attempt cap exceeded')
        if remaining:groups[remaining].append(i)
        else:exhausted.append(cell['cell_id'])
    return groups,retained,excluded,exhausted


RESERVATIONS={'changed':'changed-successor-reservation.json','remaining':'remaining-successor-reservation.json'}


def owned_elsewhere(results):
    """Routes already given to an earlier successor of this parent (read from each reservation's sequence)."""
    owned=set()
    for name in RESERVATIONS.values():
        r=results/name
        if r.exists():
            seq=Path(json.loads(r.read_text())['successor'])/'sequence.json'
            if not seq.exists():raise ValueError(f'reservation {r} has no successor sequence; inspect before continuing')
            owned|=set(json.loads(seq.read_text())['metrics'])
    return owned


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('bundle','results','output','metrics'):p.add_argument(n,type=Path)
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--purpose',choices=sorted(RESERVATIONS),default='changed')
    a=p.parse_args()
    names=set(json.loads(a.metrics.read_text()))
    if a.purpose=='changed' and len(names)!=27:raise ValueError('exact approved 27-route list required')
    if not names:raise ValueError('empty route list')
    overlap=names&owned_elsewhere(a.results)
    if overlap:raise ValueError('routes already owned by an earlier successor: '+', '.join(sorted(overlap)))
    if a.output.exists():raise ValueError('new immutable successor directory required')
    with durable.locked(a.lock,wait=True),durable.locked(a.results/'coordinator.lock'):
        verify(a.bundle)
        controller=frozen_controller(a.bundle)
        _,configs,frozen=controller.manifest(a.bundle/'manifest.json')
        saved=durable.load(a.results/'identity.json')
        identity=durable.digest(durable.encoded(saved))
        if saved['frozen']!=frozen:raise ValueError('frozen identity differs')
        # Rebuild projection from sealed attempts after the old controller stopped.
        report=controller.report(a.results,configs,identity,saved['policy']['max_attempts'])
        if not names.issubset({c['metric_name'] for c in configs}):raise ValueError('changed route absent')
        groups,retained,excluded,exhausted=select(report['cells'],names,saved['policy']['max_attempts'])
        with (a.results/RESERVATIONS[a.purpose]).open('x') as f:
            json.dump({'successor':str(a.output.resolve()),'parent_identity':identity},f)
        a.output.mkdir()
        raw=durable.load(a.bundle/'manifest.json')['cells']
        successors=[]
        for remaining,indices in sorted(groups.items()):
            selected=[raw[i] for i in indices]
            for c in selected:
                if c['only_mode']!='score' or max(c['width'],c['height'])>=7680:raise ValueError('scalar non8K required')
            item=freeze(a.bundle,a.output/f'remaining-attempts-{remaining}',selected,a.results/'report.json')
            item.update(max_attempts=remaining,parent_cell_ids=[report['cells'][i]['cell_id'] for i in indices])
            successors.append(item)
        receipt=dict(parent_identity=identity,parent_report_sha256=durable.file_hash(a.results/'report.json'),
            purpose=a.purpose,metrics=sorted(names),successors=successors,retained_changed_cell_ids=retained,
            excluded_unchanged_cell_ids=excluded,exhausted_changed_cell_ids=exhausted,
            rule='Never rerun completed changed cells; prior attempts remain consumed; unchanged history preserved')
        (a.output/'sequence.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt))

if __name__=='__main__':main()
