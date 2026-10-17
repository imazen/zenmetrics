#!/usr/bin/env python3
"""Write exact-size timing inventory; unions are coverage, never timing rankings."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def report(site):
    rows=json.loads((site/'data/speed.json').read_text())
    catalog=json.loads((site/'data/catalog.json').read_text())
    universe={r['metric'] for r in rows+catalog}
    cells=defaultdict(set)
    for r in rows:
        if not isinstance(r.get('ms'),(float,int)) or r['ms']<=0:continue
        device='CPU' if r['backend'].lower()=='cpu' else 'GPU'
        cells[(r['width'],r['height'],r['host'],device,str(r['threads']))].add(r['metric'])
    lines=['MISSING first: coverage unions below include every recorded mode, backend and implementation. They do not establish a comparable ranking. A missing host/thread label stays unknown. The ranking view separates exact workloads.',
           '', '| Missing metrics | Measured metrics | Size | Host | Device | Threads |', '| ---: | ---: | --- | --- | --- | --- |']
    for (w,h,host,device,threads),metrics in sorted(cells.items(),key=lambda item:(item[0][0]*item[0][1],item[0][2:])):
        lines.append(f'| {len(universe-metrics)} | {len(metrics)} | {int(w)}×{int(h)} | {host} | {device} | {threads} |')
    lines+=['','Missing metric names by cell:']
    for (w,h,host,device,threads),metrics in sorted(cells.items()):
        lines.append(f'\n- {int(w)}×{int(h)} / {host} / {device} / {threads}: '+', '.join(sorted(universe-metrics)))
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('site',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.write_text(report(args.site))
