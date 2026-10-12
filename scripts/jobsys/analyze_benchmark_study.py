#!/usr/bin/env python3
"""Hash-verified study observations, descriptive curves and portable SVG/HTML."""
import argparse
from collections import defaultdict
import html
import json
import math
from pathlib import Path
import statistics

from build_benchmark_study import canonical, decode, digest, file_sha, load, token


def number(value, label, zero=True):
    if type(value) not in (float, int) or not math.isfinite(value) or value < (0 if zero else 1):
        raise ValueError(f'invalid finite nonnegative {label}')
    return value


def linear_fit(points):
    """OLS on one explicitly selected regime; no uncertainty or acceptance."""
    if len(points) < 3 or len({x for x, _ in points}) < 3:
        return None
    xs, ys = zip(*points)
    center_x, center_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = math.fsum((x-center_x)**2 for x in xs)
    slope = math.fsum((x-center_x)*(y-center_y) for x, y in points) / sxx
    intercept = center_y - slope*center_x
    residuals = [y-(intercept+slope*x) for x, y in points]
    return dict(intercept=intercept, slope=slope, x_unit='pixels',
                residuals=residuals, points=len(points),
                rmse=math.sqrt(math.fsum(r*r for r in residuals)/len(points)),
                inference='descriptive OLS; no confidence interval or acceptance',
                regime='entire supplied size panel; inspect residuals and split cache regimes explicitly')


def read_cells(study):
    summary = load(study/'study.json')
    inv = load(study/'inventory.json')
    if digest(inv) != summary['inventory_sha']:
        raise ValueError('inventory hash differs from study summary')
    cells = {}
    for line in (study/'cells.jsonl').read_bytes().splitlines():
        row = decode(line)
        sha = token(row['cell_sha'])
        if row['study_sha'] != summary['study_sha'] or digest(row['contract']) != sha or sha in cells:
            raise ValueError('cell identity differs from declared study')
        cells[sha] = row
    if len(cells) != summary['declared_cells']:
        raise ValueError('declared cell count differs from inventory')
    return summary, cells


def verify_gpu_workload(case, workload):
    for key in ('backend', 'memory_mode', 'parameter_preset'):
        if workload[key] != case[key]:
            raise ValueError('GPU runtime workload differs from qualified case')
    device=workload['device']
    allowed=('CudaGpu',) if case['backend']=='cuda' else ('IntegratedGpu','DiscreteGpu')
    if device['name']!=case['expected_adapter'] or device['type'] not in allowed:
        raise ValueError('GPU runtime identity differs from qualified hardware')


def ingest(study, packaged, attempts):
    summary, cells = read_cells(study)
    identities = load(packaged/'identities.json')
    if identities['study_sha'] != summary['study_sha']:
        raise ValueError('packaged identities belong to another study')
    bundles = {row['cell_sha']: row['bundle_sha'] for row in identities['cells']}
    if len(bundles) != len(identities['cells']) or set(bundles) != {sha for sha, row in cells.items() if row['status']=='ready'}:
        raise ValueError('packaged cell inventory differs from study')
    for sha, expected in bundles.items():
        bundle_path=packaged/sha/'bundle.json'
        if file_sha(bundle_path)!=expected: raise ValueError('changed packaged bundle manifest')
        manifest=load(bundle_path);row=cells[sha]
        spec=load(study/sha/'bundle-spec.json')
        if (manifest['plan']!=spec['plan'] or manifest['target']!=row['target']
                or manifest['files']['driver']['sha256']!=row['sources']['artifact']['binary']['sha256']
                or manifest['files']['build.json']['sha256']!=row['sources']['artifact']['build']['sha256']
                or manifest['files']['case.json']['sha256']!=file_sha(study/sha/'case.json')
                or load(study/sha/'case.json')!=row['case']
                or manifest['plan']['expected_correctness']!=row['expected_correctness']):
            raise ValueError('changed packaged case, build or plan')
        for side in ['reference','distorted']:
            if manifest['files'][side+'.rgb']['sha256']!=row['sources']['inputs'][side]['sha256']:
                raise ValueError('changed declared input hash')
    evidence, seen = defaultdict(list), set()
    for directory in attempts:
        directory = directory.resolve()
        packet = load(directory/'packet.json')
        if packet['schema'] != 1 or packet['kind'] != 'zenbench_fleet_attempt':
            raise ValueError('unknown fleet packet schema')
        attempt = token(packet['attempt'])
        if attempt in seen: raise ValueError('duplicate attempt identity')
        seen.add(attempt)
        sha = packet['experiment']
        if sha not in cells or sha not in bundles: raise ValueError('attempt is not a qualified declared cell')
        row = cells[sha]
        if (packet['bundle_sha256'] != bundles[sha] or packet['machine'] != row['machine']
                or packet['domain'] != row['domain'] or packet['replicate'] != row['replicate']
                or packet['runner']['runner']['sha256'] != identities['runner_sha']):
            raise ValueError('attempt runner/bundle/testbed/replicate identity mismatch')
        spec = load(study/sha/'bundle-spec.json')
        if packet['plan'] != spec['plan']: raise ValueError('attempt workload differs from declared plan')
        for name, expected in packet['artifacts'].items():
            relative = Path(name)
            if relative.is_absolute() or not relative.parts or any(p in ('.','..') for p in relative.parts):
                raise ValueError('unsafe artifact path')
            path = (directory/relative).resolve()
            if not path.is_relative_to(directory) or file_sha(path) != expected:
                raise ValueError('changed or escaped attempt artifact')
        record = dict(attempt=attempt, cell_sha=sha, status=packet['status'],
                      packet_sha256=file_sha(directory/'packet.json'),
                      resources_before=packet.get('resources_before'), resources_after=packet.get('resources_after'),
                      placement=packet.get('placement'), runner=packet['runner']['runner'],
                      inference=packet['inference'], points=[])
        if packet['status']=='complete':
            driver = load(directory/'driver.json')
            if (packet['correctness'] != row['expected_correctness'] or driver['correctness'] != row['expected_correctness']
                    or driver['operations'] != summary['operations_per_launch']):
                raise ValueError('completed attempt correctness differs from qualification')
            if 'driver.json' not in packet['artifacts']: raise ValueError('driver evidence is not bound')
            if row['measurement']=='wall':
                value=number(packet['wall_seconds'],'wall seconds')
                record['points'].append(dict(quantity='invocation_wall',scope='whole_child',unit='seconds',value=value))
                if value > 0:
                    record['points'].append(dict(quantity='operation_throughput',scope='whole_child',unit='operations/s',
                                                 value=summary['operations_per_launch']/value))
            for memory in packet['memory']:
                quantity, scope = memory['quantity'], memory['scope']
                if memory.get('bytes') is not None:
                    record['points'].append(dict(quantity=quantity,scope=scope,unit='bytes',
                        value=number(memory['bytes'],'memory bytes'),coverage=memory.get('coverage'),
                        backend=memory.get('backend'),backend_version=memory.get('backend_version')))
                else:
                    record.setdefault('unavailable',[]).append(memory)
            if 'stdout.txt' in packet['artifacts']:
                stdout = (directory/'stdout.txt').read_text()
                lines=[decode(line) for line in stdout.splitlines() if line.startswith('{')]
                if len(lines)!=1: raise ValueError('expected one bound IQA driver stdout record')
                value=lines[0]
                if value['driver'] != driver or (row['measurement']=='wall' and value['timing_usable'] is not True):
                    raise ValueError('latency stdout and bound driver result differ')
                record['workload']=value['workload']
                if row['case']['schema']==4:
                    verify_gpu_workload(row['case'],value['workload'])
                    record.setdefault('unavailable',[]).append(dict(quantity='device_memory',reason=value['workload']['device_memory']))
                if row['measurement']=='wall':
                    durations=value['operation_ns']
                    if len(durations)!=summary['operations_per_launch']:
                        raise ValueError('operation timing budget mismatch')
                    mean=statistics.fmean(number(n,'operation ns') for n in durations)/1e9
                    record['points'].append(dict(quantity='metric_call_mean',scope='decoded_input_score',unit='seconds',value=mean,
                                                 operations=len(durations),replication='one invocation; operations are not independent blocks'))
        evidence[sha].append(record)
    audit, observations = [], []
    for sha, cell in cells.items():
        records=evidence[sha]
        complete=[record for record in records if record['status']=='complete']
        state=cell['status'] if cell['status']!='ready' else ('observed' if len(complete)==1 else 'ambiguous_retry' if complete else 'failed' if records else 'missing')
        audit.append(dict(cell_sha=sha,status=state,declared_status=cell['status'],reasons=cell['reasons'],attempts=records))
        if state=='observed':
            pair=cell['contract']['pair'];case=cell['case']
            record=complete[0]
            # Never combine a changed allocator/compiler/features/backend/viewing/content/aspect panel.
            group=dict(metric=case['metric'],testbed=cell['testbed'],machine=cell['machine'],domain=cell['domain'],
                       target=cell['target'],reuse=case['reuse'],threads=case['threads'],format=case['pixel_format'],
                       viewing=case.get('viewing'),origin=pair['origin'],content=pair['content'],
                       series=pair.get('series',pair['id']),case_schema=case['schema'],
                       products=case.get('products','primary'),backend=case.get('backend','native_cpu'),
                       memory_mode=case.get('memory_mode'),parameter_preset=case.get('parameter_preset'),
                       gpu_device=cell['contract']['testbed'].get('gpu_device'),
                       warmup_operations=case.get('warmup_operations',1 if case['reuse']=='warm_reference' else 0),
                       operations=summary['operations_per_launch'],
                       row_padding_bytes=case['stride_bytes']-case['width']*(12 if case['pixel_format']=='rgb_f32le_bt709_linear_nits' else 3),
                       aspect=[case['width']//math.gcd(case['width'],case['height']),case['height']//math.gcd(case['width'],case['height'])],
                       build_sha=cell['sources']['artifact']['build']['sha256'],measurement=cell['measurement'],
                       placement=record['placement'],native_arch_verified=record['runner']['native_arch_verified'])
            for point in record['points']:
                observations.append(dict(point,group=group,cell_sha=sha,attempt=record['attempt'],replicate=cell['replicate'],
                    pixels=case['width']*case['height'],width=case['width'],height=case['height'],
                    stride_bytes=case['stride_bytes'],packet_sha256=record['packet_sha256']))
    return dict(schema=1,kind='zenmetrics_study_analysis',study_sha=summary['study_sha'],audit=audit,
                observations=observations,inference='descriptive; no benchmark acceptance decision')


def curves(analysis):
    groups=defaultdict(list)
    for point in analysis['observations']:
        identity=dict(group=point['group'],quantity=point['quantity'],scope=point['scope'],unit=point['unit'],
                      coverage=point.get('coverage'),backend=point.get('backend'),backend_version=point.get('backend_version'))
        groups[canonical(identity)].append(point)
    result=[]
    for identity, points in sorted(groups.items()):
        by_size=defaultdict(list)
        for point in points: by_size[point['pixels']].append(point['value'])
        panel=[(pixels,statistics.fmean(values)) for pixels,values in sorted(by_size.items())]
        result.append(dict(identity=json.loads(identity),observations=points,fit=linear_fit(panel)))
    return result


def svg(curve, logarithmic=False):
    """Standalone chart of actual invocation points; no implied confirmation CI."""
    points=curve['observations'];identity=curve['identity'];group=identity['group']
    transform=math.log10 if logarithmic else lambda x:x
    plotted=points
    if not plotted or (logarithmic and any(p['value']<=0 for p in plotted)):
        raise ValueError('log chart needs positive observations; choose linear to retain zero values')
    xs=[transform(p['pixels']) for p in plotted];ys=[transform(p['value']) for p in plotted]
    xmin,xmax=min(xs),max(xs);ymin,ymax=min(ys),max(ys)
    if xmin==xmax: xmin-=0.5;xmax+=0.5
    if ymin==ymax: ymin-=0.5;ymax+=0.5
    if not logarithmic: ymin=0;ymax=max(ymax,1e-12)*1.1
    sx=lambda x:90+(transform(x)-xmin)/(xmax-xmin)*780
    sy=lambda y:420-(transform(y)-ymin)/(ymax-ymin)*280
    esc=html.escape
    title=f"{group['metric']} · {identity['quantity']} · {group['testbed']}"
    subtitle=f"{identity['unit']} · {identity['scope']} · {group['threads']} threads · {group['reuse']} · {'log10' if logarithmic else 'linear'} axes"
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540" role="img"><title>{esc(title)}</title>',
           '<rect width="960" height="540" fill="#f5f7fc"/><rect x="64" y="112" width="836" height="332" rx="16" fill="white"/>',
           '<g font-family="sans-serif" fill="#132340">',f'<text x="64" y="54" font-size="25" font-weight="700">{esc(title if len(title)<=60 else title[:57]+"…")}</text>',
           f'<text x="64" y="85" font-size="15">{esc(subtitle)}</text>']
    inverse=lambda x:10**x if logarithmic else x
    for i in range(5):
        x=xmin+(xmax-xmin)*i/4;y=ymin+(ymax-ymin)*i/4
        px=90+780*i/4;py=420-280*i/4
        parts.extend([f'<path d="M{px} 140V420 M90 {py}H870" stroke="#e2e8f1"/>',
                      f'<text x="{px}" y="465" text-anchor="middle" font-size="12">{inverse(x):.4g}</text>',
                      f'<text x="80" y="{py+4}" text-anchor="end" font-size="12">{inverse(y):.4g}</text>'])
    for p in plotted:
        label=f"{p['width']}×{p['height']}; value={p['value']}; replicate={p['replicate']}; attempt={p['attempt']}"
        parts.append(f'<g><title>{esc(label)}</title><circle cx="{sx(p["pixels"]):.4f}" cy="{sy(p["value"]):.4f}" r="5" fill="#1765db" fill-opacity=".72"/></g>')
    parts.extend(['<text x="480" y="496" text-anchor="middle" font-size="14">Image pixels</text>',
                  '<text x="64" y="524" font-size="12">Individual invocation points · descriptive evidence · no confirmation interval or acceptance</text></g></svg>'])
    return ''.join(parts)


def write(analysis, output, logarithmic=False):
    output.mkdir(parents=True,exist_ok=False)
    analysis['curves']=curves(analysis)
    (output/'analysis.json').write_bytes(canonical(analysis)+b'\n')
    cards=[]
    for index,curve in enumerate(analysis['curves']):
        name=f'curve-{index:04d}.svg';(output/name).write_text(svg(curve,logarithmic))
        identity=html.escape(json.dumps(curve['identity'],sort_keys=True))
        cards.append(f'<article data-search="{identity.lower()}"><a href="{name}" download>Download SVG</a><img src="{name}" alt="{identity}"><details><summary>Exact panel and descriptive fit</summary><pre>{html.escape(json.dumps(dict(identity=curve["identity"],fit=curve["fit"]),indent=2))}</pre></details></article>')
    counts={state:sum(row['status']==state for row in analysis['audit']) for state in sorted({row['status'] for row in analysis['audit']})}
    page='<!doctype html><meta charset="utf-8"><title>Metric cost study</title><style>body{font:16px system-ui;background:#f5f7fc;color:#132340;margin:32px}article{background:white;margin:24px 0;padding:20px;border-radius:16px}img{width:100%;max-width:960px}input{padding:12px;width:70%}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #cad4e4}</style>'
    page+='<h1>Metric cost study</h1><p>Descriptive observations. Resource cleanliness and native execution require separate qualification. No confirmatory decision.</p>'
    page+=f'<p>Study {html.escape(analysis["study_sha"])}</p><pre>{html.escape(json.dumps(counts,indent=2))}</pre><p><a href="analysis.json">Full observations, failures, hashes and fits</a></p>'
    page+='<label>Filter panels <input id="filter" placeholder="metric, testbed, threads, scope, unit"></label>'+''.join(cards)
    page+='<h2>Declared cell audit</h2><table><tr><th>Cell</th><th>Status</th><th>Reasons</th><th>Attempts</th></tr>'
    for row in analysis['audit']:
        page+='<tr>'+''.join(f'<td>{html.escape(str(value))}</td>' for value in [row['cell_sha'],row['status'],'; '.join(row['reasons']),', '.join(r['attempt']+': '+r['status'] for r in row['attempts'])])+'</tr>'
    page+='</table><script>document.querySelector("#filter").addEventListener("input",e=>{for(const card of document.querySelectorAll("article"))card.hidden=!card.dataset.search.includes(e.target.value.toLowerCase())})</script>'
    (output/'index.html').write_text(page)
    return dict(schema=1,curves=len(cards),cells=counts,output=str(output),inference=analysis['inference'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',type=Path);parser.add_argument('packaged',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('attempts',type=Path,nargs='*');parser.add_argument('--log',action='store_true')
    args=parser.parse_args()
    try: print(json.dumps(write(ingest(args.study,args.packaged,args.attempts),args.output,args.log),indent=2))
    except (ValueError,OSError,KeyError,TypeError) as error: parser.exit(2,f'study analysis refused: {error}\n')

if __name__=='__main__': main()
