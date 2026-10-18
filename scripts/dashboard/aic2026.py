#!/usr/bin/env python3
"""Static AIC2026 evidence pages. No network, models, or reference code at build time."""
import csv
import hashlib
import html
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tomllib
from urllib.parse import quote

from catalog import ROOT, md_tables, plain
from model_cards import ledger, base_name, clean

B = 'benchmarks/iqa_complexity_2026-10-17'
NAV = [('index', 'Overview'), ('reproduction', 'Reproduction'), ('configurations', 'Configurations'),
       ('models', 'Model cards'), ('coverage', 'Screenshot coverage'), ('complexity', 'Cost & runtime'), ('pareto', 'Pareto'), ('statistics', 'Statistics')]
STYLE = '''
:root{color-scheme:light dark;--bg:#f6f5f1;--card:#fff;--ink:#202521;--muted:#606b64;--line:#d5dbd6;--accent:#236c60;--warn:#97501a}
@media(prefers-color-scheme:dark){:root{--bg:#141916;--card:#1e2620;--ink:#e6eee7;--muted:#acb9ad;--line:#3d4c40;--accent:#7ed3b6;--warn:#efb779}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,sans-serif}main,header,footer{max-width:1180px;margin:auto;padding:24px}header{border-bottom:1px solid var(--line)}nav{display:flex;flex-wrap:wrap;gap:8px 20px}a{color:var(--accent);overflow-wrap:anywhere}h1,h2{font-family:Georgia,serif;line-height:1.2}h1{font-size:clamp(30px,5vw,48px)}h2{margin-top:36px}p{max-width:85ch}.eyebrow{letter-spacing:.14em;text-transform:uppercase;font-size:12px;color:var(--muted)}.notice{padding:16px;border-left:4px solid var(--warn);background:var(--card)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,270px),1fr));gap:18px}.card,details{background:var(--card);border:1px solid var(--line);padding:18px;margin:12px 0}summary{cursor:pointer;font-weight:600}.table-wrap{overflow:auto;max-width:100%;border:1px solid var(--line)}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;min-width:90px;max-width:520px;overflow-wrap:anywhere}th{background:var(--card);position:sticky;top:0}code,pre{font-family:ui-monospace,monospace;font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere}svg{display:block;width:100%;height:auto;color:var(--ink)}.plot{background:var(--card);border:1px solid var(--line);padding:12px}.plot text{fill:var(--ink);font:12px system-ui,sans-serif}.plot .point{fill:var(--accent);opacity:.65}.plot .front{stroke:var(--warn);fill:none;stroke-width:2}.plot .axis{stroke:var(--muted);fill:none}footer{font-size:12px;color:var(--muted);border-top:1px solid var(--line)}.muted{color:var(--muted)}@media(max-width:500px){main,header,footer{padding:16px}td{min-width:100px}.plot{padding:3px}}
'''


def e(value):
    return html.escape(str(value))


def source_commit():
    value = os.environ.get('GITHUB_SHA') or os.environ.get('AIC_SOURCE_COMMIT')
    if not value:
        result = subprocess.run(['jj', 'log', '-r', '@', '--no-graph', '-T', 'commit_id'], cwd=ROOT, capture_output=True, text=True)
        if result.returncode == 0:
            value = result.stdout.strip()
        else:
            git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
            if git.returncode:
                return 'unversioned-archive'
            value = git.stdout.strip()
    if not re.fullmatch('[0-9a-f]{40}', value):
        raise ValueError('full source commit required')
    return value


def source_link(path, commit, label=None):
    return '<a href="https://github.com/imazen/zenmetrics/blob/'+('master' if commit=='unversioned-archive' else commit)+'/'+quote(path, safe='/')+'">'+e(label or path)+'</a>'


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th scope="col">'+e(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def fmt(v):
    return 'not measured' if v is None else f'{v:.6g}'


def json_text(value):
    return e(json.dumps(value, ensure_ascii=False, sort_keys=True))


def load_paper():
    # Isolate its `correlate`/`coverage` imports from the dashboard's modules.
    spec = importlib.util.spec_from_file_location('aic_paper', ROOT/'scripts/complexity/paper.py')
    mod = importlib.util.module_from_spec(spec)
    saved = {name: sys.modules.pop(name, None) for name in ('correlate', 'coverage')}
    sys.path.insert(0, str(ROOT/'scripts/complexity'))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
        for name, previous in saved.items():
            sys.modules.pop(name, None)
            if previous is not None: sys.modules[name] = previous
    return mod


def timings():
    observations = {}
    for path in sorted((ROOT/B/'measured').glob('*.tsv')):
        lines = [s for s in path.read_text().splitlines() if s.strip() and not s.startswith('#')]
        for row in csv.DictReader(lines, delimiter='\t'):
            if row['shape'] not in ('1024x1024', '3840x2160') or row['threads'] not in ('1', '8'): continue
            try:
                cpu, wall = float(row['task_clock_ms_per_call']), float(row['wall_ms_per_call'])
            except ValueError: continue
            if not all(math.isfinite(x) and x > 0 for x in (cpu, wall)): raise ValueError('invalid measured timing')
            key = (row['metric'], row['shape'], int(row['threads']), path.relative_to(ROOT).as_posix())
            observations.setdefault(key, []).append((cpu, wall))
    return [dict(metric=m, shape=shape, threads=t, source=p, n=len(v), cpu_ms=statistics.median(a for a, _ in v),
                 wall_ms=statistics.median(b for _, b in v)) for (m, shape, t, p), v in observations.items()]



def ssim_dispatch_inputs():
    source=(ROOT/'crates/zenmetrics-cli/src/metrics/mod.rs').read_text()
    names=dict(re.findall(r'MetricKind::(\w+) => "([^"]+)"',source))
    start=source.index('fn ssim_pyiqa_flags(')
    body=source[start:source.index('\n    }',start)]
    result={}
    for kind,color,multiscale,rounded,factor in re.findall(r'Self::(\w+) => Some\(\((true|false), (true|false), (true|false), (\d+)\)\)',body):
        result[names[kind]]={'format':'packed sRGB8, codes 0–255', 'arithmetic':'f64 input and accumulation; f32 in the name means unrounded input',
             'channels':'per-RGB-channel mean' if color=='true' else 'BT.601 luma: 0.299R + 0.587G + 0.114B',
             'rounding':'ties-to-even before downscale' if rounded=='true' else 'unrounded',
             'fixed_box_factor':int(factor),'edge_policy':'complete blocks from (0,0); discard incomplete bottom/right blocks',
             'scales':5 if multiscale=='true' else 1,
             'source':'crates/zenmetrics-cli/src/metrics/ssim_pyiqa.rs'}
    if not result:raise ValueError('SSIM dispatch flags were not parsed')
    return result

def reproduction_rows(cards, commit):
    """Every published column, plus unavailable site experiments, from the versioned audit."""
    mapping = json.loads((ROOT/'scripts/complexity/aic2026_columns.json').read_text())['map']
    rows = []
    for tab in md_tables(ROOT/'docs/AIC2026_COVERAGE_AUDIT_2026-10-15.md'):
        if len(tab[0]) != 9 or tab[0][0] != 'name': continue
        for fields in tab[1:]:
            name, where, impl, n, med, mx, jnd, status, note = fields
            if name.startswith('JND_'): continue
            if where == 'neither' and status not in ('unverifiable', 'missing', 'unverified'): continue
            exact = mapping.get(name)
            if exact: impl = exact
            if name == 'AIMOS': status, note = 'MISSING: unidentified', '16 of 34 NR candidates scored; no match; best hyperiqa ratio Spearman 0.82.'
            if name.startswith('WD_s'): impl, status, note = 'wd (multi-output)', 'unreproduced', 'Published sigma mapping unresolved; best median 0.13–0.20 site-JND in bounded search.'
            if name == 'A-FINE': impl, status, note = 'afine', 'unreproduced', 'Entry point and multiple-of-32 size handling unresolved; best median 0.11 site-JND.'
            if name.startswith('proposal-DVIFM'): status, note = 'unreproduced', 'Released-build port presets exist outside the CLI; no published column matches. Ask which build/constants/JND map.'
            if 'HaarXYB' in name or 'HaarY-' in name or 'HaarYCoCg' in name or '-vqmt' in name: status = 'MISSING: published values'
            if name == 'DMM': status = 'fails native gate'; note = 'dmm-pyiqa square resize; worst 0.071 JND versus native oracle, above 0.001 gate (SVD).'
            if name in ('TOPIQ', 'TOPIQ-PIPAL', 'AHIQ', 'PieAPP', 'WaDIQaM', 'CW-SSIM'):
                impl = mapping[name]; status = 'implemented; native gate passes'; note += ' Public non-commercial pyiqa convention; see model card and current reproduction status.'
            if name.startswith('proposal-IDFIQA') or name == 'MSSWD': status = 'private; aggregate match'
            if status=='missing': status='MISSING: implementation or comparison'
            if where=='neither' and status=='unverified': status='MISSING: published values'
            if name.startswith('HDR_VDP_2'): impl, status, note = 'hdrvdp', 'unreproduced', 'Identified res.Q, BT.709, 200*sRGB_EOTF nits, 37.84 ppd; median residual 0.028 Q. Explicit ingress/output still needs native qualification; see configuration-search receipt.'
            if name.startswith('HDR_VDP_3'): impl, status, note = 'hdrvdp3', 'configuration identified', 'res.Q (not Q_JOD), XYZ oled, 200*sRGB_EOTF nits, 37.84 ppd; median 0.00136/max 0.0129 site-JND in reference search; native route qualification remains separate.'
            if name == 'proposal-Butteraugli':
                impl = 'butteraugli_pnorm3 + identified map'; status = 'configuration identified'
                med, mx, jnd = '', '', '5.1e-5';note = 'max(0.00459792*N, 2.0938201*N - 1.1033102); median 3.2e-8 site-JND over all 9618 rows; constants provenance unknown.'
            rows.append(dict(column=plain(name), availability=where, implementation=plain(impl), n=n, median=med, maximum=mx, max_jnd=jnd, status=status, note=plain(note)))
    rows.sort(key=lambda r: (0 if r['status'].startswith('MISSING') else 1 if r['status'] in ('unverifiable', 'unreproduced', 'fails native gate') else 2, r['column']))
    return rows


def plot(points, title, xlabel, ylabel, *, logx=True, front=False):
    """Dependency-free SVG with exact data in accessible titles and a companion table."""
    if not points: return '<p>No paired measurements.</p>'
    transform = math.log10 if logx else lambda x: x
    xs, ys = [transform(p[0]) for p in points], [p[1] for p in points]
    xlo, xhi, ylo, yhi = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (xhi-xlo or 1)*.08, (yhi-ylo or 1)*.1
    xlo-=dx; xhi+=dx; ylo-=dy; yhi+=dy
    X=lambda x: 70+650*(transform(x)-xlo)/(xhi-xlo)
    Y=lambda y: 400-350*(y-ylo)/(yhi-ylo)
    parts=['<figure class="plot"><svg role="img" aria-label="'+e(title)+'" viewBox="0 0 780 470"><title>'+e(title)+'</title><path class="axis" d="M70 40V400H730"/>']
    for i in range(6):
        x=xlo+(xhi-xlo)*i/5; y=ylo+(yhi-ylo)*i/5
        parts += [f'<text x="{70+650*i/5}" y="422" text-anchor="middle">{(10**x if logx else x):.3g}</text>', f'<text x="62" y="{400-350*i/5}" text-anchor="end">{y:.3g}</text>']
    if front:
        best=-math.inf; frontier=[]
        for x,y,name in sorted(points):
            if y>best: frontier.append((x,y));best=y
        parts.append('<path class="front" d="'+' '.join(('M' if i==0 else 'L')+f'{X(x):.2f} {Y(y):.2f}' for i,(x,y) in enumerate(frontier))+'"/>')
    for x,y,name in points:
        parts.append(f'<circle class="point" cx="{X(x):.3f}" cy="{Y(y):.3f}" r="4"><title>{e(name)}: {x:.8g}, {y:.8g}</title></circle>')
    parts.extend(['<text x="400" y="458" text-anchor="middle">'+e(xlabel)+'</text>', '<text transform="translate(18 225) rotate(-90)" text-anchor="middle">'+e(ylabel)+'</text></svg><figcaption>'+e(title)+'</figcaption></figure>'])
    return ''.join(parts)


def build(output, cards):
    commit = source_commit()
    folder = output/'aic2026'; folder.mkdir(parents=True, exist_ok=True)
    sources = set(['docs/AIC2026_REPRODUCTION_STATUS.md', 'docs/AIC2026_COVERAGE_AUDIT_2026-10-15.md',
                   'docs/AIC2026_METRICS_AND_FITTING.md', 'scripts/complexity/aic2026_columns.json',
                   'crates/zenmetrics-cli/src/metrics/mod.rs', 'docs/paper/IQA_COMPLEXITY_VS_RUNTIME.md'])
    def page(name, title, body, paths):
        sources.update(paths)
        footer='<footer>AIC2026 dataset: Jenadeleh et al. (2026), <a href="https://doi.org/10.18419/DARUS-6156">dataset DOI and attribution</a>. Dataset terms: CC BY-SA 4.0 with metadata exceptions; preliminary subjective-score terms remain an author request.<br>Source commit <code>'+commit+'</code>. Data and contracts:<ul>'+''.join('<li>'+source_link(p,commit)+'</li>' for p in sorted(set(paths)))+'</ul></footer>'
        text='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark"><meta http-equiv="Content-Security-Policy" content="default-src &#39;self&#39;; style-src &#39;self&#39; &#39;unsafe-inline&#39;; img-src &#39;self&#39; data:; base-uri &#39;none&#39;"><link rel="icon" href="data:,"><link rel="stylesheet" href="style.css"><title>'+e(title)+' · AIC2026</title></head><body><header><a href="../">Zenmetrics</a><p class="eyebrow">JPEG AIC2026 · evidence & open questions</p><nav aria-label="AIC2026">'+''.join('<a '+('aria-current="page" ' if n==name else '')+'href="'+('index.html' if n=='index' else n+'.html')+'">'+e(t)+'</a>' for n,t in NAV)+'</nav></header><main><h1>'+e(title)+'</h1>'+body+'</main>'+footer+'</body></html>'
        (folder/(name+'.html')).write_text(text)
    (folder/'style.css').write_text(STYLE)
    private=json.loads((ROOT/'benchmarks/dashboard/aic2026/private-aggregates.json').read_text())
    rows=reproduction_rows(cards,commit)
    status_paths=['docs/AIC2026_REPRODUCTION_STATUS.md','docs/AIC2026_COVERAGE_AUDIT_2026-10-15.md','benchmarks/dashboard/aic2026/private-aggregates.json']
    page('reproduction','Reproduction status', '<p class="notice">Missing evidence comes first. Native-oracle parity and matching a dataset column are separate checks. A matching method name alone qualifies neither. Current questions and detailed receipts: '+source_link(status_paths[0],commit)+'.</p>'+table(['Published column / site experiment','Availability','Implementation','n','Median |Δ|','Max |Δ|','Max site-JND Δ','Status','Configuration / qualification','Evidence'],[[e(r[k]) for k in ['column','availability','implementation','n','median','maximum','max_jnd','status','note']]+[source_link(status_paths[0],commit,'Current receipt ledger')] for r in rows]),status_paths)
    column_contract=json.loads((ROOT/'scripts/complexity/aic2026_columns.json').read_text())
    mapping=column_contract['map']
    ssim_inputs=ssim_dispatch_inputs()
    p=load_paper(); tallies=p.tallies(); measured=timings()
    by_name={c['metric']:c for c in cards}
    entries=ledger()
    configs=[]
    for name,c in sorted(by_name.items()):
        tally=tallies.get(name);params=tally[0].get('params',{}) if tally else {}
        card_source=c['source'];model_crate=card_source.split('/')[1] if card_source.startswith('crates/') else None
        entry=entries.get(base_name(name,entries),[])
        crate=model_crate or (plain(entry[1]) if len(entry)>1 else 'zenmetrics-cli (implementation in dispatch module)')
        options={'CLI':c['cli_contract'],'tally_parameters':params,
                 'AIC_mapping_qualification':{col:column_contract.get('notes',{}).get(col,'See reproduction status for measured residual and pipeline scope.') for col,m in mapping.items() if m==name}}
        if tally: options['historical_tally_assumptions_not_current_configuration']=tally[0].get('assumptions',[])
        configs.append(dict(metric=name,crate=crate,source_commit=commit,measurement_source_commit=tally[0].get('source_commit','not recorded') if tally else 'not tallied',options=clean(options),ingress=ssim_inputs.get(name,c['inputs']),aic_columns=[k for k,v in mapping.items() if v==name],licence='public source; '+c['rust_license']+'; weights: '+c['weights_license'],parity=c['parity'],parity_status=c['parity_status'],source=card_source,reproduction=c['repro']))
    config_paths=['crates/zenmetrics-cli/src/metrics/mod.rs','crates/zenmetrics-cli/src/metrics/ssim_pyiqa.rs','docs/METRIC_PROVENANCE.md','scripts/complexity/aic2026_columns.json']+sorted({c['source'] for c in cards if (ROOT/c['source']).is_file()})+sorted(str(x.relative_to(ROOT)) for x in (ROOT/B/'tallies').glob('*.json'))
    config_body='<p>Every CLI name is listed separately, including the pyiqa ports. Source commit identifies the configuration source; counted-source commits identify the historical tally. Ingress, resize and colour handling follow the linked contract. Display-dependent reproduction commands explicitly select <code>standard_fhd</code>; it is not an implicit universal default. Missing metadata is labelled, never inferred from a metric name.</p>'
    config_body+=table(['CLI metric','Crate / source','Source / counted commits','Options & defaults','Ingress (colour, resize, depth)','AIC column','Licence class','Native parity / gate'],[[e(r['metric']),e(r['crate'])+'<br>'+source_link(r['source'],commit),e(commit)+'<br>'+e(r['measurement_source_commit']),'<details><summary>Exact contract & counted options</summary><pre>'+json_text(r['options'])+'</pre></details>',json_text(r['ingress']),e(', '.join(r['aic_columns']) or 'no established column mapping'),e(r['licence']),e(r['parity_status'])+'<pre>'+json_text(r['parity'])+'</pre>'] for r in configs])
    config_body+='<h2>Private aggregate records</h2><p>Private implementations have no public CLI configuration or distributable model here. Only aggregate qualification is published.</p>'+table(['Metric','Class','Column match','Native gate'],[[e(r['metric']),'private; aggregates only',e(r['published_match']),e(r['native_gate'])] for r in private['models']])
    config_body+='<h2>Measured run configurations</h2><p>Each measurement source retains its exact binary, thread configuration and command in its adjacent metadata. Multiple sources remain separate; setup and HDR decode exceptions are stated there.</p><ul>'+''.join('<li>'+source_link(str(x.relative_to(ROOT)),commit)+'</li>' for x in sorted((ROOT/B/'measured').glob('*.meta')))+'</ul>'
    page('configurations','Exact configurations',config_body,config_paths+[str(x.relative_to(ROOT)) for x in (ROOT/B/'measured').glob('*.meta')]+['benchmarks/dashboard/aic2026/private-aggregates.json'])
    model_body='<p>Generated from the same evidence model as the existing dashboard. Code and checkpoint terms are independent. Declared gates are distinct from reported results. <a href="model-cards.json">Download complete public model cards (JSON)</a>.</p>'
    (folder/'model-cards.json').write_text(json.dumps(cards,ensure_ascii=False,allow_nan=False,separators=(',',':')))
    for c in cards:
        model_body+='<details id="'+e(c['metric'])+'"><summary>'+e(c['metric'])+' · '+e(c['parity_status'])+'</summary><p>'+e(c['rust_license'])+'; weights: '+e(c['weights_license'])+'</p><p>'+e(c['recorded_paper'])+'</p><p>'+e(c['reference_implementation'])+'</p><p>'+source_link(c['source'],commit)+'</p><pre>'+json_text({k:c[k] for k in ['citation','original_authors','reference_code_license','inputs','cli_contract','conventions','parity','reference_identity','output_range','direction','weights','repro']})+'</pre></details>'
    model_body+='<h2>Private models · aggregates only</h2>'+table(['Model','Published-column match','Native qualification'],[[e(r['metric']),e(r['published_match']),e(r['native_gate'])] for r in private['models']])
    page('models','Model cards',model_body,config_paths+['scripts/dashboard/citations.json','crates/zenmetrics-nn/models/jnd_scale.tsv','benchmarks/dashboard/aic2026/private-aggregates.json'])
    cx_paths=[str(x.relative_to(ROOT)) for sub,glob in [('tallies','*.json'),('measured','*.tsv'),('measured','*.meta')] for x in sorted((ROOT/B/sub).glob(glob))]
    cx_body='<p class="notice">Provisional counts and historical one-pair timings on i265. CPU task time and elapsed wall time differ. Samples retain their source and repeat count. The 3840×2160 panel is incomplete. Unmeasured cells are not extrapolated. Neural counts, memory traffic and halo/span annotations remain partial.</p>'
    cx_body+='<h2>Counted operations</h2>'+table(['Metric','flops / original pixel at 1024²','Halo pixels','Counter','Counted source commit'],[[e(m),fmt(r['flops_per_px']),fmt(r['halo_px']),e(t['counter']),e(t['source_commit'])] for m,(t,r) in sorted(tallies.items())])
    missing_tallies=sorted(set(by_name)-set(tallies))
    cx_body+='<p>Not counted: '+e(', '.join(missing_tallies) or 'none')+'. The count is an operation inventory, not elapsed runtime.</p><h2>Measured runtime</h2>'
    timing_rows=[[e(r['metric']),e(r['shape']),r['threads'],fmt(r['cpu_ms']),fmt(r['wall_ms']),r['n'],source_link(r['source'],commit,Path(r['source']).name)] for r in measured]
    for name in ('pieapp','pieapp-pyiqa'):
        for shape in ('1024x1024','3840x2160'):
            if not any(r['metric']==name and r['shape']==shape and r['threads']==1 for r in measured):
                timing_rows.append([name,shape,1,'skipped by owner','skipped by owner','not measured','Owner direction, 2026-10-18'])
    cx_body+=table(['Metric','Shape','Threads','CPU ms/call','Wall ms/call','Repeats','Source'],timing_rows)
    for shape in ['1024x1024','3840x2160']:
        for threads in [1,8]:
            present={r['metric'] for r in measured if r['shape']==shape and r['threads']==threads};missing=sorted(set(by_name)-present)
            cx_body+='<details><summary>Not measured: '+e(shape)+', '+str(threads)+' threads ('+str(len(missing))+')</summary><p>'+e(', '.join(missing) or 'none')+'</p><p>PieAPP missing one-thread timing: skipped by owner (2026-10-18). Other absent cells: no successful committed timing at this shape and thread budget. These files do not establish whether a missing cell was unattempted, refused, or unsupported; consult retained run metadata. HDR batch timings include additional-row PNG decode. MUSIQ at 4K exceeds its documented patch limit.</p></details>'
    wt_rows=[]
    for stem,label in p.TABLES:
        path='scripts/complexity/weights_'+stem+'.json';wt=json.loads((ROOT/path).read_text());sources.add(path)
        for op,values in wt['weights'].items():wt_rows.append([e(label),e(op),json_text(values),source_link(path,commit)])
    cx_body+='<h2>Weight-table comparison</h2><p>Tables assign different per-operation units; compare rank agreement after a fitted scale, never absolute units across tables. Missing operation weights contribute nothing and remain a limit.</p>'
    paper_tables=md_tables(ROOT/'docs/paper/IQA_COMPLEXITY_VS_RUNTIME.md')
    comparison=next(t for t in paper_tables if t[0][0]=='weights')
    cx_body+=table(comparison[0],[[e(v) for v in row] for row in comparison[1:]])
    cx_body+='<p>Not measured: generalisation across hosts or content pairs. Halo padding and 64-row recomputation factors are modelling assumptions. Timing uncertainty and cost-counter uncertainty are distinct.</p><details><summary>All per-operation weights</summary>'+table(['Table','Operation','Precision weights','Source'],wt_rows)+'</details>'
    page('complexity','Counted cost versus measured runtime',cx_body,cx_paths+['docs/paper/IQA_COMPLEXITY_VS_RUNTIME.md']+['scripts/complexity/weights_'+stem+'.json' for stem,_ in p.TABLES])
    agreement=json.loads((ROOT/B/'analysis/aic2026_agreement.json').read_text()); pareto=[]
    for row in agreement['metrics']:
        m=mapping.get(row['column'])
        if m not in tallies: continue
        cost=p.weighted(tallies[m][1],json.loads((ROOT/'scripts/complexity/weights_i265_P_rthroughput.json').read_text())['weights'])[0]
        if cost>0:pareto.append(dict(metric=m,column=row['column'],cost=cost,agreement=row['within_source_srcc'],lo=row['ci_lo'],hi=row['ci_hi']))
    pareto_body='<p class="notice"><strong>Preliminary.</strong> Subjective targets: JPEG WG1 m112107 preliminary public-split JND reconstruction, 5,768 images and 42 sources. Agreement is the mean within-source Spearman of the dataset authors’ published metric columns, polarity aligned; intervals resample sources 2,000 times (seed 20261017). These are not newly requested or final subjective scores. The cost axis counts our mapped implementation at 1024², using measured i265 P-core throughput weights. Pipeline mismatches and DMM’s failed native gate remain disclosed. No per-image AIC values or subjective scores are published here.</p>'
    pareto_body+=plot([(r['cost'],r['agreement'],r['metric']) for r in pareto],'Preliminary cost versus within-source agreement','Weighted counted cost / original pixel (log)','Mean within-source Spearman',front=True)
    pareto_body+=table(['Metric','Published column','Counted cost','Within-source Spearman','95% source interval'],[[e(r['metric']),e(r['column']),fmt(r['cost']),fmt(r['agreement']),fmt(r['lo'])+' – '+fmt(r['hi'])] for r in pareto])
    pareto_body+='<p>Missing: columns without an established implementation mapping or tally. An omitted point does not mean a method is dominated. A point on this front does not establish performance on other datasets.</p>'
    coverage_body, coverage_paths = screenshot_coverage(tallies, measured, rows, commit)
    page('coverage','Screenshot coverage',coverage_body,coverage_paths)
    page('pareto','Preliminary Pareto frontier',pareto_body,[B+'/analysis/aic2026_agreement.json','scripts/complexity/aic2026_columns.json','scripts/complexity/weights_i265_P_rthroughput.json']+cx_paths)
    stat_path='benchmarks/dashboard/aic2026/scatter-example.json';example=json.loads((ROOT/stat_path).read_text())
    stat_body='''<p>The planned report uses <code>zenmetrics-stats</code> (imported as <code>zenstats</code> by zensim). We report within-source and pooled results separately, declare polarity, maps and data versions, and resample source clusters.</p>
    <ul><li>Mohammadi panel: SROCC, PLCC, KROCC; PLCC uses four-parameter logistic predictions; SROCC and KROCC use raw score ranks in the canonical panel. Outlier Ratio with corpus σ and per-stimulus σᵢ where available, normalized PWRC SA-ST AUC, and Z-RMSE.</li><li>Paired Meng–Rosenthal–Rubin z-tests between polarity-aligned metric correlations; source-clustered bootstrap intervals and paired differences; multiplicity declared.</li><li>Pair-class agreement, JND accuracy with declared leave-one-source-out maps (RMSE, excess-error SD, epsilon-insensitive RMSE), and forced-choice agreement/log-likelihood if observer trials are provided.</li></ul>
    <h2>Scatter-v2 geometric bounds</h2><p>These are engineering diagnostics of scatter shape and outliers, not a new inferential statistic. Tie-invariant prediction ranks map to the target quantiles without changing served scores. The diagonal envelope is ±4 × shape MAD, where MAD is 1.4826 times median absolute residual deviation; <code>outside_envelope</code> (out4) counts residuals beyond that band. <code>shape_p99</code> and <code>shape_max</code> divide absolute residual tails by target p99−p1 span.</p><p>Prediction and target coverage/clump use 20 bins, occupied at ≥0.5% of all pairs. Floor/ceiling mass counts values at raw extrema, which alone does not prove clamping. Raw coverage/clump, score range, OLS slope and residual p99/max preserve compression and tails that rank mapping can hide. Zero-spread normalized values are unavailable.</p>
    <h2>Worked public-data example</h2>'''
    stat_body+='<p>'+e(example['description'])+'</p><p>Frozen full-population diagnostics; the chart bins every point, with no point subsampling. This is CID22-A, distinct from the dashboard’s 24-row preview. Fitted-panel statistics are in-sample descriptors, not out-of-fold JND accuracy.</p>'
    stat_body+=density_plot(example)
    stat_body+=table(['Panel field','Measured value'],[[e(k),fmt(v)] for k,v in example['panel'].items()])
    stat_body+=table(['Scatter-v2 field','Measured value'],[[e(section+'.'+k),fmt(v)] for section in ('geo','raw') for k,v in example[section].items()])
    stat_body+='<p>Not measured here: per-stimulus σᵢ panel, observer forced-choice, paired significance against a second metric or JND map cross-validation. Those require the appropriate uncertainties, trials, paired scores or declared folds. Full example diagnostics and density: <a href="scatter-example.json">JSON</a>.</p><h2>AIC data terms</h2><p>'+e(private['data_terms'])+'</p>'
    page('statistics','Statistics and scatter diagnostics',stat_body,[stat_path,'crates/zenmetrics-stats/README.md','crates/zenmetrics-stats/src/scatter.rs','crates/zenmetrics-stats/src/panel.rs'])
    (folder/'scatter-example.json').write_text(json.dumps(example,allow_nan=False,separators=(',',':')))
    page('index','Image-quality evidence for AIC2026','<p>Reproduction results, exact configurations, model cards, counted operations and measured runtime. All pages derive from versioned repository data; each footer pins its source commit.</p><p class="notice">Missing: current AIC2026 subjective scores and their uncertainty; unidentified AIMOS; published pipeline details for A-FINE, WD and DVIFM; private-port redistribution permission. Fixed-factor variants exist, but their dataset parity cannot be checked without published values.</p><div class="grid">'+''.join('<section class="card"><h2><a href="'+n+'.html">'+e(t)+'</a></h2><p>'+e({'coverage':'All 14 complexity-draft and 31 rank-figure entries.','reproduction':'Every published metric column, with missing evidence first.','configurations':'Every CLI name, source identity, ingress and convention.','models':'Public cards, code and weight terms, and private aggregates.','complexity':'Separate counted cost and measured CPU/wall time.','pareto':'Preliminary within-source agreement versus counted cost.','statistics':'The planned statistical panel and a real scatter envelope.'}[n])+'</p></section>' for n,t in NAV[1:])+'</div><p>All assets are served locally; system fonts, no analytics and no CDN dependencies. Evidence links are optional outbound navigation. Licence and scoring limits remain attached to their measurements.</p>',status_paths+['scripts/complexity/aic2026_columns.json',stat_path])
    manifest=dict(schema=1,source_commit=commit,configuration_count=len(configs),reproduction_count=len(rows),pages=[n+'.html' for n,_ in NAV],data_files={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sorted(sources)})
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Exact public configuration export; private records remain aggregate-only.
    (folder/'configurations.json').write_text(json.dumps(configs,allow_nan=False,separators=(',',':')))
    # Link the extension from the existing dashboard without changing its source assets.
    index=output/'index.html'
    index.write_text(index.read_text().replace('<nav aria-label="Dashboard views">','<nav aria-label="Dashboard views"><a href="aic2026/">AIC2026 results</a>',1))
    return manifest


def density_plot(example):
    grid=example['density']; bins=grid['bins']; lo,hi=grid['range']; span=hi-lo
    X=lambda x:70+650*(x-lo)/span
    Y=lambda y:400-350*(y-lo)/span
    radius=4*example['geo']['mad']; parts=['<figure class="plot"><svg role="img" aria-label="CID22 full-population scatter envelope" viewBox="0 0 780 470"><defs><clipPath id="scatter-area"><rect x="70" y="50" width="650" height="350"/></clipPath></defs><title>CID22-A SSIMULACRA2: all '+str(example['n'])+' pairs, target versus rank-mapped prediction</title><g clip-path="url(#scatter-area)">']
    parts.append(f'<path d="M70 {Y(lo+radius):.3f}L720 {Y(hi+radius):.3f}V{Y(hi-radius):.3f}L70 {Y(lo-radius):.3f}Z" fill="var(--accent)" opacity=".12"/>')
    for x,y,n in grid['cells']:
        parts.append(f'<rect x="{70+650*x/bins:.3f}" y="{400-350*(y+1)/bins:.3f}" width="{650/bins:.3f}" height="{350/bins:.3f}" fill="var(--accent)" opacity="{.25+.75*math.log1p(n)/math.log1p(max(c[2] for c in grid["cells"])):.3f}"><title>{n} pairs</title></rect>')
    parts.append('</g><path class="axis" d="M70 40V400H720"/>')
    parts.append(f'<path class="front" d="M70 400L720 50"/><text x="70" y="425">{lo:.4g}</text><text x="700" y="425">{hi:.4g}</text><text x="400" y="458" text-anchor="middle">Human quality target</text><text transform="translate(18 225) rotate(-90)" text-anchor="middle">Rank-mapped predicted quality</text></svg><figcaption>Every pair contributes to a density cell. Orange: diagonal. Shaded: ±4 robust-MAD envelope; out4 = {example["geo"]["out4"]:.6g}.</figcaption></figure>')
    return ''.join(parts)


def screenshot_coverage(tallies, measured, comparisons, commit):
    draft_path='scripts/complexity/pedzisz_2026_draft_values.json'
    ranks_path='benchmarks/dashboard/aic2026/screenshot-ranks.tsv'
    draft=json.loads((ROOT/draft_path).read_text())
    ranks=list(csv.DictReader((ROOT/ranks_path).read_text().splitlines(),delimiter='\t'))
    aliases={'FSIMc':'fsim','FSIM':'fsim-y','HaarPSI':'haarpsi','SSIMULACRA2':'ssim2','Butteraugli-3norm':'butteraugli','PSNR-HVS':'psnrhvs-daala','PSNR-HVS-Y':'psnrhvs-daala','DSSIM':'dssim','VSI':'vsi','GMSD':'gmsd','VMAF-neg v0':'vmaf-neg','CVVDP':'cvvdp'}
    def coverage(name,m):
        counted='yes' if m in tallies else 'pending (aicfix)' if 'DVIFM' in name or name.startswith('FSIM-k') else 'not counted'
        timed='yes' if any(r['metric']==m for r in measured) else 'pending (aicfix)' if 'DVIFM' in name or name.startswith('FSIM-k') else 'not timed'
        column_alias={'Butteraugli_RGB':'Butteraugli-original','CVVDP_RGB':'CVVDP','FSIMc_RGB':'FSIMc','FSIM_Y':'FSIM','VMAF_Y':'VMAF (dataset)','MS-SSIM_Y':'MS-SSIM','SSIM_Y':'SSIM','PSNR_Y':'PSNR-Y'}
        column_name=column_alias.get(name,name)
        comparison=next((r for r in comparisons if r['column']==column_name or r['column'].startswith(column_name+' /')),None)
        reason=''
        if name=='HaarPSI_Y':reason='Counted grayscale route differs from the colour HaarPSI column; no direct column comparison is qualified.'
        if name=='PSNR-HVS-Y':reason='Draft mapping uses the Ponomarenko luma route; the AIC column uses Daala. Combined Daala output cost is not a separately measured luma-only cost.'
        if name=='PSNR_Y':reason='Draft count maps to libvmaf luma; published AIC PSNR-Y uses studio BT.601. Timing coverage does not qualify that ingress match.'
        if name.startswith(('HaarXYB','HaarY-','HaarYCoCg')):
            counted=timed='pending (aicfix)';reason='Method search pending (aicfix); no published per-image values.'
        elif 'DVIFM' in name:reason='Four released-build presets are being counted and timed by aicfix; published-column build differs.'
        elif name.startswith('FSIM-k'):
            counted=timed='pending (aicfix)';reason='Luma-only fixed-factor route pending aicfix; existing fsim-k* rows count the combined output call.'
        elif '-k' in name:reason='Implemented fixed factor; no published per-image values to qualify column comparison.'
        return [e(name),e(m or 'not established'),counted,timed,e((comparison['column']+': '+comparison['status']) if comparison else 'no unambiguous column comparison'),e(reason)]
    body='<p>Both requested screenshots are covered entry by entry. Visible median ranks are aggregates transcribed from the authors’ figure, not recomputed rankings. Counts and timings refer to our implementations; the draft values refer to different reference implementations. Pending aicfix rows are not counted as complete.</p><h2>Pedzisz complexity draft · 14 entries</h2>'
    body+=table(['Screenshot name','Our route','Counted','Timed','Column comparison','Reason / qualification'],[coverage(r['draft_name'],r['zenmetrics']) for r in draft['values']])
    body+='<p>'+e(draft['caveat'])+'</p><h2>AIC2026 rank-distribution figure · 31 names</h2>'
    result=[]
    for r in ranks:
        name=r['metric'];m=aliases.get(name)
        if name.startswith('FSIMc-k'):m='fsim-'+name.split('-')[-1]
        if name.startswith('VSI-k'):m='vsi-'+name.split('-')[-1]
        row=coverage(name,m);row.insert(1,e(r['visible_median_rank']));result.append(row)
    body+=table(['Screenshot name','Visible median rank','Our route','Counted','Timed','Column comparison','Reason / qualification'],result)
    body+='<p>Missing one-thread PieAPP timing is skipped by owner; its existing pyiqa reproduction result is retained. Haar method identification and luma-only FSIM fixed-factor coverage remain pending aicfix until committed evidence arrives.</p>'
    return body,[draft_path,ranks_path,'docs/AIC2026_REPRODUCTION_STATUS.md']
