#!/usr/bin/env python3
"""Validate competition rows, split compact summaries, draw standalone SVG/HTML."""
import argparse, collections, html, json, math, pathlib, statistics
P=pathlib.Path

def svg_bars(title,rows):
    width=1200;height=70+26*len(rows);maximum=max((r['median_ns'] for r in rows),default=1)
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="12"><text x="12" y="22">{html.escape(title)} — per-call median, milliseconds</text>']
    for i,r in enumerate(rows):
        y=48+26*i;label=f"{r['metric_name']} / {r['impl']} / {r['threads']}T / {r['mode']}";color='#cf702c' if r['status']=='variant_mismatch' else ('#2166ac' if r['impl']=='zenmetrics' else '#4f9e79')
        out.append(f'<text x="12" y="{y+13}">{html.escape(label)}</text><rect x="410" y="{y}" width="{r["median_ns"]/maximum*660:.3f}" height="18" fill="{color}"/><text x="1080" y="{y+13}">{r["median_ns"]/1e6:.4g}</text>')
    return ''.join(out)+'</g></svg>'
def svg_lines(title,rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['metric_name'],r['impl'],r['threads'],r['mode'],r['status'])].append(r)
    xmax=max((r['width']*r['height']/1e6 for r in rows),default=1);ymax=max((r['median_ns']/1e6 for r in rows),default=1)
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="540"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="12">',f'<text x="12" y="22">{html.escape(title)} — x: megapixels; y: milliseconds/call</text><path d="M60 40V480H880" fill="none" stroke="black"/>']
    for i,(key,rs) in enumerate(sorted(groups.items())):
        color=['#2166ac','#4f9e79','#cf702c','#9c58a1','#6b6b6b'][i%5];points=' '.join(f'{60+820*r["width"]*r["height"]/1e6/xmax:.3f},{480-420*r["median_ns"]/1e6/ymax:.3f}' for r in sorted(rs,key=lambda r:r['width']*r['height']))
        out.append(f'<polyline points="{points}" fill="none" stroke="{color}"/><text x="900" y="{45+17*i}" fill="{color}">{html.escape(str(key))}</text>')
    for n in range(6):out.append(f'<text x="{60+164*n}" y="505">{xmax*n/5:.2g}</text><text x="2" y="{480-84*n}">{ymax*n/5:.3g}</text>')
    return ''.join(out)+'</g></svg>'
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw',nargs='+',type=P);p.add_argument('--output',type=P,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    rows=[];seen=set()
    for path in a.raw:
        for line in path.read_text().splitlines():
            r=json.loads(line);key=tuple(r.get(k) for k in ['host','metric_name','impl','device','threads','mode','size_name'])
            if key in seen:raise ValueError(f'duplicate cell {key}')
            seen.add(key)
            if r['status'] in ['ok','variant_mismatch']:
                for k in ['median_ns','mean_ns','p10_ns','p90_ns']:
                    if not isinstance(r[k],(int,float)) or not math.isfinite(r[k]) or r[k]<=0:raise ValueError((key,k))
                if r['n']<10 and r.get('timed_seconds',0)<3:raise ValueError((key,'insufficient sampling'))
            rows.append(r)
    fields=['host','metric_name','impl','impl_binary_sha','device','threads','mode','size_name','width','height','pair_sha256','n','median_ns','p10_ns','p90_ns','mean_ns','score','ref_score','abs_delta','relative_delta','status','load_before','load_after','resource']
    byhost=collections.defaultdict(list)
    for r in rows:byhost[r['host']].append({k:r.get(k) for k in fields})
    for host,values in byhost.items():
        chunk=[];index=0
        for r in values:
            candidate=chunk+[r]
            if len((json.dumps(candidate,indent=2)+'\n').encode())>29000:
                (a.output/f'{host}-{index:03}.json').write_text(json.dumps(chunk,indent=2)+'\n');chunk=[];index+=1
            chunk.append(r)
        if chunk:(a.output/f'{host}-{index:03}.json').write_text(json.dumps(chunk,indent=2)+'\n')
    panels=collections.defaultdict(list);families=collections.defaultdict(list)
    for r in rows:
        if r['status'] not in ['ok','variant_mismatch']:continue
        panels[(r['host'],r['size_name'],r['mode'],r['status'])].append(r);families[(r['host'],r['metric_family'],r['status'])].append(r)
    links=[]
    for prefix,groups,draw in [('bars',panels,svg_bars),('lines',families,svg_lines)]:
        for key,values in sorted(groups.items()):
            name=prefix+'-'+'-'.join(key).replace('+','_')+'.svg';(a.output/name).write_text(draw(' / '.join(key),sorted(values,key=lambda r:(r['metric_name'],r['threads'],r['impl']))));links.append(name)
    status=collections.Counter(r['status'] for r in rows)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>IQA compute comparison</title><h1>IQA compute comparison</h1><p>Descriptive observations, no confidence intervals. Orange marks variant mismatch; unsupported/skipped cells remain in the summaries.</p><pre>'+html.escape(json.dumps(status,indent=2))+'</pre>'+''.join(f'<p><a href="{html.escape(n)}">{html.escape(n)}</a></p>' for n in links))
    print(json.dumps(dict(rows=len(rows),statuses=status,panels=len(links))))
if __name__=='__main__':main()
