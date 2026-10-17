#!/usr/bin/env python3
"""Validate competition rows, split compact summaries, draw standalone SVG/HTML."""
import argparse, collections, html, json, math, pathlib, statistics
P=pathlib.Path

def plot_svg(figure):
    import io
    import matplotlib.pyplot as plt
    out=io.StringIO()
    figure.savefig(out,format='svg',bbox_inches='tight',metadata={'Date':None})
    plt.close(figure)
    return out.getvalue()

def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'iqa-compete','font.size':9})
    return plt

def svg_bars(title,rows):
    plt=plotting()
    figure,axis=plt.subplots(figsize=(16,max(2,.25*len(rows)+1)))
    labels=[f"{r['metric_name']} / {r['impl']} ({r.get('engine','native')}/{r['device']}) / {r['threads']}T / {r['mode']}" for r in rows]
    values=[r['median_ns']/1e6 for r in rows]
    colors=['#cf702c' if r['status']=='variant_mismatch' else ('#2166ac' if r['impl']=='zenmetrics' else '#4f9e79') for r in rows]
    bars=axis.barh(range(len(rows)),values,color=colors)
    axis.set_yticks(range(len(rows)),labels,fontsize=8)
    axis.invert_yaxis();axis.set_xlabel('Per-call median (ms)');axis.set_title(title)
    axis.bar_label(bars,labels=[f'{v:.4g}' for v in values],padding=3,fontsize=7)
    axis.margins(x=.13);axis.grid(axis='x',alpha=.2);axis.set_axisbelow(True)
    return plot_svg(figure)

def svg_lines(title,rows):
    plt=plotting();groups=collections.defaultdict(list)
    for r in rows:groups[(r['metric_name'],r['impl'],r.get('engine','native'),r['device'],r['threads'],r['mode'],r['status'])].append(r)
    figure,axis=plt.subplots(figsize=(12,max(4,.16*len(groups)+1)))
    for key,values in sorted(groups.items()):
        values=sorted(values,key=lambda r:r['width']*r['height'])
        axis.plot([r['width']*r['height']/1e6 for r in values],[r['median_ns']/1e6 for r in values],marker='o',markersize=3,label=' / '.join(map(str,key)))
    axis.set_xlabel('Image size (megapixels)');axis.set_ylabel('Per-call median (ms)')
    axis.set_title(title);axis.grid(alpha=.2);axis.set_ylim(bottom=0);axis.set_xlim(left=0)
    axis.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=6)
    return plot_svg(figure)

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
    fields=['host','metric_name','impl','impl_binary_sha','impl_version','engine','adapter_sha256','device','threads','mode','size_name','width','height','pair_sha256','n','median_ns','p10_ns','p90_ns','mean_ns','score','ref_score','abs_delta','relative_delta','status','comparison_status','parity_abs_bound','parity_rel_bound','parity_source','reference_binary_sha','reference_pair_sha256','reference_threads','ort_intra_threads_configured','ort_inter_threads_configured','ort_shared_library_sha256','ort_disable_telemetry','diagnostic_source_status','reproduction_required','load_before','load_after','resource','resource_memory_method']
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
        category=r['status']
        if r.get('comparison_status','').startswith('unavailable:'):category+='-comparison-unavailable'
        if r.get('reproduction_required'):category+='-reproduction-required'
        panels[(r['host'],r['size_name'],r['mode'],category)].append(r);families[(r['host'],r['metric_family'],category)].append(r)
    links=[]
    for prefix,groups,draw in [('bars',panels,svg_bars),('lines',families,svg_lines)]:
        for key,values in sorted(groups.items()):
            name=prefix+'-'+'-'.join(key).replace('+','_')+'.svg';(a.output/name).write_text(draw(' / '.join(key),sorted(values,key=lambda r:(r['metric_name'],r['threads'],r['impl']))));links.append(name)
    status=collections.Counter(r['status'] for r in rows)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>IQA compute comparison</title><h1>IQA compute comparison</h1><p>Descriptive observations, no confidence intervals. Orange bars mark variant mismatch; unsupported/skipped cells remain in the summaries. Missing CPU comparisons and pending mismatch reproductions have separate panels.</p><pre>'+html.escape(json.dumps(status,indent=2))+'</pre>'+''.join(f'<p><a href="{html.escape(n)}">{html.escape(n)}</a></p>' for n in links))
    print(json.dumps(dict(rows=len(rows),statuses=status,panels=len(links),renderer='matplotlib',renderer_version=__import__('matplotlib').__version__)))
if __name__=='__main__':main()
