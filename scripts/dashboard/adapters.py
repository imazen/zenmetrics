"""Schema-specific measurement normalization. No implied units or score conventions."""
import json
import csv
import math
import re
import struct


def missing(value):
    return value in [None,'','NaN','nan','N/A','NA','--','-','NOT_SUPPORTED','ERR','ERROR']


def normalize_status(status):
    raw=str(status or 'missing')
    known={'ok':'ok','measured':'ok','complete':'ok','historical':'historical','unsupported':'unsupported','variant_mismatch':'mismatch','skipped':'unavailable'}
    if raw in known:return known[raw],raw
    if raw.startswith(('failed','error','timeout','oom')):return 'failed',raw
    return 'unavailable',raw


def normalize(path,spec,table,number):
    strict_number=number
    number=lambda value:None if missing(value) else strict_number(value)
    adapter=spec['adapter']
    if adapter=='model-scales':
        lines=[line for line in path.read_text().splitlines() if line and not line.startswith('#')]
        parsed=list(csv.reader(lines,delimiter='\t'))
        if parsed[0]!=['column','jnd']:raise ValueError('model-scale header mismatch')
        for fields in parsed[1:]:
            if not 2<=len(fields)<=3:raise ValueError('model-scale row mismatch')
            yield dict(column=fields[0],jnd=number(fields[1]),note=fields[2] if len(fields)==3 else 'not recorded')
        return
    if adapter=='rounded-analysis':
        text=path.read_text();block,diagnostics=text.split('\n\n',1)
        reader=csv.DictReader(block.splitlines(),delimiter='\t')
        if reader.fieldnames!=spec['columns']:raise ValueError('rounded-analysis schema mismatch')
        for raw in reader:
            yield {(k or 'details'):v for k,v in raw.items()}
        for line in diagnostics.splitlines():
            if not re.fullmatch(r'(st|mt8)_photo_[^:]+: rounds min=\d+ errors=\[.*\]',line):raise ValueError('unknown summary diagnostic')
            yield dict(statistic='source diagnostic',details=line)
        return
    if spec.get('leading_metric_label'):
        parsed=list(csv.reader(path.read_text().splitlines(),delimiter='\t'))
        if parsed[0]!=spec['columns']:raise ValueError('leading-label schema mismatch')
        records=[]
        for fields in parsed[1:]:
            if len(fields)==len(parsed[0])+1 and fields[0]==fields[2]:fields=fields[1:]
            if len(fields)!=len(parsed[0]):raise ValueError('malformed labelled timing row')
            records.append(dict(zip(parsed[0],fields)))
    else:records=list(table(path,spec['columns'])) if path.suffix in ['.tsv','.csv'] else json.loads(path.read_text())
    if adapter=='json-summary':
        def walk(value,key=''):
            if isinstance(value,dict):
                for k,v in value.items():yield from walk(v,key+'.'+k if key else k)
            elif isinstance(value,list):
                for i,v in enumerate(value):yield from walk(v,key+f'[{i}]')
            else:yield key,value
        leaves=list(walk(records));keys={re.sub(r'\[\d+\]','[]',k) for k,v in leaves}
        if keys-set(spec['columns']):raise ValueError(f'{path}: unknown JSON keys {keys-set(spec["columns"])}')
        for key,value in leaves:
            if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
                yield dict(metric=spec.get('metric',path.stem),statistic=key,value=value,status='historical',details=spec.get('note','summary; not a pair delta'))
        return
    for r in records:
        if adapter in ['cpu-wall','cpu-full','nn-speed','gpu-cold','gpu-sweep','crossover','metal-loop','backend-ab']:
            metric=r.get('metric',r.get('crate',spec.get('metric','missing')))
            width=r.get('w',r.get('size_w',r.get('size',None)));height=r.get('h',r.get('size_h',width))
            if width and 'x' in str(width):width,height=str(width).split('x')
            width=number(width);height=number(height)
            base=dict(metric=metric,implementation=spec.get('implementation','zenmetrics'),backend=spec.get('backend','CPU'),host=r.get('host',spec.get('host','missing')),os=spec.get('os','missing'),width=width,height=height,threads=r.get('rayon_threads',r.get('threads',r.get('rten_threads',spec.get('threads','missing')))),engine=r.get('engine',spec.get('engine','native')),mode=r.get('mode','score'),phase=r.get('cold_or_warm','unspecified'),content=r.get('content','missing'),status='historical',details=spec.get('note','Historical measurement; source conditions apply'),n=number(r.get('n',r.get('n_rounds',r.get('n_samples')))),p10_ms=None,p90_ms=None,cpu=spec.get('cpu','missing'),gpu=r.get('device_name',spec.get('gpu','missing')) if r.get('device','gpu')!='cpu' else spec.get('gpu','missing'),simd_tier=spec.get('simd_tier','missing'),parity_status='missing')
            if adapter=='cpu-wall':
                base['ms']=number(r['mean_ms']);base['phase']=r['cold_or_warm']
                if str(r['mode']).startswith('par'):
                    base['threads']=r['mode'][3:];base['mode']='parallel pairs per-pair normalized'
                yield base
            elif adapter=='cpu-full':yield dict(base,ms=number(r['wall_ms']),peak_heap_bytes=number(r['peak_heap_bytes']))
            elif adapter=='nn-speed':yield dict(base,ms=number(r['median_ms']),backend=r.get('device','CUDA' if 'cuda' in r.get('engine','') else 'CPU'),mode=r['mode'],phase='warm' if r['mode']=='warm' else 'not recorded',dtype=r.get('dtype','native'),tf32_conv=r.get('tf32_conv','missing'),library_version=r.get('torch','missing'),min_ms=number(r.get('min_ms')),notes=r.get('note',''))
            elif adapter=='gpu-cold':
                for phase,col in [('cold','cold_total_ms'),('warm','warm_per_call_ms')]:yield dict(base,backend=r['backend'],phase=phase,ms=number(r[col]))
            elif adapter=='gpu-sweep':yield dict(base,backend=r['backend'],ms=number(r['wall_median_ms']),peak_vram_bytes=number(r['peak_vram_bytes']),phase='measured wall median')
            elif adapter=='crossover':
                for backend,phase,col in [('CPU','full','cpu_full_ms'),('CUDA','cold','gpu_cold_total_ms'),('CUDA','warm','gpu_warm_ms')]:yield dict(base,backend=backend,phase=phase,ms=number(r[col]))
            elif adapter=='metal-loop':
                for phase,col in [('cold','cold_per_cand_ms'),('warm reference','warm_per_cand_ms')]:yield dict(base,backend='Metal',mode=r['memory_mode'],phase=phase,ms=number(r[col]))
            elif adapter=='backend-ab':
                for backend,prefix in [('CUDA','cuda'),('Vulkan','vk')]:
                    value=r[prefix+'_med_ms'];yield dict(base,backend=backend,ms=None if missing(value) else number(value),p90_ms=None if missing(r[prefix+'_p90_ms']) else number(r[prefix+'_p90_ms']),status='unavailable' if missing(value) else 'historical',phase='warm')
        elif adapter=='parity-wide':
            for measurement in spec['measurements']:
                ours=number(r[measurement['ours']]);ref=number(r[measurement['reference']]);prefix=measurement.get('delta_prefix','')
                yield dict(metric=measurement.get('metric',r.get('metric',spec.get('metric',path.stem))),pair=r.get('key',r.get('image',r.get('condition_id',r.get('size','missing')))),series=measurement['series'],ours=ours,reference=ref,abs_delta=abs(ours-ref) if ours is not None and ref is not None else None,relative_delta=abs((ours-ref)/ref) if ref else None,jnd_delta=number(r.get(measurement.get('jnd','missing'))),status='historical',reproduction=spec.get('reproduction','not qualified by this panel'),oracle_precision=spec.get('oracle_precision','not recorded'),size=r.get('size','missing'))
        elif adapter=='parity-summary':
            yield dict(metric=r.get('metric',r.get('column',spec.get('metric',path.stem))),pair=r.get('worst_pair','aggregate; no individual pair'),series=spec.get('series',r.get('oracle','summary')),statistic=spec.get('statistic','maximum absolute delta'),abs_delta=number(r.get(spec['absolute'])),jnd_delta=number(r.get(spec.get('jnd','missing'))),n=number(r.get('pairs',r.get('n'))),engine=r.get('engine','missing'),status='historical',oracle_precision='not recorded',reproduction=spec.get('reproduction','not qualified by summary'))
        elif adapter=='hex-residual':
            decode=lambda value,fmt:struct.unpack('>'+('f' if len(value)==8 else 'd'),bytes.fromhex(value))[0]
            yield dict(metric=r['metric'],pair=r['key'],series='published',ours=decode(r['common_score_bits'],'f'),abs_delta=abs(decode(r['common_native_residual_bits'],'d')),jnd_delta=abs(decode(r['common_jnd_residual_bits'],'d')),status='historical',oracle_precision='binary score (width from hex) and f64 residuals',reproduction='residual observation; not a parity qualification')
        elif adapter=='scales':
            yield {k:(number(v) if k in spec.get('numeric',[]) and not missing(v) else v) for k,v in r.items()}
        elif adapter=='normalized':
            yield {k:(number(v) if k in spec.get('numeric',[]) else v) for k,v in r.items()}
        else:raise ValueError('unknown adapter '+adapter)
