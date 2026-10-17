#!/usr/bin/env python3
"""Strict registry adapters and dependency-free static dashboard assembly."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tomllib
from adapters import normalize, normalize_status, strict_json
from catalog import rows as catalog_rows
from model_cards import cards as model_cards, exports as card_exports

ROOT = Path(__file__).resolve().parents[2]
DATA_SUFFIXES = {'.csv', '.tsv', '.json', '.jsonl', '.parquet', '.meta'}


def inventory(root=ROOT):
    return sorted(p.relative_to(root).as_posix() for p in (root / 'benchmarks').rglob('*')
                  if p.is_file() and p.name != 'classification.tsv' and (p.suffix in DATA_SUFFIXES or p.name.endswith('.pointer.md')))


def number(value):
    if value is None or value == '':
        return None
    if value in ['NaN','nan','N/A','NA','--']:
        raise ValueError(f'non-finite measurement: {value}')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'non-finite measurement: {value}')
    return result


def table(path, columns):
    lines = [line for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]
    reader = csv.DictReader(lines, delimiter='\t' if path.suffix == '.tsv' else ',')
    if set(reader.fieldnames or []) != set(columns) or len(reader.fieldnames or []) != len(columns):
        raise ValueError(f'{path.name}: schema mismatch: {reader.fieldnames}')
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f'{path}: malformed row')
        yield row


def safe_text(text):
    # Publish selected evidence fields only; historical metadata can contain local paths.
    return re.sub(r'(?:/(?:home|mnt|Users|tmp)/[^\s`"<>]*|~[^\s`"<>]*)', '[local path]', text)


def provenance(path, spec):
    paths = [ROOT / p for p in spec.get('metadata', [])]
    paths += [path.with_suffix('.meta'), path.with_suffix('.meta.json')]
    paths += [path.parent / 'README.md']
    paths += list(path.parents[0].glob('run.meta'))
    if path.parent.parent.name.startswith('cpu_metrics_1t8t'): paths += [path.parent.parent/'run.meta']
    existing = list(dict.fromkeys(p for p in paths if p.is_file()))
    evidence = '\n'.join(p.read_text(errors='strict') for p in existing)
    commits = re.findall(r'(?<![A-Za-z0-9_])(?:build_commit|repo_commit[^:\n]*|commit_sha|base_commit|git_sha|git_commit|^commit)[\"\s]*[:=][\"\s]*([0-9a-f]{7,40})\b', evidence,re.MULTILINE|re.IGNORECASE)
    commands = re.findall(r'^(?:run|command|reproduce)\s*[:=]\s*(.+)$', evidence, re.MULTILINE)
    recorded_date=None;recorded_host=None
    for metadata in existing:
        if metadata.suffix=='.json':
            obj=json.loads(metadata.read_text())
            if isinstance(obj,dict):
                if isinstance(obj.get('command'),str):commands.insert(0,obj['command'])
                recorded_date=obj.get('generated_date_utc',recorded_date)
                if obj.get('host') and valid_host(obj['host']):recorded_host=obj['host']
    host=spec.get('host',recorded_host or 'missing')
    if host=='missing':
        found=re.search(r'^host\s*[:=]\s*([a-z0-9]+)',evidence,re.MULTILINE|re.IGNORECASE)
        if found and valid_host(found[1]):host=found[1]
        elif '7950X' in evidence:host='r7950x'
        elif '7900X' in evidence:host='r7900x'
        elif 'Apple M4' in evidence:host='mac'
    if host!='missing' and not valid_host(host):raise ValueError('noncanonical registry host')
    os_name=spec.get('os','missing')
    os_match=re.search(r'^os\s*[:=]\s*(.+)',evidence,re.MULTILINE|re.IGNORECASE)
    if os_name=='missing' and os_match:
        line=os_match[1].lower();os_name='Linux (WSL2)' if 'wsl' in line else 'macOS' if 'darwin' in line or 'macos' in line else 'Linux' if 'linux' in line else 'Windows' if 'windows' in line else 'missing'
    dates = re.findall(r'20\d\d-\d\d-\d\d', str(path))
    return dict(source=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                commit=spec.get('commit',commits[0] if commits else 'missing'), host=host, os=os_name,
                date=recorded_date or (dates[0] if dates else 'missing'), command=safe_text(spec.get('command', commands[0] if commands else 'missing')),
                metadata=[p.relative_to(ROOT).as_posix() for p in existing],
                note=spec.get('note', 'Historical observation; consult source conditions.'))


def adapt(path, spec):
    adapter = spec['adapter']
    if adapter == 'competition-meta':
        text = path.read_text()
        start = 0 if text.startswith('{') else text.index('\n{') + 1
        metadata = json.loads(text[start:])
        for field in ['identity', 'build_commit', 'command', 'host', 'hardware', 'toolchains', 'competitors']:
            if field not in metadata:
                raise ValueError(f'missing competition metadata: {field}')
        if metadata['build_commit'] != metadata['identity']['commit']:
            raise ValueError('competition metadata commit differs')
        return
    if adapter == 'libjxl-samples':
        import statistics
        for row in table(path, spec['columns']):
            samples=sorted(number(v)/1e6 for v in row['samples_ns'].rstrip(',').split(','))
            if len(samples)<2 or any(v<=0 for v in samples):raise ValueError('invalid libjxl samples')
            def quantile(p):
                index=(len(samples)-1)*p;lo=int(index);hi=min(lo+1,len(samples)-1)
                return samples[lo]+(samples[hi]-samples[lo])*(index-lo)
            yield dict(metric='butteraugli',implementation='libjxl v0.12',backend='CPU',
                       host=spec['host'],os=spec['os'],threads=int(row['threads']),mode=row['mode'],
                       width=int(row['w']),height=int(row['h']),size=row['size'],ms=statistics.median(samples),
                       p10_ms=quantile(.1),p90_ms=quantile(.9),n=len(samples),status='ok')
    elif adapter == 'loop':
        for row in table(path, spec['columns']):
            for mode, col in [('cold', 'oneshot_ms'), ('warm reference', 'warm_per_cand_ms')]:
                yield dict(metric=row['metric'], implementation='zenmetrics', backend='CPU',
                           host=spec['host'], os='macOS', threads=spec['threads'], mode=mode,
                           width=int(row['w']), height=int(row['h']), ms=number(row[col]), status='ok')
    elif adapter == 'parity':
        for row in table(path, spec['columns']):
            for series in ['oracle','published']:
                if series not in row: continue
                ours,ref=number(row['ours']),number(row[series])
                yield dict(metric=spec['metric'],pair=row['key'],series=series,ours=ours,reference=ref,
                           abs_delta=number(row.get(series+'_error')) if ours is not None and ref is not None else None,relative_delta=abs((ours-ref)/ref) if ref else None,
                           jnd_delta=number(row.get(series+'_jnd_error')),size='missing',status='historical' if ours is not None and ref is not None else 'failed',details=safe_text(row.get(series+'_error','')) if ours is None or ref is None else 'observed delta',
                           oracle_precision=spec.get('oracle_precision','not recorded'),
                           reproduction=spec.get('reproduction','not qualified by this panel'))
    elif adapter == 'quality':
        for row in table(path, spec['columns']):
            yield dict(metric=f"margarine/{row['arm']}/{row.get('norm',row.get('pooling','unspecified'))}",experiment=path.stem, dataset=row['dataset'],
                       scope=row['scope'], subset=row['key'], n=int(row['n']), status=row['status'],
                       srocc=number(row['srocc']), plcc=number(row['plcc']), krocc=number(row['krocc']),
                       ci='missing', mos=None, score=None)
    elif adapter == 'quality-points':
        for row in table(path, spec['columns']):
            yield dict(metric=row['metric'], dataset=row['dataset'], pair=row['pair'],
                       mos=number(row['mos']), score=number(row['score']), status='measured',
                       scope='per-image', subset='all', ci='missing')
    elif adapter == 'competition':
        rows = strict_json(path.read_text()) if path.suffix == '.json' else [strict_json(l) for l in path.read_text().splitlines()]
        if not isinstance(rows, list):
            raise ValueError('competition input must be a row array or JSONL')
        for row in rows:
            extra = set(row) - set(spec['columns'])
            if extra:
                raise ValueError(f'{path}: unknown competition columns: {extra}')
            for field in ['host', 'metric_name', 'impl', 'device', 'threads', 'mode', 'width', 'height', 'status']:
                if field not in row:
                    raise ValueError(f'missing competition field: {field}')
            for field in ['width', 'height', 'threads']:
                if not isinstance(row[field], int) or row[field] <= 0:
                    raise ValueError(f'invalid competition {field}')
            if row.get('median_ns') is not None and number(row['median_ns']) <= 0:
                raise ValueError('nonpositive timing')
            if row['status'] == 'ok' and row.get('median_ns') is None:
                raise ValueError('successful timing row has no measurement')
            result = dict(metric=row['metric_name'], implementation=row['impl'], implementation_version=row.get('impl_version'), backend=row['device'],
                          host=row['host'], os=row.get('os', 'missing'), threads=row['threads'], mode=row['mode'],
                          width=row['width'], height=row['height'], ms=None if row.get('median_ns') is None else number(row['median_ns'])/1e6,
                          status=row['status'], size=row.get('size_name', f"{row['width']}×{row['height']}"), engine=row.get('engine', 'missing'),
                          comparison_status=row.get('comparison_status', 'missing'),
                          reproduction_required=row.get('reproduction_required', False),
                          ours=number(row.get('score')), reference=number(row.get('ref_score')),
                          abs_delta=number(row.get('abs_delta')), relative_delta=number(row.get('relative_delta')),
                          pair=row.get('pair_sha256','missing'),series='competition reference',
                          p10_ms=number(row.get('p10_ns'))/1e6 if row.get('p10_ns') is not None else None,
                          p90_ms=number(row.get('p90_ns'))/1e6 if row.get('p90_ns') is not None else None,
                          n=row.get('n'),notes=safe_text(row.get('notes','')),sample_gate_clean=row.get('sample_gate_clean'),impl_binary_sha=row.get('impl_binary_sha'),input_sha256=row.get('input_sha256'),peak_rss_gib=(row.get('resource') or {}).get('peak_rss_gib'),
                          cpu=row.get('cpu_model'),gpu=row.get('gpu'),simd_tier=row.get('simd_tier'),
                          parity_status=row.get('parity_status','missing'),timing_quality=row.get('timing_quality'),contention_peak_cores=row.get('contention_peak_cores'),coverage_first=row.get('coverage_first'),required_samples=row.get('required_samples'),parity_checked=row.get('parity_checked'))
            yield result
    else:
        yield from normalize(path,spec,table,number)


def catalog():
    yield from catalog_rows()


def valid_host(host):
    return host in {'dev','tower','mac'} or bool(re.fullmatch(r'(?:r\d{4,5}(?:x3d|xt|x|g|ge|hx|h|u)?|i\d{3,5}(?:kf|k|f|hx|h|u)?)',str(host)))


def scan_output(output):
    import importlib.util
    file=ROOT/'scripts/ci/check_hygiene.py'
    module_spec=importlib.util.spec_from_file_location('dashboard_hygiene',file)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    patterns=module.load_patterns()
    if not patterns:raise ValueError('hygiene patterns missing')
    for p in output.rglob('*'):
        if not p.is_file():continue
        text=p.read_text()
        if any(rx.search(text) for _,rx in patterns):raise ValueError(f'hygiene class in output: {p.name}')
        if re.search(r'/(?:home|mnt|Users)/',text):raise ValueError(f'local path in output: {p.name}')


def validate_inventory(root,registered):
    records=list(table(root/'benchmarks/dashboard/classification.tsv',['path','classification','reason']))
    classified={r['path']:r for r in records}
    if len(classified)!=len(records):raise ValueError('duplicate classifications')
    unregistered=set(inventory(root))-registered
    unknown=unregistered-set(classified)
    if unknown:raise ValueError('unclassified benchmark uploads: '+str(sorted(unknown)))
    stale=set(classified)-set(inventory(root))
    if stale:raise ValueError('stale classifications: '+str(sorted(stale)))
    if any(not r['reason'] or r['classification']!='ignored' for r in classified.values()):raise ValueError('invalid classification')
    return unregistered


# Explicit historical family joins: these do not qualify the current implementation.
HISTORICAL_QUALITY_FAMILY = {
    'score_butter_max':'butteraugli','score_butter_p3':'butteraugli',
    'score_cvvdp':'cvvdp','score_dssim':'dssim','score_hdr_vdp_2':'hdrvdp',
    'score_hdr_vdp_3':'hdrvdp3','score_iw_ssim':'iwssim','score_ms_ssim':'msssim',
    'score_psnr_y':'psnr-y','score_ssim':'ssim','score_ssim2_gpu':'ssim2-gpu',
    'score_ssim2_paper':'ssim2','score_vmaf_neg':'vmaf-neg',
    **{name:'zensim' for name in ['score_zensim','score_zensim_v0_16','score_zensim_v0_17',
       'score_zensim_v0_17_c14_s42','score_zensim_v0_17_c14_s7','score_zensim_v0_26',
       'score_zensim_v0_31','score_zensim_v0_38','score_v0_2_linear']}}


def build(output):
    registry = tomllib.loads((ROOT/'benchmarks/dashboard/datasets.toml').read_text())
    families = {k: [] for k in ['speed', 'agreement', 'quality', 'catalog', 'coverage', 'scales', 'complexity']}
    sources = {}; registered = set(); matches=[];source_counts={};source_mode_counts={}
    aliases={c:r['metric'] for r in catalog_rows() for c in r['columns'].split(';') if c};aliases['butter']='butteraugli';aliases['ms-ssim']='msssim'
    for spec in registry['dataset']:
        paths = sorted(ROOT.glob(spec['glob']))
        if not paths and not spec.get('allow_absent', False):
            raise ValueError(f"registered glob has no files: {spec['glob']}")
        matches.append(dict(glob=spec['glob'],allow_absent=spec.get('allow_absent',False),matches=len(paths)))
        for path in paths:
            key = path.relative_to(ROOT).as_posix()
            if key in registered:
                raise ValueError(f'duplicate registration: {key}')
            registered.add(key); sources[key] = provenance(path, spec)
            source_counts[key]=0;source_mode_counts[key]={}
            runtime=dict(spec,host=sources[key]['host'],os=sources[key]['os'])
            for row in adapt(path, runtime):
                row['source'] = key
                if spec.get('comparison_group'):row['comparison_group']=spec['comparison_group']
                if spec['family']=='quality' and 'mos' in row:
                    row['human_label']=row.pop('mos');row.setdefault('target','MOS')
                if spec['family']=='speed' and row.get('metric') in aliases:
                    row['metric_recorded']=row['metric'];row['metric']=aliases[row['metric']]
                if 'status' in row:
                    normalized,detail=normalize_status(row['status']);row['status']=normalized;row['details']=row.get('details',detail)
                if spec['family']=='speed' and row.get('ms') is None and row['status'] in ['ok','historical']:
                    row['status']='unavailable';row['details']='Missing timing or recorded refusal; '+row.get('details','')
                if sources[key]['host']=='missing' and row.get('host') and valid_host(row['host']):sources[key]['host']=row['host']
                if 'host' in row and row['host']!='missing' and not valid_host(row['host']):raise ValueError('noncanonical data host')
                source_counts[key]+=1
                if 'mode' in row:
                    mode=str(row['mode']);source_mode_counts[key][mode]=source_mode_counts[key].get(mode,0)+1
                families[spec['family']].append(row)
                if spec['adapter']=='competition' and row.get('abs_delta') is not None:
                    families['agreement'].append(row.copy())
                if spec.get('also_agreement'):
                    for raw in table(path,spec['columns']):
                        ours=raw['cuda_score'];ref=raw['vk_score']
                        if ours not in ['NaN','nan',''] and ref not in ['NaN','nan','']:
                            a=number(ours);b=number(ref)
                            families['agreement'].append(dict(metric='ssim2-gpu',pair=raw['size'],size=raw['size'],series='CUDA vs Vulkan',ours=a,reference=b,abs_delta=abs(a-b),relative_delta=abs((a-b)/b) if b else None,source=key,status='historical'))
                    # This linked panel is emitted once per file, below the row loop.
                    spec['also_agreement']=False
            if spec.get('also_pairs'):
                for raw in strict_json(path.read_text()):
                    for series in ['oracle','published']:
                        if raw.get(series) is not None:
                            a=number(raw['ours']);b=number(raw[series])
                            families['agreement'].append(dict(metric=spec['metric'],pair=raw['key'],series=series,ours=a,reference=b,abs_delta=abs(a-b),relative_delta=abs((a-b)/b) if b else None,source=key,status='historical'))
    families['catalog'] = list(catalog())
    families['models'] = list(model_cards())
    for card in families['models']:
        names={card['metric'],*card['columns'].split(';')}
        match=lambda row:row.get('metric') in names or str(row.get('metric','')).replace('_','-')==card['metric']
        card['speed_observations']=sum(match(row) for row in families['speed'] if row.get('ms') is not None)
        card['agreement_observations']=sum(match(row) for row in families['agreement'])
        card['quality_panels']=[{k:row.get(k) for k in ['metric','dataset','scope','target','subset','n','srocc','signed_srocc','srocc_lo','srocc_hi','ci','source']}
                                for row in families['quality'] if (match(row) or HISTORICAL_QUALITY_FAMILY.get(row.get('metric'))==card['metric']) and row.get('srocc') is not None
                                and any(name in str(row.get('dataset','')).lower() for name in ['aic','cid22'])]

    output.mkdir(parents=True, exist_ok=True)
    for path in (ROOT/'web/dashboard').iterdir():
        if path.is_file(): shutil.copyfile(path, output/path.name)
    data = output/'data'; data.mkdir(exist_ok=True)
    for family, rows in families.items():
        (data/f'{family}.json').write_text(json.dumps(rows, allow_nan=False, separators=(',', ':')))
    for name,content in card_exports(families['models']).items():(data/name).write_text(content)
    shutil.copyfile(ROOT/'scripts/dashboard/favorites.json', data/'favorites.json')
    (data/'unimplemented.json').write_text(json.dumps(unimplemented_methods(), separators=(',', ':')))
    unregistered=validate_inventory(ROOT,registered)
    manifest = dict(schema=2, counts={k:len(v) for k,v in families.items()}, sources=sources,
                    quality_correlation_count=sum(r.get('srocc') is not None for r in families['quality']),
                    quality_point_count=sum(r.get('human_label') is not None and r.get('score') is not None for r in families['quality']),
                    registered=sorted(registered),unregistered_count=len(unregistered),
                    glob_matches=matches,source_counts=source_counts,source_mode_counts=source_mode_counts)
    (data/'manifest.json').write_text(json.dumps(manifest, indent=2))
    scan_output(output)
    print(json.dumps(manifest['counts']))
    return manifest


def unimplemented_methods(root=ROOT):
    # Planned, held or declined methods that the CLI does not implement; each cites its source line.
    methods = tomllib.loads((root/'benchmarks/dashboard/unimplemented.toml').read_text())['method']
    for m in methods:
        if set(m) != {'name', 'status', 'source'}: raise ValueError(f'unimplemented method fields: {m}')
        path, line = m['source'].rsplit(':', 1)
        lines = (root/path).read_text().splitlines()
        if not 1 <= int(line) <= len(lines): raise ValueError(f'unimplemented source line missing: {m}')
        m['url'] = 'https://github.com/imazen/zenmetrics/blob/master/' + path + '#L' + line
    return methods


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
