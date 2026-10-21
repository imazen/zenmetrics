#!/usr/bin/env python3
"""One evidence model for the dashboard, README metrics table and full cards.

Reads repository metadata only. It never imports reference packages or weights.
Bibliographic facts are frozen from publisher/arXiv metadata in citations.json.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import re
import tomllib
from catalog import ROOT, rows as catalog_rows, md_tables, plain, function, model_paths
import repro
from aic_semantics import fixed_factor_contract, description as variant_description

SITE='https://imazen.github.io/zenmetrics/'
REPO='https://github.com/imazen/zenmetrics/blob/master/'
MISSING='not recorded'


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    return plain(value) if isinstance(value,str) else value


def ledger():
    result={}
    for table in md_tables(ROOT/'docs/METRIC_PROVENANCE.md'):
        if 'reference impl (validation oracle)' not in table[0]:continue
        previous=[]
        for raw in table[1:]:
            fields=[previous[i] if f.strip() in ['"','same','“'] and len(previous)>i else f for i,f in enumerate(raw)]
            previous=fields
            for metric in re.findall(r'`([^`]+)`',raw[0]):result[metric]=fields
    return result


def base_name(metric, names):
    if metric in names:return metric
    candidates=[name for name in names if metric.startswith(name+'-')]
    return max(candidates,key=len) if candidates else None


def explicit_aic_inputs(metric):
    if metric in ('hdrvdp-aic2026','hdrvdp3-aic2026'):
        v3=metric=='hdrvdp3-aic2026'
        return {'format':'packed sRGB8; f64 conversion to absolute nits',
                'sdr_peak_nits':200,'ppd':37.842505866 if v3 else 37.84,
                'code_bits_before_srgb_eotf':10 if v3 else 0,
                'code_rounding':'nearest, ties away from zero when enabled',
                'encoding':'D65 XYZ from linear sRGB' if v3 else 'RGB BT.709',
                'xyz_matrix':[[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]] if v3 else None,
                'task':'quality','emission':'OLED' if v3 else 'HDR-VDP 2 RGB BT.709 reference defaults',
                'age':24 if v3 else 'reference defaults','surround':'none' if v3 else 'reference defaults',
                'overrides':['--hdrvdp-sdr-peak','--hdrvdp-sdr-ppd','--hdrvdp-sdr-code-bits'],
                'v2_strict_arithmetic':'--hdrvdp-sdr-strict; default false',
                'outputs':['hdrvdp3_aic2026_q','hdrvdp3_aic2026_jod'] if v3 else ['hdrvdp_aic2026_q'],
                'source':'crates/zenmetrics-cli/src/metrics/hdrvdp_aic.rs'}
    if metric=='cambi-aic2026':
        return {'format':'packed sRGB8; distorted image only','resize':'crop to even width and height',
                'luma':'t=(8414R+16519G+3208B+(32<<14)+(1<<8))>>9; Y=(t+32)>>6',
                'kernel':'CAMBI 8-bit luma; existing defaults','source':'crates/zenmetrics-cli/src/metrics/vmaf.rs'}
    if metric=='proposal-butteraugli':
        return {'format':'packed sRGB8','native_output':'butteraugli pnorm3, existing CPU defaults',
                'output_map':'max(0.00459792*N, 2.0938201*N - 1.1033102)',
                'source':'crates/zenmetrics-cli/src/metrics/mod.rs'}
    return None


def bibtex(citation):
    if not citation:return ''
    key=re.sub(r'[^a-zA-Z0-9]','',citation['key'])+citation['year']
    fields={'title':citation['title'],'author':' and '.join(citation['authors']),
            'year':citation['year'],'note':citation['venue'],'url':citation['url']}
    if 'doi.org/' in citation['url']:fields['doi']=citation['url'].split('doi.org/')[1]
    else:fields.update(eprint=citation['url'].split('/abs/')[-1],archivePrefix='arXiv')
    def literal(v):return v.replace('\\','\\textbackslash{}').replace('{','\\{').replace('}','\\}')
    return '@misc{'+key+',\n'+',\n'.join('  '+k+' = {'+literal(str(v))+'}' for k,v in fields.items())+'\n}\n'


def cards():
    citations=json.loads((ROOT/'scripts/dashboard/citations.json').read_text())
    by_metric={metric:dict(c,key=key) for key,c in citations.items() for metric in c['metrics'] if not metric.startswith('--')}
    nn=[]
    for path in model_paths():
        nn.append((path,tomllib.loads(path.read_text())))
    entries=ledger()
    source=(ROOT/'crates/zenmetrics-cli/src/metrics/mod.rs').read_text()
    kinds=dict(re.findall(r'MetricKind::(\w+) => "([^"]+)"',function(source,'name')))
    contracts={kinds.get(kind,kind):plain(re.sub(r'^    /// ?', '', docs,flags=re.M).strip())
               for docs,kind in re.findall(r'((?:    ///[^\n]*\n)+)(?:    #\[[^\n]*\]\n)*    (\w+),',source)}
    # Keep a linked section rather than treating all divergences as one metric's facts.
    divergences=(ROOT/'DIVERGENCES.md').read_text()
    sections=re.split(r'(?m)^(?=## )',divergences)
    for row in catalog_rows():
        metric=row['metric'];columns=set(row['columns'].split(';'))
        item=next(((p,c) for p,c in nn if columns&set(c.get('columns',[]))),None)
        if item is None:
            # A few original cards predate the columns field. An exact named
            # card still owns its reference and results; aliases do not inherit.
            item=next(((p,c) for p,c in nn if c.get('name')==metric),None)
        path,model=item if item else (None,{})
        entry=entries.get(base_name(metric,entries),[])
        cite=by_metric.get(model.get('name')) if model else None
        if not cite: cite=by_metric.get(base_name(metric,by_metric))
        if cite:cite=dict(cite,bibtex=bibtex(cite))
        parity=dict(model.get('parity',{}))
        if path:
            raw=path.read_text().split('[parity]',1)[-1]
            comments='\n'.join(line.lstrip('# ').strip() for line in raw.splitlines() if line.lstrip().startswith('#'))
            if comments:parity['recorded_source_comments']=comments

        status='Reported results' if parity.get('results') or parity.get('status') or any('reported_worst' in k or k.startswith('measured_') for k in parity) or 'Measured' in parity.get('recorded_source_comments','') or row['parity'].startswith('reported gate:') else 'Declared gate' if parity else 'Not compared' if 'not compared' in row['parity'] else 'Not documented'
        if (metric=='dmm' and parity.get('regression_floor_policy')) or (metric=='dmm-pyiqa' and parity.get('blocker')):
            status='Partial canonical parity'
        inputs=model.get('input',{}) or ({'convention':plain(entry[5])} if len(entry)>5 else {'cli_contract':contracts.get(metric,MISSING)})
        explicit=explicit_aic_inputs(metric)
        if explicit:inputs=explicit
        variant_contract=fixed_factor_contract(metric)
        if variant_contract:inputs=dict(inputs,fixed_factor_contract=variant_contract)
        output=plain(entry[6]) if len(entry)>6 else MISSING
        code=model.get('code',{});weights=model.get('weights',{})
        reference=code.get('upstream',parity.get('oracle',row['reference']))
        record_path=path.relative_to(ROOT).as_posix() if path else 'docs/METRIC_PROVENANCE.md'
        if explicit:record_path=explicit['source']
        implementation_license=row['license']
        manifest=ROOT/'crates'/('zenmetrics-cli' if explicit else path.parts[-4] if path else 'zenmetrics-nn' if row['nn']=='true' or model else metric)/'Cargo.toml'
        if manifest.is_file():
            license_value=tomllib.loads(manifest.read_text()).get('package',{}).get('license')
            if isinstance(license_value,str):implementation_license=license_value
            elif isinstance(license_value,dict):implementation_license=tomllib.loads((ROOT/'Cargo.toml').read_text())['workspace']['package'].get('license',MISSING)
        related=[]
        name_pattern=r'(?<![\w-])'+re.escape(model.get('name',metric))+r'(?![\w-])'
        for section in sections:
            heading=section.splitlines()[0] if section else ''
            if re.search(name_pattern,heading,re.I):
                anchor=re.sub(r'[^\w\s-]','',heading.lstrip('# ').lower()).replace(' ','-')
                related.append({'heading':plain(heading.lstrip('# ')), 'url':REPO+'DIVERGENCES.md#'+anchor,
                                'excerpt':plain(section.split('\n',1)[-1].strip()[:1800])})
        evidence=[record_path]
        for value in parity.values():
            if isinstance(value,str):
                for rel in re.findall(r'(?:\.\./)+[^\s,;]+',value):
                    if path:
                        candidate=(path.parent/rel.rstrip(').')).resolve()
                        if candidate.is_file() and candidate.is_relative_to(ROOT):evidence.append(candidate.relative_to(ROOT).as_posix())
        readme_path=ROOT/'crates'/metric/'README.md'
        readme_evidence={'source':readme_path.relative_to(ROOT).as_posix(),'text':plain(readme_path.read_text())} if readme_path.is_file() else None
        if readme_evidence:evidence.append(readme_evidence['source'])
        if metric in ['brisque','niqe']:
            weights={'notice':'Static trained/statistical parameters; consult the linked crate README and licence notices. Exact checkpoint terms are not normalized in this card.'}
        licenses=sorted(p.relative_to(ROOT).as_posix() for directory in ([path.parent,ROOT/'crates/zenmetrics-nn'] if path else [ROOT/'crates'/metric]) for p in directory.glob('LICENSE*') if not model or p.name.lower()=='license-'+model.get('name','').lower())
        yield clean(dict(metric=metric,name=cite.get('full_name',metric) if cite else metric,
             family='Classical' if metric=='cw-ssim-pyiqa' else 'Learned' if row['nn']=='true' or model or metric=='brisque' else 'Classical',
             kind='NR' if ('no-reference' in model.get('kind','') and 'full-reference' not in model.get('kind','')) or 'wadiqam-nr-' in metric or metric in ['niqe','piqe','brisque','cambi','cambi-aic2026'] else 'FR',
             backend=row['backend'],backends=['CPU','GPU'] if model else [row['backend']],engines=model.get('execution',{}).get('engines',['rten','ort-cuda'] if model else [row['backend']]),
             citation=cite,citation_status='Verified primary metadata' if cite else 'citation needed',
             recorded_paper=model.get('paper',row['paper']),
             original_authors=cite['authors'] if cite else MISSING,
             reference_implementation=reference,reference_identity=code.get('upstream_package',code.get('reference_sha256',parity.get('oracle',row['reference']))),
             reference_code_license=code.get('licence',MISSING),rust_license=implementation_license,
             weights_license=weights.get('terms',weights.get('licence_class',MISSING if model or metric in ['brisque','niqe'] else 'not applicable')),
             weights=weights,license_evidence=licenses,
             inputs=inputs,display='Required: consult explicit viewing contract' if row['needs_display']=='true' else 'No display requirement in CLI mapping; see input convention',
             output_range=output,direction=model.get('direction',row['direction'] or MISSING),cli_contract=variant_description(metric) or contracts.get(metric,MISSING),
             parity_status=status,parity=parity or {'reported':row['parity'],'oracle':row['reference']},
             evidence=list(dict.fromkeys(evidence+(['crates/zenmetrics-cli/src/metrics/mod.rs'] if not model else []))),
             divergences=related,conventions={k:model[k] for k in ['aic4','oracle_convention','convention'] if k in model},
             source=record_path,readme_evidence=readme_evidence,columns=row['columns'],
             repro=repro.commands(metric,row,{'_card_dir':path.parent.name,'_card_source':record_path,
                 '_consent':next((value for value in re.findall(r'ZENMETRICS_NN_ACCEPT_WEIGHTS=([a-z0-9_]+)',contracts.get(metric,''))),None)} if path else {}),
             links={'speed':SITE+'#speed?metric='+metric,'quality':SITE+'#quality?metric='+metric,'agreement':SITE+'#agreement?metric='+metric} ))


def exports(items):
    all_bib={c['citation']['url']:c['citation']['bibtex'] for c in items if c['citation']}
    output=io.StringIO();fields=['metric','name','family','kind','backend','backends','citation_status','paper','authors','venue','year','paper_url','reference_implementation','reference_identity','reference_code_license','rust_license','weights_license','parity_status','source','inputs','output_range','direction','parity','evidence','divergences','conventions','weights','license_evidence','original_authors','engines','cli_contract','readme_evidence','speed_observations','agreement_observations','quality_panels']
    writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for card in items:
        row={k:card.get(k,'') for k in fields};c=card['citation'] or {}
        row.update(paper=c.get('title','citation needed'),authors='; '.join(c.get('authors',[])),venue=c.get('venue',''),year=c.get('year',''),paper_url=c.get('url',''))
        row={key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value for key,value in row.items()}
        writer.writerow(row)
    return {'model-cards.json':json.dumps(items,ensure_ascii=False,allow_nan=False,indent=2)+'\n',
            'model-cards.csv':output.getvalue(),'model-cards.bib':'\n'.join(all_bib.values())}


def comparison_results(path):
    if not path.exists():
        return {}
    value=json.loads(path.read_text())
    if value.get('schema')!=2:
        return value
    import hashlib
    combined={}
    for shard in value['shards']:
        source=path.parent/shard['file']
        if source.resolve().parent!=path.parent.resolve():
            raise ValueError('comparison shard must be adjacent to its index')
        content=source.read_bytes()
        if hashlib.sha256(content).hexdigest()!=shard['sha256']:
            raise ValueError('comparison shard hash mismatch')
        rows=json.loads(content)
        if len(rows)!=shard['metrics'] or combined.keys() & rows.keys():
            raise ValueError('comparison shard count or duplicate metric')
        combined.update(rows)
    if len(combined)!=value['metrics']:
        raise ValueError('comparison metric count mismatch')
    return combined


def decimal_delta(value):
    """Print retained precision without exponent notation or invented zeroes."""
    from decimal import Decimal
    if value is None:
        return 'not recorded'
    return format(Decimal(str(value)), 'f')


def variant_family(metric):
    for prefix in ('ms-ssim', 'ssimc', 'ssim', 'msssim', 'psnrhvs', 'psnr',
                   'butteraugli', 'vmaf', 'fsim', 'vsi', 'haarpsi', 'hdrvdp',
                   'iwssim', 'cw-ssim', 'lpips', 'stlpips', 'wadiqam', 'topiq'):
        if metric == prefix or metric.startswith(prefix+'-'):
            return prefix
    return metric


def readme_variant_tables(items, comparisons):
    lines = ['', '### Executable, versions, output differences and terms', '',
        'Use each exact metric name below with `target/release/zenmetrics score --metric NAME '
        '--reference ref.png --distorted dist.png`. The linked card supplies its build features, '
        'model export, consent and display arguments; the command is not a substitute for those requirements.', '',
        '**Licensing is not uniformly cleared.** Our Rust license, upstream reference-code terms '
        'and weight terms are separate. See the [pinned upstream audit](docs/PYIQA_UPSTREAM_LICENSES_2026-10-15.md) '
        'for research-only clauses and open questions affecting FSIM, VSI, PSNR-HVS, HDR-VDP, '
        'unlicensed reference sources and pretrained backbones. A permissive reference-code license '
        'does not grant permission for separately licensed weights. “Not recorded” means unresolved, not unrestricted.', '',
        '**Deltas are absolute raw-score differences**, written as decimals (for example `0.00003`). '
        'Maxima below come only from the named [local canonical comparison cohorts]'
        '(benchmarks/canonical_recheck_2026-10-19/README.md), with their exact Rust revision, '
        'input dimensions and thread configurations. They are not global error bounds or AIC '
        'published-column agreement. A median is shown only when retained as a median of paired '
        'score differences; it is never inferred from a maximum or a pass threshold. '
        'Missing local deltas do not erase older parity evidence in the cards.', '']
    groups = {}
    for card in items:
        groups.setdefault(variant_family(card['metric']), []).append(card)
    for family, variants in groups.items():
        lines += ['<details>', '<summary>'+family+' — variants and configurations</summary>', '',
                  '| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |',
                  '| --- | --- | ---: | ---: | --- |']
        for card in variants:
            obs = comparisons.get(card['metric'], {})
            terms = str(card['reference_code_license'])+' / '+str(card['weights_license'])
            if variant_family(card['metric']) in ('fsim','vsi','psnrhvs','hdrvdp'):
                terms += '; open terms: see pinned audit'
            values = ['[`'+card['metric']+'`](docs/MODEL_CARDS.md#'+card['metric']+')',
                      obs.get('canonical', card['reference_identity']),
                      decimal_delta(obs.get('max_abs_delta')),
                      decimal_delta(obs.get('median_abs_delta')), terms]
            lines.append('| '+' | '.join(str(v).replace('|',r'\|').replace('\n',' ') for v in values)+' |')
        lines += ['', '</details>', '']
    lines += ['DVIFMISH is a separate, unpublished benchmark dependency: `publish = false`; '
              'publication requires the original author’s consent. Its six benchmark configurations '
              'are `dvifmish-port-min-v0.2`, `dvifmish-port-2d-v0.2`, '
              '`dvifmish-port-lap-v0.2`, `dvifmish-port-lap-v0.3`, '
              '`dvifmish-talk-faithful-luma-float` and `dvifmish-talk-faithful-luma-int`. '
              'These are not public `zenmetrics score` names, and benchmark repeatability '
              'does not establish native-author oracle parity. Private adapters, models and '
              'weights are not distributed by these instructions.', '']
    return '\n'.join(lines)


def cold_numerical_misses(path, seen=None):
    """Count source-bound cold cases that exceed an assigned numerical gate."""
    import hashlib
    import math
    index=json.loads(path.read_text())
    misses={}
    if seen is None:seen=set()
    for shard in index['shards']:
        source=path.parent/shard['file']
        if source.resolve().parent!=path.parent.resolve():
            raise ValueError('cold shard must be adjacent to its index')
        content=source.read_bytes()
        if hashlib.sha256(content).hexdigest()!=shard['sha256']:
            raise ValueError('cold shard hash mismatch')
        rows=json.loads(content)['cases']
        if len(rows)!=shard['cases']:
            raise ValueError('cold shard count mismatch')
        for row in rows:
            key=(row['case'],row['native_source'])
            if key in seen:raise ValueError('duplicate cold case')
            seen.add(key)
            bound=row['absolute_bound']
            if bound is None:continue
            for block in row['blocks']:
                delta=block['delta']
                canonical=block.get('arms',{}).get('canonical',{}).get('score')
                if delta is None or canonical is None:continue
                if (math.isfinite(delta) and math.isfinite(canonical)
                        and delta>bound+row['relative_bound']*abs(canonical)):
                    misses[row['metric']]=misses.get(row['metric'],0)+1
                    break
    return misses


def cold_batch_numerical_misses(path):
    """Combine immutable cold batches without counting an observation twice."""
    import hashlib
    batches=json.loads(path.read_text())['batches']
    totals={};seen=set()
    for batch in batches:
        source=path.parent/batch['file']
        if not source.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError('cold batch must remain inside its ledger')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=batch['sha256']:
            raise ValueError('cold batch index hash mismatch')
        for metric,count in cold_numerical_misses(source,seen).items():
            totals[metric]=totals.get(metric,0)+count
    return totals


def readme_section(items):
    comparison_path=ROOT/'benchmarks/canonical_recheck_2026-10-19/results.json'
    comparisons=comparison_results(comparison_path)
    statuses=json.loads(comparison_path.with_name('unmeasured.json').read_text())
    cold_misses=cold_batch_numerical_misses(comparison_path.parent/'cold-panel-publications.json')
    lines=['## Metrics','','[Models & metrics explorer]('+SITE+'#models) · [Full cards, papers and terms](docs/MODEL_CARDS.md)',
           '', 'Every CLI name is listed below. FR/NR indicates reference requirements; NN marks neural models. '
           'Code and model terms, paper citations and full reference identities remain in the linked cards.',
           '', '**Rust/C** is Rust time divided by canonical time (lower is faster), '
           'using a representative warm cohort, preferring one thread. '
           'Approximate local comparisons are descriptive, with their input size, thread budget and source pins '
           'in the [comparison ledger](benchmarks/canonical_recheck_2026-10-19/README.md). '
           'A dash means no comparable measurement. “Prior” means older recorded evidence; '
           '“gate” means a declared bound, not a measured pass. Δ is the measured difference without an assigned bound. '
           'Pass counts include retained follow-up cohorts; they are not counts of distinct inputs. '
           '! flags any observed case above 1.08; '
           'R/O identifies RTEN/ORT CPU for neural ratios. '
           'Cold numerical misses are shown separately from the retained warm evidence; '
           'See the ledger for confirmation status.',
           '', '| Metric | Type | Canonical | Check | Rust/C |', '| --- | --- | --- | --- | ---: |']
    for c in items:
        observation=comparisons.get(c['metric'],statuses.get(c['metric'],{}))
        ref=observation.get('canonical',c['reference_implementation'])
        if ref.startswith('https://github.com/'):
            ref='/'.join(ref.split('/')[3:5])
        elif ref in ('no evidence','—'):
            ref='unrecorded'
        ref=ref.split(';')[0]
        if observation.get('ratio_canonical_device')=='cuda':ref='CUDA '+ref
        if len(ref)>24:ref=ref[:21].rstrip()+'…'
        check=observation.get('check',{'Reported results':'prior','Declared gate':'gate','Partial canonical parity':'partial'}.get(c['parity_status'],'unchecked'))
        if observation and c['parity_status']=='Partial canonical parity':check='partial; '+check
        if c['metric'] in cold_misses:
            check=str(cold_misses[c['metric']])+' cold misses'
        ratio=observation.get('ratio')
        # Keep values near the 1.08 flag distinguishable in the compact table.
        digits=3 if ratio is not None and .8<=ratio<=1.3 else 2
        speed=f'≈{ratio:.{digits}g}×' if ratio is not None else '—'
        if ratio is not None:speed+={'rten':' R','ort-cpu':' O'}.get(observation.get('ratio_engine'),'')
        worst=observation.get('worst_ratio',ratio)
        if worst is not None and worst>1.08:speed+=' !'
        metric='[`'+c['metric']+'`](docs/MODEL_CARDS.md#'+c['metric']+')'
        values=[metric,c['kind']+('/NN' if c['family']=='Learned' and c['metric']!='brisque' else ''),ref,check,speed]
        lines.append('| '+' | '.join(str(v).replace('|','\\|').replace('\n',' ') for v in values)+' |')
    return '\n'.join(lines)+'\n'+readme_variant_tables(items, comparisons)+'\n\n'


def documentation(items):
    out=['# Models and metrics','','Generated from model cards, the CLI catalog and provenance ledger. Bibliographic metadata comes from the linked original publisher/arXiv record. A missing field remains missing. Reference parity, published-dataset reproduction and performance are separate evidence.','',
         'Regenerate with `python3 scripts/dashboard/model_cards.py`; verify currency with `python3 scripts/dashboard/model_cards.py --check`. The dashboard workflow regenerates before checking.','',
         'The Learned family includes neural metrics and BRISQUE’s trained regressor. Classical includes analytical and natural-scene-statistics methods. Backend is the CLI route; optional learned engines are listed separately.','']
    for c in items:
        out+=['## '+c['metric'],'',c['kind']+' · '+c['family']+' · '+c['backend']+' · '+c['parity_status'],'']
        cite=c['citation']
        if cite:out += ['**Paper:** ['+cite['title']+']('+cite['url']+'). '+ '; '.join(cite['authors'])+'. '+cite['venue']+'. '+cite['year']+(' ('+cite['year_note']+')' if cite.get('year_note') else '')+'.','', '```bibtex',cite['bibtex'].strip(),'```','']
        else:out+=['**Paper: citation needed.**','']
        for key in ['recorded_paper','reference_implementation','reference_identity','reference_code_license','rust_license','weights_license','display','output_range','direction','cli_contract']:
            out+=['**'+key.replace('_',' ').capitalize()+':** '+str(c[key]),'']
        out+=['**Reproduce from the CLI:**','','```sh',*c['repro']['lines'],'```','',*[n+'' for n in c['repro']['notes']],*([''] if c['repro']['notes'] else [])]
        for key in ['weights','inputs','parity','conventions']:
            out+=['**'+key.capitalize()+':**','', '```json',json.dumps(c[key],ensure_ascii=False,indent=2),'```','']
        if c['readme_evidence']:out+=['**Crate documentation:** ['+c['readme_evidence']['source']+']('+REPO+c['readme_evidence']['source']+')', '', c['readme_evidence']['text'],'']
        out+=['**Sources:** '+ ' · '.join('['+p+']('+REPO+p+')' for p in c['evidence']+c['license_evidence']),'',
              '**Known divergences:** '+(' · '.join('['+d['heading']+']('+d['url']+')' for d in c['divergences']) or '[Consult the divergence ledger]('+REPO+'DIVERGENCES.md); no exact metric heading joined.'),'',
              '[Speed observations]('+c['links']['speed']+') · [AIC/CID22 and other correlation panels]('+c['links']['quality']+') · [Agreement evidence]('+c['links']['agreement']+')','']
    return '\n'.join(out)


def generate(check=False):
    items=list(cards());readme=ROOT/'README.md';text=readme.read_text()
    start=text.index('## Metrics\n');end=text.index('\n## ',start+1)
    expected={readme:text[:start]+readme_section(items)+text[end+1:], ROOT/'docs/MODEL_CARDS.md':documentation(items)}
    for path,content in expected.items():
        if check:
            if not path.exists() or path.read_text()!=content:raise SystemExit(path.name+' model cards stale: run scripts/dashboard/model_cards.py')
        else:path.write_text(content)
    print(f'Model cards: {len(items)} metrics; {sum(c["citation"] is None for c in items)} citation needed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();generate(args.check)
