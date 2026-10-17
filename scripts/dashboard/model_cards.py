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
from catalog import ROOT, rows as catalog_rows, md_tables, plain, function
import repro

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
    for path in sorted((ROOT/'crates/zenmetrics-nn/models').glob('*/card.toml')):
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

        status='Reported results' if parity.get('results') or parity.get('status') or any('reported_worst' in k for k in parity) or 'Measured' in parity.get('recorded_source_comments','') or row['parity'].startswith('reported gate:') else 'Declared gate' if parity else 'Not compared' if 'not compared' in row['parity'] else 'Not documented'
        inputs=model.get('input',{}) or ({'convention':plain(entry[5])} if len(entry)>5 else {'cli_contract':contracts.get(metric,MISSING)})
        output=plain(entry[6]) if len(entry)>6 else MISSING
        code=model.get('code',{});weights=model.get('weights',{})
        reference=code.get('upstream',parity.get('oracle',row['reference']))
        record_path=path.relative_to(ROOT).as_posix() if path else 'docs/METRIC_PROVENANCE.md'
        implementation_license=row['license']
        manifest=ROOT/'crates'/('zenmetrics-nn' if row['nn']=='true' or model else metric)/'Cargo.toml'
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
             family='Learned' if row['nn']=='true' or model or metric=='brisque' else 'Classical',
             kind='NR' if ('no-reference' in model.get('kind','') and 'full-reference' not in model.get('kind','')) or 'wadiqam-nr-' in metric or metric in ['niqe','piqe','brisque','cambi'] else 'FR',
             backend=row['backend'],backends=['CPU','GPU'] if model else [row['backend']],engines=model.get('execution',{}).get('engines',['rten','ort-cuda'] if model else [row['backend']]),
             citation=cite,citation_status='Verified primary metadata' if cite else 'citation needed',
             recorded_paper=model.get('paper',row['paper']),
             original_authors=cite['authors'] if cite else MISSING,
             reference_implementation=reference,reference_identity=code.get('upstream_package',code.get('reference_sha256',parity.get('oracle',row['reference']))),
             reference_code_license=code.get('licence',MISSING),rust_license=implementation_license,
             weights_license=weights.get('terms',weights.get('licence_class',MISSING if model or metric in ['brisque','niqe'] else 'not applicable')),
             weights=weights,license_evidence=licenses,
             inputs=inputs,display='Required: consult explicit viewing contract' if row['needs_display']=='true' else 'No display requirement in CLI mapping; see input convention',
             output_range=output,direction=model.get('direction',row['direction'] or MISSING),cli_contract=contracts.get(metric,MISSING),
             parity_status=status,parity=parity or {'reported':row['parity'],'oracle':row['reference']},
             evidence=list(dict.fromkeys(evidence+(['crates/zenmetrics-cli/src/metrics/mod.rs'] if not model else []))),
             divergences=related,conventions={k:model[k] for k in ['aic4','oracle_convention','convention'] if k in model},
             source=record_path,readme_evidence=readme_evidence,columns=row['columns'],
             repro=repro.commands(metric,row,{'_card_dir':path.parent.name} if path else {}),
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


def readme_section(items):
    lines=['## Metrics','','[Models & metrics explorer]('+SITE+'#models) · [Full cards and BibTeX](docs/MODEL_CARDS.md)',
           '', 'Source-linked records for each CLI metric; rows group variants only when the displayed paper, reference, terms and parity status match. Reported parity is scoped to its oracle and fixtures; a declared gate is not a measured result. Missing bibliographic fields are labelled **citation needed**. Code and checkpoint terms are separate.',
           '', '| Metric | Family | Paper | Reference implementation | Licence | Parity status |', '| --- | --- | --- | --- | --- | --- |']
    grouped={}
    for c in items:
        cite=c['citation'];paper=('['+cite['title']+']('+cite['url']+')') if cite else 'citation needed'
        ref=c['reference_implementation'];lic='Rust: '+c['rust_license']+'; weights: '+c['weights_license']
        key=(c['kind']+' · '+c['family'],paper,ref,lic,c['parity_status'],'' if cite else c['metric'])
        grouped.setdefault(key,[]).append(c)
    for key,group in grouped.items():
        values=[', '.join('[`'+c['metric']+'`](docs/MODEL_CARDS.md#'+c['metric']+')' for c in group),*key[:5]]
        lines.append('| '+' | '.join(str(v).replace('|','\\|').replace('\n',' ') for v in values)+' |')
    return '\n'.join(lines)+'\n\n'


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
