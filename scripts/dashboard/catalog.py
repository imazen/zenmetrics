#!/usr/bin/env python3
"""Generate reviewable metric catalog shards from code and model cards."""
import argparse
import csv
import io
from pathlib import Path
import re
import tomllib
ROOT=Path(__file__).resolve().parents[2]


def plain(text):
    text=re.sub(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)',r'\1 (\2)',text)
    text=re.sub(r'[`*]', '', text).replace('\\|','|')
    text=re.sub(r'(?:/(?:home|mnt|Users|tmp)/[^\s)]+|~/[^\s)]+)','[local path]',text)
    return text.strip()


def md_tables(path):
    tables=[];current=[]
    for line in path.read_text().splitlines()+['']:
        if line.startswith('|'):
            for token in ['|Q_JOD|','|Δ|']:line=line.replace(token,token.replace('|',r'\|'))
            cells=[v.strip() for v in re.split(r'(?<!\\)\|',line.strip().strip('|'))]
            if not all(re.fullmatch(r':?-+:?',c.strip()) for c in cells):current.append(cells)
        elif current:
            if len(current)>1:tables.append(current)
            current=[]
    return tables


def function(source,name):
    start=source.index('pub fn '+name+'(');a=source.index('{',start);depth=1;i=a+1
    while depth:
        if source[i]=='{':depth+=1
        elif source[i]=='}':depth-=1
        i+=1
    return source[a+1:i-1]


def rows():
    source=(ROOT/'crates/zenmetrics-cli/src/metrics/mod.rs').read_text()
    names=dict(re.findall(r'MetricKind::(\w+) => "([^"]+)"',function(source,'name')))
    columns={}
    for kind,expr in re.findall(r'MetricKind::(\w+)\s*=>\s*(.*?)(?=,\s*(?:MetricKind|\n\s*}))',function(source,'column_names'),re.S):
        literals=re.findall(r'"([^"]+)"',expr)
        if expr.strip().startswith('&['):
            if literals:columns[kind]=';'.join(literals)
            else:
                constants=re.findall(r'\b[A-Z][A-Z_]+\b',expr);resolved=[]
                for const in constants:
                    declarations=re.findall(r'const '+const+r':[^=]+=[^;]+;',source)
                    literal=next((re.findall(r'"([^"]+)"',d) for d in reversed(declarations) if '"' in d),[])
                    if not literal:raise ValueError('unresolved column constant '+const)
                    resolved.extend(literal)
                columns[kind]=';'.join(resolved)
        elif 'intern_variant_columns' in expr:
            base=expr.split('(')[1].split(',')[0].strip();fallback=re.findall(r'const '+base+r':.*?=\s*&\[(.*?)\];',source,re.S)
            basecols=re.findall(r'"([^"]+)"',fallback[-1]) if fallback else []
            columns[kind]=';'.join(c+'_'+literals[-1] for c in basecols)
        else:
            const=expr.strip();fallback=re.findall(r'const '+re.escape(const)+r':.*?=\s*&?\[?(.*?)\]?;',source,re.S)
            columns[kind]=';'.join(re.findall(r'"([^"]+)"',fallback[-1])) if fallback else ''
    if set(columns)!=set(names):raise ValueError('unparsed column arms: '+str(set(names)-set(columns)))
    gpu=set(re.findall(r'MetricKind::(\w+)',function(source,'requires_gpu')))
    display=set(re.findall(r'MetricKind::(\w+)',function(source,'needs_display')))
    luma=set(re.findall(r'MetricKind::(\w+)',function(source,'is_luma_only')))
    compiled=function(source,'scoring_compiled_in')
    nn=set(re.findall(r'MetricKind::(\w+)',compiled.split('let nr')[0]));nr=set(re.findall(r'MetricKind::(\w+)',compiled.split('let nr')[1]))
    cards=[]
    for p in sorted((ROOT/'crates/zenmetrics-nn/models').glob('*/card.toml')):
        card=tomllib.loads(p.read_text());cards.append((p,card))
    scales={}
    for line in (ROOT/'crates/zenmetrics-nn/models/jnd_scale.tsv').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            f=line.split('\t');
            if f[0] not in ['column','metric']:scales[f[0]]=f[1]
    ledgers=md_tables(ROOT/'docs/METRIC_PROVENANCE.md')
    refs={};last=[]
    for table in ledgers:
        if 'reference impl (validation oracle)' not in ' '.join(table[0]):continue
        for fields in table[1:]:
            if len(fields)<7:continue
            resolved=[]
            for i,f in enumerate(fields):resolved.append(last[i] if f.strip() in ['"','same','“'] and len(last)>i else f)
            last=resolved
            for name in re.findall(r'`([^`]+)`',fields[0]):refs[name]=resolved
    for kind,name in sorted(names.items(),key=lambda x:x[1]):
        cols=columns[kind].split(';');entry=refs.get(name,[])
        card=next(((p,c) for p,c in cards if set(c.get('columns',[]))&set(cols)),None)
        license_name='no evidence';paper='no evidence';oracle=plain(entry[3]) if len(entry)>3 else 'no evidence'
        parity='no evidence';evidence='';card_kind='';direction='';crate=''
        if card:
            p,c=card;weights=c.get('weights',{});code=c.get('code',{});par=c.get('parity',{})
            license_name='code: '+str(code.get('licence','not recorded'))+'; weights: '+str(weights.get('terms',weights.get('licence_class','not recorded')))
            paper=c.get('paper','no evidence');oracle=par.get('oracle',oracle)
            card_kind=c.get('kind','');direction=c.get('direction','')
            if par:parity='declared JND gate '+str(par.get('tolerance_jnd','unspecified'));evidence=p.relative_to(ROOT).as_posix()
        elif entry:
            crate_match=re.search(r'`(?:crates/)?([^`/ ]+)`',entry[1]);crate=crate_match.group(1) if crate_match else ''
            manifest=ROOT/'crates'/crate/'Cargo.toml'
            if manifest.exists():
                package=tomllib.loads(manifest.read_text()).get('package',{});lic=package.get('license')
                if isinstance(lic,dict):lic=tomllib.loads((ROOT/'Cargo.toml').read_text())['workspace']['package'].get('license','no evidence')
                license_name=lic or 'no evidence'
            if 'not yet' in oracle.lower() or 'not compared' in oracle.lower():parity='not compared (ledger)'
            else:parity='reference named; gate evidence in ledger'
            evidence='docs/METRIC_PROVENANCE.md';paper=plain(entry[2])
        yield dict(metric=name,backend='GPU' if kind in gpu else 'CPU',requires_gpu=str(kind in gpu).lower(),needs_display=str(kind in display).lower(),luma_only=str(kind in luma).lower(),columns=columns[kind],nn=str(kind in nn).lower(),nr=str(kind in nr).lower(),kind=card_kind,direction=direction,license=plain(license_name),paper=plain(paper),reference=plain(str(oracle)),parity=parity,evidence=evidence or 'no evidence',jnd_scale=';'.join(c+'='+scales[c] for c in cols if c in scales),column_scope='static feature-off names; enabled implementations may version columns' if not card else 'model output columns')


def render():
    shards={};chunk=[];index=0
    def encode(rs):
        f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(rs[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rs);return f.getvalue()
    for row in rows():
        if chunk and len(encode(chunk+[row]).encode())>29000:
            shards[f'catalog-{index:02}.tsv']=encode(chunk);index+=1;chunk=[]
        chunk.append(row)
    if chunk:shards[f'catalog-{index:02}.tsv']=encode(chunk)
    return shards


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    dest=ROOT/'benchmarks/dashboard/catalog';expected=render()
    if args.check:
        actual={p.name:p.read_text() for p in dest.glob('*.tsv')}
        if actual!=expected:raise SystemExit('catalog stale: run scripts/dashboard/catalog.py')
    else:
        dest.mkdir(exist_ok=True)
        for name,text in expected.items():(dest/name).write_text(text)
