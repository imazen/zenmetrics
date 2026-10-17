#!/usr/bin/env python3
"""Deterministic machine-readable extracts, with source links and currentness gate."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from catalog import ROOT,md_tables,plain
SOURCES=['docs/METRIC_PROVENANCE.md','DIVERGENCES.md','docs/AIC2026_METRICS_AND_FITTING.md','docs/AIC2026_COVERAGE_AUDIT_2026-10-15.md','benchmarks/nn_aic2026_crosscheck_2026-10-15.md']


def render():
    rows=[];identities={}
    for source in SOURCES:
        path=ROOT/source;identities[source]=hashlib.sha256(path.read_bytes()).hexdigest()
        tables=md_tables(path)
        if not tables:raise ValueError('expected markdown tables: '+source)
        for t,table in enumerate(tables):
            for index,cells in enumerate(table[1:]):
                if len(cells)!=len(table[0]):raise ValueError(f'malformed markdown table in {source}, table {t}, row {index}')
                for column,value in zip(table[0],cells):
                    rows.append(dict(document=source,table=t,row=index,column=plain(column),value=plain(value)))
    def encode(rs):
        f=io.StringIO();w=csv.DictWriter(f,fieldnames=['document','table','row','column','value'],delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rs);return f.getvalue()
    outputs={};chunk=[];index=0
    for row in rows:
        if chunk and len(encode(chunk+[row]).encode())>29000:
            outputs[f'tables-{index:02}.tsv']=encode(chunk);index+=1;chunk=[]
        chunk.append(row)
    if chunk:outputs[f'tables-{index:02}.tsv']=encode(chunk)
    outputs['sources.json']=json.dumps(identities,indent=2)+'\n'
    return outputs


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();dest=ROOT/'benchmarks/dashboard/extracts';expected=render()
    if a.check:
        actual={p.name:p.read_text() for p in dest.iterdir() if p.is_file()}
        if actual!=expected:raise SystemExit('markdown extracts stale: run scripts/dashboard/extract_markdown.py')
    else:
        dest.mkdir(exist_ok=True)
        for name,text in expected.items():(dest/name).write_text(text)
