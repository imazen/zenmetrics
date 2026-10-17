#!/usr/bin/env python3
"""Configure ORT session worker budgets only in the owned benchmark snapshot."""
import argparse,hashlib,json,pathlib
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('snapshot',type=P);p.add_argument('original',type=P);p.add_argument('receipt',type=P);a=p.parse_args()
relative='crates/zenmetrics-nn/src/engine.rs';path=a.snapshot/relative
if path.samefile(a.original/relative):raise ValueError('snapshot aliases the source checkout')
original=path.read_bytes();s=original.decode()
needle='''            let session = Session::builder(env)
                .and_then(|b| b.with_deterministic_compute(true).map_err(Into::into))'''
assert s.count(needle)==3,'Every ORT image/pair/trunk builder must be configured'
replacement='''            // BENCHMARK SNAPSHOT ONLY: explicitly configure the cell budget.
            let budget = super::workers();
            println!("IQA_ORT_THREADS {budget}");
            let session = Session::builder(env)
                .and_then(|b| b.with_intra_threads(budget).map_err(Into::into))
                .and_then(|b| b.with_inter_threads(1).map_err(Into::into))
                .and_then(|b| b.with_deterministic_compute(true).map_err(Into::into))'''
s=s.replace(needle,replacement);path.write_text(s)
sha=lambda data:hashlib.sha256(data).hexdigest()
receipt=json.dumps(dict(adapter_sha256=sha(P(__file__).read_bytes()),path=relative,
    original_sha256=sha(original),adapted_sha256=sha(path.read_bytes()),builders=3,
    scope='Benchmark snapshot only; explicit intra-op cell budget, inter-op one; graph operations, providers and deterministic-compute options unchanged'),indent=2)+'\n'
if a.receipt.exists() and a.receipt.read_text()!=receipt:
    backup=a.receipt.with_name(a.receipt.stem+'-'+sha(a.receipt.read_bytes())[:16]+'.bak.json')
    if not backup.exists():a.receipt.rename(backup)
a.receipt.write_text(receipt)
