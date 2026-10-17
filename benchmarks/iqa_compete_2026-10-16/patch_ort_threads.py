#!/usr/bin/env python3
"""Configure ORT session worker budgets only in the owned benchmark snapshot."""
import argparse,hashlib,json,pathlib
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('snapshot',type=P);p.add_argument('original',type=P);p.add_argument('receipt',type=P);a=p.parse_args()
relative='crates/zenmetrics-nn/src/engine.rs';path=a.snapshot/relative
if path.samefile(a.original/relative):raise ValueError('snapshot aliases the source checkout')
original=path.read_bytes();s=original.decode()
assert s.count('Session::builder(env)') in (1,3),'Every ORT image/pair/trunk builder must be configured'
needle='''            let session = Session::builder(env)
                .and_then(|b| b.with_deterministic_compute(true).map_err(Into::into))'''
replacement='''            // BENCHMARK SNAPSHOT ONLY: explicitly configure the cell budget.
            let budget = super::workers();
            println!("IQA_ORT_THREADS {budget}");
            let session = Session::builder(env)
                .and_then(|b| b.with_intra_threads(budget).map_err(Into::into))
                .and_then(|b| b.with_inter_threads(1).map_err(Into::into))
                .and_then(|b| b.with_deterministic_compute(true).map_err(Into::into))'''
if s.count(needle)==3:
    s=s.replace(needle,replacement);builders=3
else:
    signature='fn cuda_session(env: &Environment, path: &Path) -> Result<Session, Error>'
    call='            let session = cuda_session(environment()?, path)?;'
    shared='''        Session::builder(env)
            .and_then(|b| b.with_deterministic_compute(true).map_err(Into::into))'''
    assert s.count(signature)==1 and s.count(call)==3 and s.count(shared)==1,'Every ORT image/pair/trunk builder must be configured'
    s=s.replace(signature,signature.replace('path: &Path','path: &Path, budget: usize'))
    prefix=replacement.split('            let session = Session::builder(env)')[0]
    s=s.replace(call,prefix+call.replace('path)?','path, budget)?'))
    s=s.replace(shared,'''        Session::builder(env)
            .and_then(|b| b.with_intra_threads(budget).map_err(Into::into))
            .and_then(|b| b.with_inter_threads(1).map_err(Into::into))
            .and_then(|b| b.with_deterministic_compute(true).map_err(Into::into))''')
    builders=1
path.write_text(s)
sha=lambda data:hashlib.sha256(data).hexdigest()
receipt=json.dumps(dict(adapter_sha256=sha(P(__file__).read_bytes()),path=relative,
    original_sha256=sha(original),adapted_sha256=sha(path.read_bytes()),builders=builders,session_call_sites=3,
    scope='Benchmark snapshot only; explicit intra-op cell budget, inter-op one; graph operations, providers and deterministic-compute options unchanged'),indent=2)+'\n'
if a.receipt.exists() and a.receipt.read_text()!=receipt:
    backup=a.receipt.with_name(a.receipt.stem+'-'+sha(a.receipt.read_bytes())[:16]+'.bak.json')
    if not backup.exists():a.receipt.rename(backup)
a.receipt.write_text(receipt)
