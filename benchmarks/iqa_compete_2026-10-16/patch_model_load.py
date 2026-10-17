#!/usr/bin/env python3
"""Benchmark-only model initialization adapter in an owned pinned snapshot.

Keep mathematical code unchanged. Cache parsed immutable VMAF parameters after
warmup; public owned-model returns clone those parameters. No shipped source is
changed. The receipt binds original/patched source and this adapter's bytes.
"""
import argparse,hashlib,json,pathlib
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('snapshot',type=P);p.add_argument('original',type=P);p.add_argument('receipt',type=P);p.add_argument('--reference',action='store_true');a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
rows=[]
for relative,cls,variant in ([] if a.reference else [('crates/vmaf/src/lib.rs','VmafModel','ModelVariant'),('crates/vmaf/src/v0.rs','VmafV0Model','VmafV0Variant')]):
    path=a.snapshot/relative
    if path.samefile(a.original/relative):raise ValueError('snapshot aliases the source checkout')
    original=path.read_bytes();s=original.decode()
    declaration=f'pub struct {cls} {{';assert s.count(declaration)==1
    s=s.replace(declaration,'#[derive(Clone)]\n'+declaration,1)
    if cls=='VmafModel':
        needle='struct SupportVector<const N: usize> {';assert s.count(needle)==1
        s=s.replace(needle,'#[derive(Clone)]\n'+needle,1)
    signature=f'    pub fn new(variant: {variant}) -> Result<Self, Error> {{';assert s.count(signature)==1
    adapter=f'''    // BENCHMARK SNAPSHOT ONLY: exclude immutable SVM parsing after warmup.
    pub fn new(variant: {variant}) -> Result<Self, Error> {{
        std::thread_local! {{
            static MODELS: std::cell::RefCell<std::collections::HashMap<usize, {cls}>> = std::cell::RefCell::new(std::collections::HashMap::new());
        }}
        MODELS.with(|models| {{
            let mut models = models.borrow_mut();
            if let Some(model) = models.get(&(variant as usize)) {{ return Ok(model.clone()); }}
            let model = Self::new_uncached(variant)?;
            models.insert(variant as usize, model.clone());
            Ok(model)
        }})
    }}
    fn new_uncached(variant: {variant}) -> Result<Self, Error> {{'''
    s=s.replace(signature,adapter,1);path.write_text(s)
    rows.append(dict(path=relative,original_sha256=sha(original),adapted_sha256=sha(path.read_bytes())))
relative='crates/zenmetrics-cli/src/metrics/vmaf.rs';path=a.snapshot/relative;assert not path.samefile(a.original/relative)
original=path.read_bytes();s=original.decode();needle='fn to_yuv420_even(';assert s.count(needle)==1;s=s.replace(needle,'pub(crate) fn to_yuv420_even(',1);path.write_text(s)
rows.append(dict(path=relative,original_sha256=sha(original),adapted_sha256=sha(path.read_bytes()),reason='reuse exact existing ingress in the benchmark source module'))
receipt=json.dumps(dict(adapter_sha256=sha(P(__file__).read_bytes()),scope='benchmark-only owned snapshot; immutable model parsing cached; model cloning stays timed',files=rows,reference=a.reference),indent=2)+'\n'
if a.receipt.exists() and a.receipt.read_text()!=receipt:
    backup=a.receipt.with_name(a.receipt.stem+'-'+sha(a.receipt.read_bytes())[:16]+'.bak.json')
    if not backup.exists():a.receipt.rename(backup)
a.receipt.write_text(receipt)
print(a.receipt)
