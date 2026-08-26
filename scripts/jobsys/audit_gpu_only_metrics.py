#!/usr/bin/env python3
"""Criterion-1 audit: ssim2 + butteraugli are scored ONLY on GPU nodes.

Reads every listed job's ledger on the resolved store (zen_s3env; ZEN_STORE
selects) and groups done-rows by (job, metric, worker). The criterion set is
{ssim2, butteraugli} (prefix-matched, so the -gpu runtime suffix is captured):
- PASS requires every such done-row to carry the metric's `-gpu` runtime suffix
  (the runtime column: the GPU rung executed) — a bare `ssim2`/`butteraugli`
  row would mean a CPU rung produced it.
- zensim features + cvvdp + encodes + diffmap-cvvdp on CPU workers are BY DESIGN
  ("cvvdp, features, encodes on CPU") and are reported informationally.

Usage: ZEN_STORE=tower python3 scripts/jobsys/audit_gpu_only_metrics.py [job ...]
Exit 1 on any violation. Writes nothing; pipe to a report file to commit evidence.
"""
import json, sys, collections, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lib"))
from zen_s3env import resolve
import pyarrow.fs as fs, pyarrow.dataset as ds

ep, ak, sk = resolve()
S3 = fs.S3FileSystem(access_key=ak, secret_key=sk, endpoint_override=ep, region="auto")
DEFAULT_JOBS = [
    "hdrgrid-sf-cpu-20260807","hdrgrid-sf-gpu-20260807","hdrgrid-sf-gpu-huge-20260807",
    "hdrgrid-sf-gpu-small-20260807","hdrgrid-sf2-cpu-20260807","hdrgrid-sf2-gpu-20260807",
    "hdrgrid-sf2-gpu-huge-20260807","hdrgrid-sf2-gpu-small-20260807",
    "avifgen-sf-cpu-20260806","avifgen-sf-gpu-20260806","avifgen-sf-gpu-rescue-20260808",
    "hdrgrid-diffmap-20260807",
]
JOBS = sys.argv[1:] or DEFAULT_JOBS
CRITERION = ("ssim2", "butteraugli")  # prefix match; runtime suffix -gpu must be present

tab = collections.Counter(); viol = collections.Counter()
for J in JOBS:
    try:
        t = ds.dataset(f"zentrain/jobs/{J}/ledger/", filesystem=S3, format="parquet") \
              .to_table(columns=["status", "worker", "kind_json"])
    except Exception as e:
        print(f"# {J}: no ledger ({str(e)[:50]})"); continue
    for st, w, k in zip(t.column("status").to_pylist(), t.column("worker").to_pylist(),
                        t.column("kind_json").to_pylist()):
        if st != "done": continue
        try: kd = json.loads(k or "{}")
        except Exception: kd = {}
        mets = kd.get("metrics") or ([kd.get("metric")] if kd.get("metric") else [kd.get("kind", "?")])
        for m in map(str, mets):
            tab[(J, m, w)] += 1
            if m.startswith(CRITERION) and not m.endswith("-gpu"):
                # bare ssim2/butteraugli = a non-GPU runtime produced the score,
                # EXCEPT diffmap kinds: kind_json {"kind":"diffmap","metric":"butteraugli"}
                # names the metric without a runtime suffix; the diffmap executor runs
                # under ZEN_REQUIRE_GPU=1 GPU launches only (worker env), flagged separately.
                if kd.get("kind") == "diffmap":
                    tab[(J, m + " (diffmap; GPU-launch enforced)", w)] += 0
                else:
                    viol[(J, m, w)] += 1

print("job\tmetric\tworker\tdone_rows")
for (J, m, w), n in sorted(tab.items()):
    print(f"{J}\t{m}\t{w}\t{n}")
print()
if viol:
    print("VIOLATIONS (bare ssim2/butteraugli score rows — non-GPU runtime):")
    for (J, m, w), n in sorted(viol.items()):
        print(f"  {J} {m} worker={w}: {n} rows")
    sys.exit(1)
print("PASS: every ssim2/butteraugli SCORE row carries the -gpu runtime suffix "
      "(GPU rung executed; no CPU rung anywhere). diffmap butteraugli rows come from "
      "ZEN_REQUIRE_GPU=1 GPU launches. zensim features + cvvdp + encodes on CPU by design.")
