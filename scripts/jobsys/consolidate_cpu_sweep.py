#!/usr/bin/env python3
# Consolidate a Hetzner CPU sweep run (hetzner_cpu_sweep.sh) into the unified CPU training store: merge the
# per-box omni TSVs (ssim2 + zensim) + per-box 372-feature parquets into ONE omni.tsv + features.parquet
# per codec. Encoded variants stay on R2 as the master record (re-derivable metrics) — not re-downloaded.
#   usage: consolidate_cpu_sweep.py <codec_dir> <run_id> [<run_id2> ...]   (multiple runs = bulk + big-tier)
import sys, os, subprocess, glob
import pyarrow.parquet as pq, pyarrow as pa
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from zen_s3env import resolve_full  # noqa: E402
codec, RUNS = sys.argv[1], sys.argv[2:]
OUT = "/mnt/v/zen/zensim-training/2026-06-24-cpu/unified/%s" % codec
# Store resolution via scripts/lib/zen_s3env.py -- defaults to the LAN store,
# ZEN_STORE=r2 opts out. Fixed 2026-08-26 (was: unconditional R2_* env vars).
ep, _ak, _sk, _store_kind, _reachable = resolve_full()
print(f"consolidate_cpu_sweep: store={_store_kind} endpoint={ep}", file=sys.stderr)
env = dict(os.environ, AWS_ACCESS_KEY_ID=_ak, AWS_SECRET_ACCESS_KEY=_sk, AWS_REGION="auto")
def s5(*a): subprocess.run(["s5cmd", "--endpoint-url", ep, *a], env=env, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.makedirs(OUT, exist_ok=True)
import shutil
work = "/mnt/v/zen/cpu-consolidate-%s" % codec
if os.path.isdir(work):
    shutil.rmtree(work)  # box-N filenames COLLIDE across runs -> download each run into its own subdir
os.makedirs(work)
omnis = []; feats = []
for RUN in RUNS:
    PFX = "picker-sweep-2026-06-22/runs/%s" % RUN
    od, fd = work + "/" + RUN + "/omni", work + "/" + RUN + "/feat"
    os.makedirs(od); os.makedirs(fd)
    s5("cp", "s3://codec-corpus/%s/omni/*" % PFX, od + "/")
    s5("cp", "s3://codec-corpus/%s/features/*" % PFX, fd + "/")
    omnis += sorted(glob.glob(od + "/*.tsv"))
    feats += sorted(glob.glob(fd + "/*.parquet"))
# concat omni (keep one header)
omnis = sorted(omnis)
with open(OUT + "/omni.tsv", "w") as out:
    for i, f in enumerate(omnis):
        with open(f) as fh:
            for j, line in enumerate(fh):
                if j == 0 and i > 0:
                    continue
                out.write(line)
rows = max(0, sum(1 for _ in open(OUT + "/omni.tsv")) - 1)
# concat feature parquets (feats accumulated across all runs above)
feats = sorted(feats)
ft = pa.concat_tables([pq.read_table(f) for f in feats]) if feats else None
if ft is not None:
    pq.write_table(ft, OUT + "/features.parquet", compression="zstd")
print("WROTE %s : omni %d rows (%d box files), features %s rows x %s cols [runs: %s]"
      % (OUT, rows, len(omnis), ft.num_rows if ft else 0, ft.num_columns if ft else 0, ",".join(RUNS)), flush=True)
