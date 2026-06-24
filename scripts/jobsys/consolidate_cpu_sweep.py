#!/usr/bin/env python3
# Consolidate a Hetzner CPU sweep run (hetzner_cpu_sweep.sh) into the unified CPU training store: merge the
# per-box omni TSVs (ssim2 + zensim) + per-box 372-feature parquets into ONE omni.tsv + features.parquet
# per codec. Encoded variants stay on R2 as the master record (re-derivable metrics) — not re-downloaded.
#   usage: consolidate_cpu_sweep.py <codec_dir> <run_id> [<run_id2> ...]   (multiple runs = bulk + big-tier)
import sys, os, subprocess, glob
import pyarrow.parquet as pq, pyarrow as pa
codec, RUNS = sys.argv[1], sys.argv[2:]
OUT = "/mnt/v/zen/zensim-training/2026-06-24-cpu/unified/%s" % codec
ep = "https://%s.r2.cloudflarestorage.com" % os.environ["R2_ACCOUNT_ID"]
env = dict(os.environ, AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
           AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"], AWS_REGION="auto")
def s5(*a): subprocess.run(["s5cmd", "--endpoint-url", ep, *a], env=env, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.makedirs(OUT, exist_ok=True)
work = "/mnt/v/zen/cpu-consolidate-%s" % codec
od, fd = work + "/omni", work + "/feat"
os.makedirs(od, exist_ok=True); os.makedirs(fd, exist_ok=True)
for RUN in RUNS:
    PFX = "picker-sweep-2026-06-22/runs/%s" % RUN
    s5("cp", "s3://codec-corpus/%s/omni/*" % PFX, od + "/")
    s5("cp", "s3://codec-corpus/%s/features/*" % PFX, fd + "/")
# concat omni (keep one header)
omnis = sorted(glob.glob(od + "/*.tsv"))
with open(OUT + "/omni.tsv", "w") as out:
    for i, f in enumerate(omnis):
        with open(f) as fh:
            for j, line in enumerate(fh):
                if j == 0 and i > 0:
                    continue
                out.write(line)
rows = max(0, sum(1 for _ in open(OUT + "/omni.tsv")) - 1)
# concat feature parquets
feats = sorted(glob.glob(fd + "/*.parquet"))
ft = pa.concat_tables([pq.read_table(f) for f in feats]) if feats else None
if ft is not None:
    pq.write_table(ft, OUT + "/features.parquet", compression="zstd")
print("WROTE %s : omni %d rows (%d box files), features %s rows x %s cols [runs: %s]"
      % (OUT, rows, len(omnis), ft.num_rows if ft else 0, ft.num_columns if ft else 0, ",".join(RUNS)), flush=True)
