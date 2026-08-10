#!/usr/bin/env bash
# gpu_util_harness.sh — the REPEATABLE GPU-utilization before/after harness (zenmetrics#48).
#
# Measures the executor PROCESS-TOPOLOGY lever on a representative q-ladder: the same ScoreFile
# jobs run (A) one process per job — the pre-#45 chunked-fleet topology — versus (B) one warm
# `jobexec --serve` process fed every job, each arm sampled by `nvidia-smi dmon`. Emits a TSV row
# per arm plus a .meta provenance header (commit, host, GPU, driver, grid) under benchmarks/.
#
# The ladder: N_Q q-points (default 30, evenly spaced 3..90 — low-q density equal to high-q per
# the workspace sweep discipline) × each supplied reference, encoded with zenjpeg via
# `zenmetrics sweep --encoded-out-dir`, staged to the store s3env.sh selects, scored as one
# fleet-shaped ScoreFile jobs (VARIANTS_PER_JOB each; metrics default ssim2 + butteraugli-gpu — the sf-gpu wave shape).
#
# Usage:
#   REF_LARGE=/path/big.png [REF_MED=/path/med.png] [N_Q=30] [METRICS=ssim2,butteraugli-gpu] [VARIANTS_PER_JOB=5] \
#   [ZM_BIN=target/release/zenmetrics] [NSYS=1] [KEEP_STAGE=1] \
#     bash scripts/jobsys/gpu_util_harness.sh <out_prefix>
#
#   REF_MED unset -> derived from REF_LARGE by a 1400px Mitchell downscale (real content,
#   self-contained). NSYS=1 additionally profiles arm B under `nsys --stats=true` (api_sum vs
#   kern_sum — read api_sum FIRST per the workspace GPU discipline).
#
# Store selection + creds: scripts/lib/s3env.sh (ZEN_S3_ENDPOINT set = LAN store; unset = R2).
# Scratch objects land under s3://zentrain/scratch/gpu-util-harness/<run-id>/ and are deleted at
# exit unless KEEP_STAGE=1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$ROOT/scripts/lib/s3env.sh"

OUT_PREFIX="${1:?usage: gpu_util_harness.sh <out_prefix> (e.g. benchmarks/gpu_util_$(hostname)_$(date +%F))}"
REF_LARGE="${REF_LARGE:?set REF_LARGE=/path/to/large.png (canonical pick: an imazen-26 4000x3000 SDR png)}"
N_Q="${N_Q:-30}"
# Default = the sf-gpu wave shape (GPU ssim2 + GPU butteraugli max+pnorm3). Plain "butteraugli"
# is the CPU port — at 12 MP x 30 variants it dominates wall by minutes and measures the wrong
# thing for a GPU-utilization harness.
METRICS="${METRICS:-ssim2,butteraugli-gpu}"
# Variants per ScoreFile job. The fleet pays cold-start once per JOB (a job = one source's
# variant chunk; the avifgen rescue ran CHUNK=9-class jobs), so arm A's process count — and the
# cold-start tax it measures — is representative only if the ladder is split into fleet-shaped
# jobs rather than one giant job per ref.
VARIANTS_PER_JOB="${VARIANTS_PER_JOB:-5}"
ZM_BIN="${ZM_BIN:-$ROOT/target/release/zenmetrics}"
[ -x "$ZM_BIN" ] || { echo "zenmetrics binary not found at $ZM_BIN (build: cargo build --release -p zenmetrics-cli --no-default-features --features sweep,png,gpu,gpu-cuda,orchestrator,orchestrator-cuda)"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required"; exit 1; }

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
S3_PREFIX="s3://zentrain/scratch/gpu-util-harness/$RUN_ID"
WORK="$(mktemp -d "${TMPDIR:-$HOME/tmp}/gpu_util_harness.XXXXXX")"
cleanup() {
  if [ "${KEEP_STAGE:-0}" != "1" ]; then
    aws s3 rm "$S3_PREFIX/" --recursive --endpoint-url "$EP" >/dev/null 2>&1 || true
    rm -rf "$WORK"
  else
    echo "KEEP_STAGE=1: staged objects at $S3_PREFIX, workdir $WORK"
  fi
}
trap cleanup EXIT

# ---- refs -------------------------------------------------------------------------------------
mkdir -p "$WORK/src"
cp "$REF_LARGE" "$WORK/src/large.png"
if [ -n "${REF_MED:-}" ]; then
  cp "$REF_MED" "$WORK/src/med.png"
else
  command -v convert >/dev/null || { echo "ImageMagick 'convert' needed to derive REF_MED (or set REF_MED)"; exit 1; }
  convert "$REF_LARGE" -filter Mitchell -resize 1400x1400 "$WORK/src/med.png"
fi

# ---- ladder encode (zenjpeg, N_Q points evenly spaced 3..90) ----------------------------------
QGRID="$(python3 - "$N_Q" <<'PY'
import sys
n = int(sys.argv[1])
qs = [round(3 + i * (90 - 3) / (n - 1)) for i in range(n)]
print(",".join(str(q) for q in sorted(set(qs))))
PY
)"
echo "[harness] encoding ladder: q={$QGRID} x {large,med}"
"$ZM_BIN" sweep --codec zenjpeg --sources "$WORK/src" --q-grid "$QGRID" \
  --metric ssim2 --output "$WORK/ladder.tsv" --encoded-out-dir "$WORK/enc" >/dev/null 2>"$WORK/encode.err" \
  || { echo "ladder encode failed:"; tail -5 "$WORK/encode.err"; exit 1; }

# ---- stage to the store ------------------------------------------------------------------------
echo "[harness] staging refs + variants to $S3_PREFIX (store: $EP)"
aws s3 cp "$WORK/src/large.png" "$S3_PREFIX/refs/large.png" --endpoint-url "$EP" >/dev/null
aws s3 cp "$WORK/src/med.png" "$S3_PREFIX/refs/med.png" --endpoint-url "$EP" >/dev/null
aws s3 cp "$WORK/enc/" "$S3_PREFIX/enc/" --recursive --endpoint-url "$EP" >/dev/null

# ---- ScoreFile jobs (FULL-URI inputs; fleet-shaped: VARIANTS_PER_JOB per job) -----------------
python3 - "$WORK" "$S3_PREFIX" "$METRICS" "$VARIANTS_PER_JOB" <<'PY'
import json, os, sys
work, prefix, metrics, vpj = sys.argv[1], sys.argv[2], sys.argv[3].split(","), int(sys.argv[4])
jobs = []
for ref in ("large", "med"):
    # save_encoded_variant_named: {stem}_{src_hash}_{label}_q{q}_{knob_hash}.{ext}
    variants = sorted(
        f for f in os.listdir(os.path.join(work, "enc")) if f.startswith(ref + "_") and f.endswith(".jpg")
    )
    assert variants, f"no encoded variants matched {ref}_*.jpg in {work}/enc"
    for i in range(0, len(variants), vpj):
        jobs.append({
            "kind": {"kind": "score_file", "metrics": metrics},
            "inputs": [f"{prefix}/enc/{v}" for v in variants[i : i + vpj]],
            "cell": {"image_path": f"{prefix}/refs/{ref}.png", "codec": "zenjpeg", "q": 0,
                     "knob_tuple_json": "{}"},
        })
with open(os.path.join(work, "jobs.jsonl"), "w") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print(f"[harness] {len(jobs)} ScoreFile jobs, variants per job: {[len(j['inputs']) for j in jobs]}")
PY

# ---- measurement arms --------------------------------------------------------------------------
export ZEN_R2_ENDPOINT="$EP"  # jobexec's objstore reads this + AWS_* (s3env exported those)
DMON_PID=""
start_dmon() { nvidia-smi dmon -s um -d 1 -o T > "$1" 2>/dev/null & DMON_PID=$!; }
stop_dmon() { [ -n "$DMON_PID" ] && kill "$DMON_PID" 2>/dev/null && wait "$DMON_PID" 2>/dev/null || true; DMON_PID=""; }

run_oneshot() { # one fresh process per job — the pre-#45 chunked topology
  while IFS= read -r job; do
    printf '%s' "$job" | "$ZM_BIN" jobexec >> "$WORK/rows_oneshot.jsonl" 2>> "$WORK/oneshot.err" || true
    printf '\n' >> "$WORK/rows_oneshot.jsonl"
  done < "$WORK/jobs.jsonl"
}
cat > "$WORK/serve_driver.py" <<'PY'
# Streams every job in jobs.jsonl through ONE warm `jobexec --serve` child (the #45 topology).
import struct, subprocess, sys
work, bin_ = sys.argv[1], sys.argv[2]
p = subprocess.Popen([bin_, "jobexec", "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=open(f"{work}/serve.err", "w"))
out = open(f"{work}/rows_serve.jsonl", "ab")
for line in open(f"{work}/jobs.jsonl"):
    b = line.strip().encode()
    p.stdin.write(struct.pack("<I", len(b))); p.stdin.write(b); p.stdin.flush()
    st = p.stdout.read(1)[0]
    ln = struct.unpack("<I", p.stdout.read(4))[0]
    payload = p.stdout.read(ln)
    if st == 0:
        out.write(payload + b"\n")
    else:
        print(f"[harness] serve job status={st}: {payload[:120]!r}", file=sys.stderr)
p.stdin.close(); p.wait(timeout=120)
PY
run_serve() { python3 "$WORK/serve_driver.py" "$WORK" "$ZM_BIN"; }

measure() { # $1=arm-name $2=fn
  local dmon="$WORK/dmon_$1.txt" t0 t1
  start_dmon "$dmon"
  t0=$(date +%s.%N)
  "$2"
  t1=$(date +%s.%N)
  stop_dmon
  python3 - "$1" "$dmon" "$t0" "$t1" <<'PY'
import sys
arm, dmon, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
sm = [int(p[5]) for p in (l.split() for l in open(dmon) if not l.startswith("#")) if len(p) > 6 and p[5].isdigit()]
mean = sum(sm) / len(sm) if sm else 0.0
busy = 100.0 * sum(1 for v in sm if v > 5) / len(sm) if sm else 0.0
print(f"{arm}\t{t1-t0:.2f}\t{mean:.1f}\t{max(sm) if sm else 0}\t{busy:.1f}\t{len(sm)}")
PY
}

echo "[harness] arm A: one-shot (fresh process per job)"
ROW_A="$(measure oneshot run_oneshot)"
echo "[harness] arm B: warm serve (one process, all jobs)"
ROW_B="$(measure serve run_serve)"
if [ "${NSYS:-0}" = "1" ]; then
  NSYS_BIN="${NSYS_BIN:-/usr/local/cuda/bin/nsys}"
  [ -x "$NSYS_BIN" ] || NSYS_BIN=nsys
  echo "[harness] arm C: warm serve under nsys (api_sum vs kern_sum; wall not comparable)"
  rm -f "$WORK/rows_serve.jsonl"
  "$NSYS_BIN" profile -t cuda --stats=true --force-overwrite=true -o "${OUT_PREFIX}.nsys" \
    python3 "$WORK/serve_driver.py" "$WORK" "$ZM_BIN" > "${OUT_PREFIX}.nsys_stats.txt" 2>&1 || true
  echo "[harness] nsys stats -> ${OUT_PREFIX}.nsys_stats.txt (read cuda_api_sum FIRST)"
fi

# ---- emit --------------------------------------------------------------------------------------
GPU_INFO="$(nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader | head -1)"
N_ROWS_A=$(grep -c metric "$WORK/rows_oneshot.jsonl" 2>/dev/null || echo 0)
N_ROWS_B=$(grep -c metric "$WORK/rows_serve.jsonl" 2>/dev/null || echo 0)
{
  echo -e "arm\twall_sec\tsm_mean_pct\tsm_max_pct\tbusy_gt5_pct\tdmon_samples"
  echo -e "$ROW_A"
  echo -e "$ROW_B"
} | tee "${OUT_PREFIX}.tsv"
{
  echo "# gpu_util_harness run $RUN_ID"
  echo "commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "host: $(hostname)  gpu: $GPU_INFO"
  echo "store: $EP"
  echo "grid: zenjpeg q={$QGRID} x refs{large=$(identify -format %wx%h "$WORK/src/large.png" 2>/dev/null || echo '?'),med=$(identify -format %wx%h "$WORK/src/med.png" 2>/dev/null || echo '?')}"
  echo "metrics: $METRICS   jobs: $(wc -l < "$WORK/jobs.jsonl")   rows: oneshot=$N_ROWS_A serve=$N_ROWS_B"
  echo "cmd: REF_LARGE=$REF_LARGE REF_MED=${REF_MED:-<derived-1400px-mitchell>} N_Q=$N_Q VARIANTS_PER_JOB=$VARIANTS_PER_JOB METRICS=$METRICS bash scripts/jobsys/gpu_util_harness.sh $OUT_PREFIX"
  echo "arms: oneshot = fresh jobexec process per ScoreFile job (pre-#45 chunked topology);"
  echo "      serve   = one warm 'jobexec --serve' process for all jobs (#45 topology)."
  echo "note: ZEN_SCOREFILE_WARMREF default-ON in both arms (orchestrator-cuda build) — the arms"
  echo "      isolate PROCESS topology; sm% from 'nvidia-smi dmon -s um' 1 Hz sampling."
} > "${OUT_PREFIX}.meta"
echo "[harness] wrote ${OUT_PREFIX}.tsv + .meta"

# Row-parity gate: both arms must produce the same number of metric rows.
if [ "$N_ROWS_A" != "$N_ROWS_B" ]; then
  echo "[harness] WARNING: row-count mismatch oneshot=$N_ROWS_A serve=$N_ROWS_B — investigate before citing numbers" >&2
  exit 3
fi
