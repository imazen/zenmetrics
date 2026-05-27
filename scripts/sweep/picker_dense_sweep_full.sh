#!/usr/bin/env bash
# Dense q + size sweep for zenjpeg picker training data — FULL-SCALE, LOCAL.
#
# Scales the verified `picker_dense_sweep_dryrun.sh` pipeline to the
# production grid documented in
# zenanalyze/benchmarks/picker_dense_sweep_dryrun_2026-05-27.md
# "FULL-SCALE GRID SPEC":
#
#   sources : K=20 clustered (centroid-nearest, dense feature embedding)
#   sizes   : 16 log-spaced maxdims, DOWNSCALE-ONLY (Lanczos), one dir each
#   q       : 29-level dense grid (step-5 in 5..69, step-2 in 70..100)
#   cells   : 36 (subsampling{444,422,420} × progressive{f,t}
#                  × sharp_yuv{f,t} × effort{0,1,2}) — unified_v13 cell space
#
# Scoring (LOCAL CUDA): ssim2-gpu (the picker reach-ladder TARGET — the
# shipped zensim metric is non-monotone on photo content, the v39 defect)
# + zensim-gpu (carried through for v39 characterization). cvvdp is SKIPPED
# (slow, not needed for the picker, backfillable later from the persisted
# content-addressed encodes).
#
# RESUMABLE: the box reboots unpredictably. Resume granularity is per
# size-dir for the sweep (a completed size's pareto TSV + feature parquet
# are not regenerated) and per-row for the parquet build (sha-dedup +
# already-scored skip). Re-launch the same command to continue.
#
# NO fleet, NO cloud spend. Requires zen-metrics built with
# `--features sweep,gpu,gpu-cuda` + a local CUDA GPU.
#
# Usage:
#   picker_dense_sweep_full.sh [out_root] [zen_metrics_bin] [zenanalyze_dir]
set -euo pipefail

OUT_ROOT="${1:-/mnt/v/zen/picker-dense-full-2026-05-27}"
ZM="${2:-$HOME/work/zen/zenmetrics/target/release/zen-metrics}"
ZA="${3:-$HOME/work/zen/zenanalyze}"
AGENT_ID="${AGENT_ID:-claude-picker-dense-full}"

Q_GRID="5,10,15,20,25,30,35,40,45,50,55,60,65,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100"
KNOB_GRID='{"subsampling":["444","422","420"],"progressive":[false,true],"sharp_yuv":[false,true],"effort":[0,1,2]}'
N_CELLS=36
N_Q=29

SIZES=(32 40 48 64 80 96 128 160 192 256 320 384 512 640 768 1024)

LOGDIR="$OUT_ROOT/logs"
mkdir -p "$LOGDIR" "$OUT_ROOT/encoded" "$OUT_ROOT/features" "$OUT_ROOT/parquet" "$OUT_ROOT/bake"

refresh_marker() {
  local ts
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '%s %s %s\n' "$ts" "$AGENT_ID" "$1" > "$HOME/work/zen/zenmetrics/.workongoing" 2>/dev/null || true
  printf '%s %s %s\n' "$ts" "$AGENT_ID" "$1" > "$ZA/.workongoing" 2>/dev/null || true
}

# ---- step 3: encode + score, one sweep per size-dir (resumable) ----
for sz in "${SIZES[@]}"; do
  SRCDIR="$OUT_ROOT/sources_${sz}"
  PARETO="$OUT_ROOT/pareto_sz${sz}.tsv"
  FEAT="$OUT_ROOT/features/feat_sz${sz}.parquet"
  if [[ ! -d "$SRCDIR" ]]; then
    echo "WARN: missing source dir $SRCDIR — skipping size $sz" | tee -a "$LOGDIR/driver.log" >&2
    continue
  fi
  N_IMG=$(find "$SRCDIR" -maxdepth 1 -name '*.png' | wc -l)
  EXPECT=$(( N_IMG * N_Q * N_CELLS ))
  # Resume: skip a size whose pareto already has the expected row count.
  if [[ -f "$PARETO" && -f "$FEAT" ]]; then
    HAVE=$(( $(wc -l < "$PARETO") - 1 ))
    if [[ "$HAVE" -ge "$EXPECT" ]]; then
      echo "=== size $sz already complete ($HAVE/$EXPECT rows) — skip ===" | tee -a "$LOGDIR/driver.log"
      continue
    fi
    echo "=== size $sz partial ($HAVE/$EXPECT) — re-running ===" | tee -a "$LOGDIR/driver.log"
  fi
  refresh_marker "sweep size=$sz ($N_IMG img × $N_Q q × $N_CELLS cells = $EXPECT cells)"
  echo "=== sweep size=$sz sources=$SRCDIR expect=$EXPECT ===" | tee -a "$LOGDIR/driver.log"
  "$ZM" sweep \
    --codec zenjpeg \
    --sources "$SRCDIR" \
    --q-grid "$Q_GRID" \
    --knob-grid "$KNOB_GRID" \
    --metric zensim-gpu --metric ssim2-gpu \
    --gpu-runtime cuda \
    --zensim-features-regime with-iw \
    --output "$PARETO" \
    --feature-output "$FEAT" \
    --encoded-out-dir "$OUT_ROOT/encoded/sz${sz}" \
    --pairs-tsv "$OUT_ROOT/pairs_sz${sz}.tsv" \
    2>&1 | tee "$LOGDIR/sweep_sz${sz}.log"
done

# ---- step 3b: assemble the reach-ladder ssim2 score-pairs TSV ----
# The sweep already scores ssim2-gpu inline (0-100 mapped) into the pareto
# TSV's score_ssim2_gpu column — VERIFIED byte-identical to the batch path
# (dry-run check). So we derive score_pairs_ssim2.tsv directly from the
# pareto TSVs instead of a redundant second GPU pass.
refresh_marker "assembling ssim2 reach-ladder score-pairs from pareto TSVs"
PAIRS="$OUT_ROOT/score_pairs_ssim2.tsv"
python3 - "$OUT_ROOT" "$PAIRS" <<'PY' 2>&1 | tee "$LOGDIR/build_score_pairs.log"
import csv, os, sys, glob
root, out = sys.argv[1], sys.argv[2]
n = 0
with open(out, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["ref_path","dist_path","image_basename","q","knob_tuple_json","size_class","ssim2_gpu"])
    for pareto in sorted(glob.glob(os.path.join(root, "pareto_sz*.tsv"))):
        sz = os.path.basename(pareto)[len("pareto_"):-len(".tsv")]  # e.g. sz512
        enc_dir = os.path.join(root, "encoded", sz)
        with open(pareto, newline="") as pf:
            for r in csv.DictReader(pf, delimiter="\t"):
                enc_fn = (r.get("encoded_filename") or "").strip()
                s = r.get("score_ssim2_gpu", "")
                if not enc_fn or s in ("", "nan", "NaN"):
                    continue
                dist = os.path.join(enc_dir, enc_fn)
                w.writerow([r["image_path"], os.path.abspath(dist),
                            os.path.basename(r["image_path"]).rsplit(".",1)[0],
                            r["q"], r["knob_tuple_json"], sz, s])
                n += 1
sys.stderr.write(f"wrote {n} score-pairs to {out}\n")
PY

# ---- step 4: build picker parquet (content-address sha256, all metrics) ----
refresh_marker "build_picker_parquet (content-address + join)"
SIZES_CSV=$(IFS=,; echo "${SIZES[*]/#/sz}")
echo "=== build picker parquet: sizes=$SIZES_CSV ===" | tee -a "$LOGDIR/driver.log"
python3 "$ZA/zenpicker-train/scripts/build_picker_parquet.py" \
  --out-root "$OUT_ROOT" \
  --sizes "$SIZES_CSV" \
  --out-parquet "$OUT_ROOT/parquet/picker_dense_full_zenjpeg.parquet" \
  --codec zenjpeg \
  2>&1 | tee "$LOGDIR/build_parquet.log"

# ---- VERIFICATION GATE: artifacts on disk ----
N_ART=$(find "$OUT_ROOT/artifacts" -maxdepth 1 -name '*.jpg' | wc -l)
echo "=== verification gate: $N_ART content-addressed artifacts on disk ===" | tee -a "$LOGDIR/driver.log"
if [[ "$N_ART" -lt 1 ]]; then
  echo "FATAL: no artifacts content-addressed — aborting before retrain" | tee -a "$LOGDIR/driver.log" >&2
  exit 1
fi

# ---- step 6: retrain the picker (no q-leakage, grouped holdout) ----
refresh_marker "zenpicker-train retrain (full dense parquet)"
echo "=== retrain zenpicker-train ===" | tee -a "$LOGDIR/driver.log"
"$ZA/target/release/zenpicker-train" \
  --input "$OUT_ROOT/parquet/picker_dense_full_zenjpeg.parquet" \
  --codec zenjpeg \
  --val-frac 0.25 \
  --out "$OUT_ROOT/bake/zenjpeg_picker_dense_full.bin" \
  2>&1 | tee "$LOGDIR/retrain.log"

echo "=== FULL RUN COMPLETE ===" | tee -a "$LOGDIR/driver.log"
refresh_marker "full run complete"
