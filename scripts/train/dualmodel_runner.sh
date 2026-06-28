#!/usr/bin/env bash
# Per-codec dual-model picker runner — the ENTRYPOINT of the zen-train:hybrid-cpu
# image. Runs the per-codec picker pipeline for ONE codec entirely in-container:
#
#   Stage A (GATING, proven): canonical {train,validate,test}.parquet  -> prep_combined
#     -> picker_tree_ab A/B (GBDT/RF/MLP + permutation importance) on the origin split.
#   Stage B (best-effort): omni sweep TSV + features TSV -> omni_to_pareto ->
#     check_mandatory_coverage -> train_hybrid (sklearn HistGBDT teacher + torch
#     LeakyReLU MLP student) -> bake_picker (ZNPR .bin).
#
# Bake-everything: every tool is verified present at boot; if one is MISSING the
# runner FAILS LOUD (it does NOT install anything — the image is broken, rebuild it).
# All results + the full runner log + a STATUS marker are uploaded to R2 even on
# failure (EXIT trap), so the launcher's monitor can tear the box down promptly.
set -u
export DEBIAN_FRONTEND=noninteractive

# ── config from env (set by the launcher's cloud-init container env-file) ──────
CODEC="${CODEC:?CODEC required (e.g. zenwebp_lossy)}"
# train_hybrid operates at codec-FAMILY granularity (zenwebp = vp8+vp8l cells);
# picker_tree_ab at the lossy/lossless granularity. Derive the family if unset.
CODEC_FAMILY="${CODEC_FAMILY:-${CODEC%_lossy}}"; CODEC_FAMILY="${CODEC_FAMILY%_lossless}"
EP="${R2_ENDPOINT:?R2_ENDPOINT required}"
BUCKET="${RUN_BUCKET:-zentrain}"
CANON_PREFIX="${CANON_PREFIX:-canonical/2026-06-27}"
OUT_PREFIX="${OUT_PREFIX:-dualmodel-2026-06-28/$CODEC}"
INPUTS_PREFIX="${INPUTS_PREFIX:-dualmodel-2026-06-28/inputs}"
OMNI_NAME="${OMNI_NAME:-${CODEC_FAMILY}.zensim.combined.tsv}"
FEATURES_NAME="${FEATURES_NAME:-combined_features_vn_tiled.tsv}"
PICKER_TARGET="${PICKER_TARGET:-zensim_a}"
METRIC_COL="${METRIC_COL:-score_zensim}"
RUN_ID="${RUN_ID:-dualmodel-$(date +%s)}"
SKIP_TRAIN_HYBRID="${SKIP_TRAIN_HYBRID:-0}"

WORK=/work; mkdir -p "$WORK"
LOG="$WORK/runner.log"
PYTHONUNBUFFERED=1; export PYTHONUNBUFFERED
# All training code is baked at these paths (see Dockerfile.hybrid-cpu).
export PYTHONPATH=/opt/picker:/opt/picker/configs:/opt/zentrain/tools:/opt/zentrain/examples
PICKERDIR=/opt/picker
ZATOOLS=/opt/za/tools          # bake_picker.py
ZENTRAIN=/opt/zentrain/tools   # train_hybrid.py + _*.py helpers

s5(){ s5cmd --endpoint-url="$EP" "$@"; }
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

STAGE_A="not-run"; STAGE_B="not-run"
on_exit(){
  local rc=$?
  {
    echo "=== STATUS ==="
    echo "run_id=$RUN_ID codec=$CODEC family=$CODEC_FAMILY"
    echo "stage_a_picker_tree_ab=$STAGE_A"
    echo "stage_b_train_hybrid=$STAGE_B"
    echo "runner_rc=$rc finished=$(date -u +%FT%TZ)"
  } > "$WORK/STATUS.txt"
  # best-effort uploads (trap must never abort)
  s5 cp "$LOG" "s3://$BUCKET/$OUT_PREFIX/runner.log" 2>>"$LOG" || true
  s5 cp "$WORK/STATUS.txt" "s3://$BUCKET/$OUT_PREFIX/STATUS.txt" 2>>"$LOG" || true
  # the launcher's monitor polls for this marker key to tear the box down
  if [ "$STAGE_A" = "ok" ]; then
    printf 'stage_a=ok stage_b=%s\n' "$STAGE_B" > "$WORK/_DONE"
    s5 cp "$WORK/_DONE" "s3://$BUCKET/$OUT_PREFIX/_DONE" 2>>"$LOG" || true
  else
    printf 'stage_a=%s rc=%s\n' "$STAGE_A" "$rc" > "$WORK/_FAILED"
    s5 cp "$WORK/_FAILED" "s3://$BUCKET/$OUT_PREFIX/_FAILED" 2>>"$LOG" || true
  fi
}
trap on_exit EXIT

exec > >(tee -a "$LOG") 2>&1
log "dualmodel_runner start: codec=$CODEC family=$CODEC_FAMILY target=$PICKER_TARGET run=$RUN_ID"
log "endpoint=$EP bucket=$BUCKET out=$OUT_PREFIX"

# ── verify baked tools — FAIL LOUD, never install at boot ─────────────────────
fail_missing=0
for t in s5cmd picker_tree_ab zenpredict-bake python3; do
  command -v "$t" >/dev/null 2>&1 || { log "FATAL: baked tool '$t' MISSING — image is broken, rebuild it"; fail_missing=1; }
done
python3 - <<'PY' || fail_missing=1
import sys
try:
    import torch, sklearn, pandas, pyarrow, numpy  # noqa: F401
    print(f"  ML stack OK: torch={torch.__version__} sklearn={sklearn.__version__} "
          f"pandas={pandas.__version__} pyarrow={pyarrow.__version__} numpy={numpy.__version__}")
except Exception as e:
    print(f"FATAL: ML stack import failed: {e}", file=sys.stderr); sys.exit(1)
PY
for f in "$PICKERDIR/prep_combined.py" "$PICKERDIR/omni_to_pareto.py" "$PICKERDIR/origin_split.py" \
         "$PICKERDIR/check_mandatory_coverage.py" "$ZENTRAIN/train_hybrid.py" "$ZATOOLS/bake_picker.py"; do
  [ -s "$f" ] || { log "FATAL: baked code '$f' MISSING — image is broken"; fail_missing=1; }
done
[ "$fail_missing" = 0 ] || { log "aborting: baked tooling incomplete"; exit 3; }
log "baked tooling verified"

# ══════════════════════════════════════════════════════════════════════════════
# Stage A — picker_tree_ab A/B (GATING)
# ══════════════════════════════════════════════════════════════════════════════
log "=== Stage A: picker_tree_ab for $CODEC ==="
CANON_DIR="/data/canon/$CODEC"; mkdir -p "$CANON_DIR"
for f in train validate test; do
  log "fetch canonical $f.parquet"
  s5 cp "s3://$BUCKET/$CANON_PREFIX/$CODEC/$f.parquet" "$CANON_DIR/$f.parquet" \
    || { log "FATAL: canonical $f.parquet fetch failed"; STAGE_A="fetch-failed"; exit 4; }
done
log "prep_combined"
python3 "$PICKERDIR/prep_combined.py" "$CODEC" --src "$CANON_DIR" --out "$WORK" \
  || { log "FATAL: prep_combined failed"; STAGE_A="prep-failed"; exit 4; }

DUMP="$WORK/dump_$CODEC"; mkdir -p "$DUMP"
# val FIRST (the gate) then test (bonus). Upload each split's A/B log + dump
# IMMEDIATELY so a slow `test` split that hits the self-destruct backstop can't
# cost us the already-finished val results. STAGE_A flips to ok the moment val
# uploads, so the box is creditably done even if test is cut short.
for split in val test; do
  log "picker_tree_ab --eval-split $split (this is the dual-model A/B; GBDT/RF/MLP)"
  picker_tree_ab \
      --input "$WORK/combined_$CODEC.parquet" \
      --split-map "$WORK/splitmap_$CODEC.parquet" \
      --eval-split "$split" \
      --dump-dir "$DUMP/$split" \
      --codec-tag "$CODEC" 2>&1 | tee "$WORK/ab_${CODEC}_${split}.log"
  rc=${PIPESTATUS[0]}
  # upload this split's artifacts right away (incremental — robust to a later kill)
  s5 cp "$WORK/ab_${CODEC}_${split}.log" "s3://$BUCKET/$OUT_PREFIX/picker_tree_ab/ab_${CODEC}_${split}.log" || true
  if [ -d "$DUMP/$split" ]; then
    tar -cf "$WORK/dump_${CODEC}_${split}.tar" -C "$DUMP" "$split" 2>/dev/null \
      && s5 cp "$WORK/dump_${CODEC}_${split}.tar" "s3://$BUCKET/$OUT_PREFIX/picker_tree_ab/dump_${CODEC}_${split}.tar" || true
  fi
  if [ "$rc" != 0 ]; then
    log "picker_tree_ab --eval-split $split exited rc=$rc"
    [ "$split" = "val" ] && { STAGE_A="ab-val-failed"; exit 4; }   # val is the gate
  fi
  [ "$split" = "val" ] && { STAGE_A="ok"; log "Stage A gate MET (val A/B uploaded)"; }
done
log "=== Stage A complete (A/B logs + dumps uploaded incrementally) ==="

# ══════════════════════════════════════════════════════════════════════════════
# Stage B — train_hybrid teacher/student + bake (best-effort; does not gate)
# ══════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_TRAIN_HYBRID" = "1" ]; then
  log "Stage B skipped (SKIP_TRAIN_HYBRID=1)"; STAGE_B="skipped"; exit 0
fi
log "=== Stage B: train_hybrid for family=$CODEC_FAMILY target=$PICKER_TARGET ==="
run_stage_b(){
  mkdir -p /data/tin
  log "fetch omni=$OMNI_NAME + features=$FEATURES_NAME"
  s5 cp "s3://$BUCKET/$INPUTS_PREFIX/$OMNI_NAME"     "/data/tin/$OMNI_NAME"     || { log "omni fetch failed"; return 1; }
  s5 cp "s3://$BUCKET/$INPUTS_PREFIX/$FEATURES_NAME" "/data/tin/$FEATURES_NAME" || { log "features fetch failed"; return 1; }
  # picker_config_common hardcodes PP=/home/lilith/picker-pp and resolves PARETO/FEATURES/OUT
  # paths from it; replicate that layout so the config imports cleanly (keep_features opens
  # the FEATURES path at import time) and the outputs land where the config expects.
  local PP=/home/lilith/picker-pp
  mkdir -p "$PP/train" "$PP/models"
  log "omni_to_pareto (metric_col=$METRIC_COL -> $CODEC_FAMILY.$PICKER_TARGET.pareto.parquet)"
  python3 "$PICKERDIR/omni_to_pareto.py" \
    --omni "/data/tin/$OMNI_NAME" --features-tsv "/data/tin/$FEATURES_NAME" \
    --metric-col "$METRIC_COL" \
    --out-pareto "$PP/train/$CODEC_FAMILY.$PICKER_TARGET.pareto.parquet" \
    --out-features "$PP/train/$CODEC_FAMILY.features.tsv" || { log "omni_to_pareto failed"; return 1; }
  log "check_mandatory_coverage (--codec $CODEC_FAMILY)"
  if ! python3 "$PICKERDIR/check_mandatory_coverage.py" \
        --pareto "$PP/train/$CODEC_FAMILY.$PICKER_TARGET.pareto.parquet" --codec "$CODEC_FAMILY"; then
    log "mandatory coverage gate FAILED — a first-class mode is missing from this omni; not training"
    return 2
  fi
  log "train_hybrid --codec-config ${CODEC_FAMILY}_picker"
  CUDA_VISIBLE_DEVICES="" PICKER_TARGET="$PICKER_TARGET" OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}" \
    python3 "$ZENTRAIN/train_hybrid.py" --codec-config "${CODEC_FAMILY}_picker" \
    --activation leakyrelu --hidden 192,192,192 2>&1 | tee "$WORK/train_hybrid.log"
  local trc=${PIPESTATUS[0]}
  [ "$trc" = 0 ] || { log "train_hybrid exited rc=$trc"; return 1; }
  local MJ="$PP/models/${CODEC_FAMILY}_predict_${PICKER_TARGET}_v0.1.json"
  if [ -s "$MJ" ]; then
    log "bake_picker -> .bin"
    python3 "$ZATOOLS/bake_picker.py" --model "$MJ" \
      --out "$PP/models/${CODEC_FAMILY}_predict_${PICKER_TARGET}_v0.1.bin" \
      --dtype f16 --bake-bin /usr/local/bin/zenpredict-bake --allow-unsafe 2>&1 | tee -a "$WORK/train_hybrid.log" || true
  fi
  return 0
}
if run_stage_b; then STAGE_B="ok"; else rcb=$?; STAGE_B="failed(rc=$rcb)"; fi
# upload train_hybrid artifacts regardless of pass/fail
s5 cp "$WORK/train_hybrid.log" "s3://$BUCKET/$OUT_PREFIX/train_hybrid/train_hybrid.log" 2>/dev/null || true
for ext in json bin log manifest.json; do
  for f in /home/lilith/picker-pp/models/${CODEC_FAMILY}_predict_${PICKER_TARGET}_v0.1.${ext}; do
    [ -s "$f" ] && s5 cp "$f" "s3://$BUCKET/$OUT_PREFIX/train_hybrid/$(basename "$f")" 2>/dev/null || true
  done
done
log "=== Stage B: $STAGE_B ==="
log "dualmodel_runner done (stage_a=$STAGE_A stage_b=$STAGE_B)"
exit 0
