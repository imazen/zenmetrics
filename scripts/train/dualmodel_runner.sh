#!/usr/bin/env bash
# Per-codec dual-model picker runner — the ENTRYPOINT of the zen-train:hybrid-cpu
# image. Runs the per-codec picker pipeline for ONE codec entirely in-container:
#
#   Stage A (NON-GATING): canonical {train,validate,test}.parquet -> prep_combined
#     -> picker_tree_ab (FAST, time-boxed: --skip-mlp + --max-train + --perm-val-cap
#        + --skip-rf so the GBDT + dataset dump finish well inside the box cap) ->
#        cart_analysis CART depth-curve + tree->Rust code-heuristic (.rs).
#   Stage B (NON-GATING): omni sweep TSV + features TSV -> omni_to_pareto ->
#     check_mandatory_coverage -> train_hybrid (sklearn HistGBDT teacher + torch
#     LeakyReLU MLP student) -> bake_picker (ZNPR .bin).
#
# The two deliverables are independent: the CART .rs (Stage A) and the MLP .bin
# (Stage B). The full picker_tree_ab A/B (MLP grid + 469-feature permutation on the
# whole val set) is NOT a deliverable and is what blew the 90-min cap last run while
# GATING Stage B — so Stage A no longer gates, and picker_tree_ab is capped to a fast
# GBDT + dump (the MLP-vs-GBDT comparison comes from train_hybrid's student instead).
# `_DONE` is written when EITHER deliverable uploads; `_FAILED` only when neither does.
#
# ssim2: set SCORE_COL=score_ssim2 (CART) + METRIC_COL=score_ssim2 PICKER_TARGET=ssim2_a
# (MLP). picker_tree_ab hardcodes the "score_zensim" column, so prep_combined renames
# the chosen metric into that slot (no zenanalyze edit). train_hybrid is metric-
# parameterized via omni_to_pareto --metric-col.
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
METRIC_COL="${METRIC_COL:-score_zensim}"          # omni column for the Stage B MLP
SCORE_COL="${SCORE_COL:-score_zensim}"            # canonical-parquet column for the Stage A CART
RUN_ID="${RUN_ID:-dualmodel-$(date +%s)}"
SKIP_TRAIN_HYBRID="${SKIP_TRAIN_HYBRID:-0}"
# ── Stage A speed knobs (defaults make picker_tree_ab finish fast + dump the dataset
#    for the CART; the dump is written LAST in picker_tree_ab, after permutation
#    importance, so the caps — not just the timeout — are what guarantee it lands). ──
AB_SKIP_MLP="${AB_SKIP_MLP:-1}"                   # the MLP comparison comes from train_hybrid's student
AB_SKIP_RF="${AB_SKIP_RF:-${SKIP_RF:-1}}"         # RF baseline is auxiliary; off for the time-boxed fan-out
AB_MAX_TRAIN="${AB_MAX_TRAIN:-8000}"              # subsample train -> fast per-cell GBDT fit
AB_PERM_VAL_CAP="${AB_PERM_VAL_CAP:-2000}"        # cap permutation-importance eval rows
AB_TIMEOUT="${AB_TIMEOUT:-25m}"                   # wall backstop per picker_tree_ab invocation
SKIP_TEST_SPLIT="${SKIP_TEST_SPLIT:-1}"           # val A/B is the gate+CART; test is bonus (skip for speed)

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

STAGE_A="not-run"; STAGE_B="not-run"; CART_UP=0; BIN_UP=0
on_exit(){
  local rc=$?
  {
    echo "=== STATUS ==="
    echo "run_id=$RUN_ID codec=$CODEC family=$CODEC_FAMILY target=$PICKER_TARGET metric=$METRIC_COL score_col=$SCORE_COL"
    echo "stage_a_cart=$STAGE_A   cart_rs_uploaded=$CART_UP"
    echo "stage_b_train_hybrid=$STAGE_B   hybrid_bin_uploaded=$BIN_UP"
    echo "runner_rc=$rc finished=$(date -u +%FT%TZ)"
  } > "$WORK/STATUS.txt"
  # best-effort uploads (trap must never abort)
  s5 cp "$LOG" "s3://$BUCKET/$OUT_PREFIX/runner.log" 2>>"$LOG" || true
  s5 cp "$WORK/STATUS.txt" "s3://$BUCKET/$OUT_PREFIX/STATUS.txt" 2>>"$LOG" || true
  # the launcher's monitor polls for this marker key to tear the box down. The box is
  # creditably DONE if EITHER deliverable (CART .rs or MLP .bin) actually uploaded.
  if [ "$CART_UP" = 1 ] || [ "$BIN_UP" = 1 ]; then
    printf 'cart_rs=%s hybrid_bin=%s stage_a=%s stage_b=%s\n' "$CART_UP" "$BIN_UP" "$STAGE_A" "$STAGE_B" > "$WORK/_DONE"
    s5 cp "$WORK/_DONE" "s3://$BUCKET/$OUT_PREFIX/_DONE" 2>>"$LOG" || true
  else
    printf 'cart_rs=0 hybrid_bin=0 stage_a=%s stage_b=%s rc=%s\n' "$STAGE_A" "$STAGE_B" "$rc" > "$WORK/_FAILED"
    s5 cp "$WORK/_FAILED" "s3://$BUCKET/$OUT_PREFIX/_FAILED" 2>>"$LOG" || true
  fi
}
trap on_exit EXIT

exec > >(tee -a "$LOG") 2>&1
log "dualmodel_runner start: codec=$CODEC family=$CODEC_FAMILY target=$PICKER_TARGET metric=$METRIC_COL score_col=$SCORE_COL run=$RUN_ID"
log "endpoint=$EP bucket=$BUCKET out=$OUT_PREFIX"
log "stageA caps: skip_mlp=$AB_SKIP_MLP skip_rf=$AB_SKIP_RF max_train=$AB_MAX_TRAIN perm_val_cap=$AB_PERM_VAL_CAP timeout=$AB_TIMEOUT skip_test=$SKIP_TEST_SPLIT"

# ── verify baked tools — FAIL LOUD, never install at boot ─────────────────────
fail_missing=0
for t in s5cmd picker_tree_ab zenpredict-bake python3 timeout; do
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
for f in "$PICKERDIR/prep_combined.py" "$PICKERDIR/cart_analysis.py" "$PICKERDIR/omni_to_pareto.py" \
         "$PICKERDIR/origin_split.py" "$PICKERDIR/check_mandatory_coverage.py" \
         "$ZENTRAIN/train_hybrid.py" "$ZATOOLS/bake_picker.py"; do
  [ -s "$f" ] || { log "FATAL: baked code '$f' MISSING — image is broken"; fail_missing=1; }
done
[ "$fail_missing" = 0 ] || { log "aborting: baked tooling incomplete"; exit 3; }
log "baked tooling verified"

# ══════════════════════════════════════════════════════════════════════════════
# Stage A — picker_tree_ab (fast GBDT + dump) -> CART code-heuristic. NON-GATING:
# every failure path `return`s; Stage B always runs afterward.
# ══════════════════════════════════════════════════════════════════════════════
run_stage_a(){
  log "=== Stage A: picker_tree_ab + CART for $CODEC (metric col=$SCORE_COL) ==="
  local CANON_DIR="/data/canon/$CODEC"; mkdir -p "$CANON_DIR"
  local f
  for f in train validate test; do
    log "fetch canonical $f.parquet"
    s5 cp "s3://$BUCKET/$CANON_PREFIX/$CODEC/$f.parquet" "$CANON_DIR/$f.parquet" \
      || { log "canonical $f.parquet fetch failed"; STAGE_A="fetch-failed"; return 1; }
  done
  log "prep_combined (--score-col $SCORE_COL)"
  python3 "$PICKERDIR/prep_combined.py" "$CODEC" --src "$CANON_DIR" --out "$WORK" --score-col "$SCORE_COL" \
    || { log "prep_combined failed"; STAGE_A="prep-failed"; return 1; }

  local DUMP="$WORK/dump_$CODEC"; mkdir -p "$DUMP"
  local SPLITS="val test"; [ "$SKIP_TEST_SPLIT" = "1" ] && SPLITS="val"
  local FAST_FLAGS=(--max-train "$AB_MAX_TRAIN" --perm-val-cap "$AB_PERM_VAL_CAP")
  [ "$AB_SKIP_MLP" = "1" ] && FAST_FLAGS+=(--skip-mlp)
  [ "$AB_SKIP_RF" = "1" ]  && FAST_FLAGS+=(--skip-rf)
  local split rc
  for split in $SPLITS; do
    log "picker_tree_ab --eval-split $split (timeout $AB_TIMEOUT; ${FAST_FLAGS[*]})"
    timeout "$AB_TIMEOUT" picker_tree_ab \
        --input "$WORK/combined_$CODEC.parquet" \
        --split-map "$WORK/splitmap_$CODEC.parquet" \
        --eval-split "$split" \
        --dump-dir "$DUMP/$split" \
        --codec-tag "$CODEC" "${FAST_FLAGS[@]}" 2>&1 | tee "$WORK/ab_${CODEC}_${split}.log"
    rc=${PIPESTATUS[0]}
    # upload this split's artifacts right away (incremental — robust to a later kill)
    s5 cp "$WORK/ab_${CODEC}_${split}.log" "s3://$BUCKET/$OUT_PREFIX/picker_tree_ab/ab_${CODEC}_${split}.log" || true
    if [ -d "$DUMP/$split" ]; then
      tar -cf "$WORK/dump_${CODEC}_${split}.tar" -C "$DUMP" "$split" 2>/dev/null \
        && s5 cp "$WORK/dump_${CODEC}_${split}.tar" "s3://$BUCKET/$OUT_PREFIX/picker_tree_ab/dump_${CODEC}_${split}.tar" || true
    fi
    [ "$rc" = 0 ] || log "picker_tree_ab --eval-split $split exited rc=$rc (timeout=124) — non-gating, continuing"
    if [ "$split" = "val" ]; then
      STAGE_A="ab-done(rc=$rc)"
      # CART fit + tree->Rust code-heuristic on the SAME dumped dataset (same cells/
      # reach/oracle as the A/B -> overhead directly comparable). The dump survives a
      # picker_tree_ab timeout only if it was written before the kill; guard on meta.
      if [ -s "$DUMP/val/${CODEC}_meta.json" ]; then
        log "CART depth-curve + tree->Rust codegen (depth 6) on the val dump"
        python3 "$PICKERDIR/cart_analysis.py" --dump-dir "$DUMP/val" --codec-tag "$CODEC" \
          --eval-split val --codegen-depth 6 --codegen-out "$WORK/${CODEC}_cart_heuristic.rs" \
          2>&1 | tee "$WORK/cart_${CODEC}.log" || log "cart_analysis exited nonzero (non-fatal)"
        s5 cp "$WORK/cart_${CODEC}.log" "s3://$BUCKET/$OUT_PREFIX/picker_tree_ab/cart_${CODEC}.log" || true
        if [ -s "$WORK/${CODEC}_cart_heuristic.rs" ]; then
          s5 cp "$WORK/${CODEC}_cart_heuristic.rs" "s3://$BUCKET/$OUT_PREFIX/picker_tree_ab/${CODEC}_cart_heuristic.rs" \
            && { CART_UP=1; STAGE_A="ok"; log "Stage A deliverable: CART .rs uploaded"; }
          [ -s "$WORK/${CODEC}_cart_heuristic_cases.bin" ] && \
            s5 cp "$WORK/${CODEC}_cart_heuristic_cases.bin" "s3://$BUCKET/$OUT_PREFIX/picker_tree_ab/${CODEC}_cart_heuristic_cases.bin" || true
        else
          log "cart_analysis produced no .rs — CART deliverable missing"
        fi
      else
        log "no dump meta at $DUMP/val/${CODEC}_meta.json (picker_tree_ab killed before dump?) — skipping CART"
      fi
    fi
  done
  log "=== Stage A complete (cart_rs_uploaded=$CART_UP) ==="
  return 0
}

# ══════════════════════════════════════════════════════════════════════════════
# Stage B — train_hybrid teacher/student + bake. NON-GATING (best-effort).
# ══════════════════════════════════════════════════════════════════════════════
run_stage_b(){
  mkdir -p /data/tin
  log "fetch omni=$OMNI_NAME + features=$FEATURES_NAME"
  s5 cp "s3://$BUCKET/$INPUTS_PREFIX/$OMNI_NAME"     "/data/tin/$OMNI_NAME"     || { log "omni fetch failed (codec has no omni -> CART-only; expected for png/jxl)"; return 1; }
  s5 cp "s3://$BUCKET/$INPUTS_PREFIX/$FEATURES_NAME" "/data/tin/$FEATURES_NAME" || { log "features fetch failed"; return 1; }
  # picker_config_common hardcodes PP=/home/lilith/picker-pp and resolves PARETO/FEATURES/OUT
  # paths from it (keyed on PICKER_TARGET); replicate that layout so the config imports
  # cleanly (keep_features opens the FEATURES path at import time).
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
  log "train_hybrid --codec-config ${CODEC_FAMILY}_picker (PICKER_TARGET=$PICKER_TARGET)"
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

# --- run both stages; NEITHER gates the other -------------------------------
run_stage_a || log "Stage A returned nonzero (non-gating) — proceeding to Stage B"

if [ "$SKIP_TRAIN_HYBRID" = "1" ]; then
  log "Stage B skipped (SKIP_TRAIN_HYBRID=1)"; STAGE_B="skipped"
else
  log "=== Stage B: train_hybrid for family=$CODEC_FAMILY target=$PICKER_TARGET ==="
  if run_stage_b; then STAGE_B="ok"; else rcb=$?; STAGE_B="failed(rc=$rcb)"; fi
  # upload train_hybrid artifacts regardless of pass/fail
  s5 cp "$WORK/train_hybrid.log" "s3://$BUCKET/$OUT_PREFIX/train_hybrid/train_hybrid.log" 2>/dev/null || true
  for ext in json bin log manifest.json; do
    for f in /home/lilith/picker-pp/models/${CODEC_FAMILY}_predict_${PICKER_TARGET}_v0.1.${ext}; do
      if [ -s "$f" ]; then
        s5 cp "$f" "s3://$BUCKET/$OUT_PREFIX/train_hybrid/$(basename "$f")" 2>/dev/null \
          && { [ "$ext" = "bin" ] && { BIN_UP=1; log "Stage B deliverable: MLP .bin uploaded"; }; } || true
      fi
    done
  done
  log "=== Stage B: $STAGE_B (hybrid_bin_uploaded=$BIN_UP) ==="
fi

log "dualmodel_runner done (stage_a=$STAGE_A cart_rs=$CART_UP / stage_b=$STAGE_B hybrid_bin=$BIN_UP)"
exit 0
