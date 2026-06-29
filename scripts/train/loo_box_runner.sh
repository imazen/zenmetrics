#!/usr/bin/env bash
# Per-box leave-one-out (LOO) feature-ablation runner — the in-container job for the
# fleet-LOO tool (scripts/train/loo_fleet.sh). Runs inside ghcr.io/imazen/zen-train:hybrid-cpu
# (all training tooling baked). One box owns a BATCH of LOO variants; the expensive
# fixed cost (boot + image pull + pareto pull) is paid ONCE, then train_hybrid loops
# over MANY --drop-features variants — that's the whole point of batching.
#
# What a "variant" is: one train_hybrid run with a set of features dropped. The
# baseline (no drop) is just the variant with drop="". LOO importance is computed by
# the collector (loo_collect.py) as overhead(drop f) - overhead(baseline). Round 2
# group-drop / pairwise-joint variants ride the SAME mechanism (drop="feat_a,feat_b,...").
#
# This runner is DATA-SOURCE-AGNOSTIC: the launcher pre-builds the pareto (omni→pareto
# for jpeg/webp/avif, canonical→pareto for jxl) and uploads it; the box just pulls the
# small pareto+features and the config reads them. So one runner serves every codec.
#
# Reuse, not reinvent: mirrors dualmodel_runner.sh's verify-baked-tools + EXIT-trap
# upload + _DONE/_FAILED-marker patterns so the launcher's monitor tears the box down
# the instant work lands. It does NOT touch train_hybrid / bake_picker / zenpredict
# logic (read-only); it only loops the existing --drop-features CLI and parses output.
set -u
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
# Baked code layout (see scripts/train/Dockerfile.hybrid-cpu).
export PYTHONPATH=/opt/picker:/opt/picker/configs:/opt/zentrain/tools:/opt/zentrain/examples
ZENTRAIN=/opt/zentrain/tools          # train_hybrid.py

# ── config from env (cloud-init container env-file) ───────────────────────────
EP="${R2_ENDPOINT:?R2_ENDPOINT required}"
BUCKET="${RUN_BUCKET:-zentrain}"
RUN_PREFIX="${RUN_PREFIX:?RUN_PREFIX required e.g. loo-2026-06-29/zenjpeg_lossy_ssim2}"
BOX_ID="${BOX_ID:?BOX_ID required}"
CONFIG_MODULE="${CONFIG_MODULE:?CONFIG_MODULE required e.g. zenjpeg_picker}"
PARETO_STEM="${PARETO_STEM:?PARETO_STEM required - the config CODEC var, names the pareto file}"
PICKER_TARGET="${PICKER_TARGET:-ssim2_a}"
METRIC_TAG="${METRIC_TAG:-ssim2}"
SEED="${SEED:-12345}"                 # FIXED so the only diff between runs is the dropped feature
HIDDEN="${HIDDEN:-192,192,192}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-25m}"
# Threading: the torch MLP student is the per-run bottleneck and reads OMP_NUM_THREADS.
# Fleet boxes are DEDICATED (exempt from the shared-workstation oversubscription rule), so
# give the MLP all cores (OMP=nproc) and run the teacher's per-cell HistGB fits SERIALLY
# (LOKY=1) — each HistGB fit is itself OpenMP-parallel, so total threads never exceed nproc
# (no oversubscription) while the MLP goes ~nproc× faster than the OMP=1 path.
OMP_THREADS="${OMP_THREADS:-$(nproc)}"
LOKY_WORKERS="${LOKY_WORKERS:-1}"

WORK=/work; mkdir -p "$WORK"
PP=/home/lilith/picker-pp; mkdir -p "$PP/train" "$PP/models"
LOG="$WORK/loo_box_${BOX_ID}.log"
RESJSON="$WORK/box-${BOX_ID}.json"
RESJSONL="$WORK/box-${BOX_ID}.results.jsonl"
: > "$RESJSONL"

s5(){ s5cmd --endpoint-url="$EP" "$@"; }
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

NRUN=0; NOK=0; NFAIL=0; STATE="init"
NKEEP=0; PARETO_CELLS=0; STARTED=""   # defaults so the EXIT trap's assemble_json is set -u safe
on_exit(){
  local rc=$?
  assemble_json "$rc" || true
  s5 cp "$LOG"     "s3://$BUCKET/$RUN_PREFIX/logs/box-${BOX_ID}.log"  2>>"$LOG" || true
  [ -s "$RESJSON" ] && s5 cp "$RESJSON" "s3://$BUCKET/$RUN_PREFIX/results/box-${BOX_ID}.json" 2>>"$LOG" || true
  # marker keys the launcher's monitor polls to tear the box down. DONE if any result landed.
  if [ "$NOK" -gt 0 ]; then
    printf 'box=%s runs=%s ok=%s fail=%s state=%s\n' "$BOX_ID" "$NRUN" "$NOK" "$NFAIL" "$STATE" > "$WORK/_DONE.$BOX_ID"
    s5 cp "$WORK/_DONE.$BOX_ID" "s3://$BUCKET/$RUN_PREFIX/markers/_DONE.box-${BOX_ID}" 2>>"$LOG" || true
  else
    printf 'box=%s runs=%s ok=0 fail=%s state=%s rc=%s\n' "$BOX_ID" "$NRUN" "$NFAIL" "$STATE" "$rc" > "$WORK/_FAILED.$BOX_ID"
    s5 cp "$WORK/_FAILED.$BOX_ID" "s3://$BUCKET/$RUN_PREFIX/markers/_FAILED.box-${BOX_ID}" 2>>"$LOG" || true
  fi
}
trap on_exit EXIT

# assemble box-<id>.json from the incremental JSONL + run metadata (best-effort)
assemble_json(){
  local rc="$1"
  python3 - "$RESJSONL" "$RESJSON" "$BOX_ID" "$CONFIG_MODULE" "$PARETO_STEM" "$METRIC_TAG" \
           "$PICKER_TARGET" "$SEED" "$NKEEP" "$PARETO_CELLS" "$STARTED" "$rc" <<'PY' 2>>"$LOG" || true
import json, sys, socket, datetime
jsonl, out, box, cfg, stem, metric, tgt, seed, nkeep, cells, started, rc = sys.argv[1:13]
rows = []
try:
    with open(jsonl) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
except FileNotFoundError:
    pass
doc = {
    "box_id": int(box), "codec_config": cfg, "pareto_stem": stem, "metric": metric,
    "picker_target": tgt, "seed": int(seed),
    "n_keep_features": int(nkeep) if nkeep.isdigit() else None,
    "pareto_cells": int(cells) if cells.isdigit() else None,
    "host": socket.gethostname(),
    "started_utc": started, "finished_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "runner_rc": int(rc), "n_results": len(rows), "results": rows,
}
with open(out, "w") as f:
    json.dump(doc, f, indent=1)
print(f"assembled {out}: {len(rows)} results")
PY
}

exec > >(tee -a "$LOG") 2>&1
STARTED="$(date -u +%FT%TZ)"
log "loo_box_runner start: box=$BOX_ID config=$CONFIG_MODULE stem=$PARETO_STEM target=$PICKER_TARGET metric=$METRIC_TAG seed=$SEED hidden=$HIDDEN"
log "endpoint=$EP bucket=$BUCKET run_prefix=$RUN_PREFIX nproc=$(nproc)"

# ── verify baked tooling — FAIL LOUD, never install at boot (bake-everything) ──
STATE="verify"
fail=0
for t in s5cmd python3 timeout; do command -v "$t" >/dev/null 2>&1 || { log "FATAL baked tool '$t' MISSING — image broken"; fail=1; }; done
[ -s "$ZENTRAIN/train_hybrid.py" ] || { log "FATAL train_hybrid.py MISSING at $ZENTRAIN — image broken"; fail=1; }
python3 - <<'PY' || fail=1
import sys
try:
    import torch, sklearn, pandas, pyarrow, numpy  # noqa
    print(f"  ML stack OK: torch={torch.__version__} sklearn={sklearn.__version__} pandas={pandas.__version__}")
except Exception as e:
    print(f"FATAL ML stack import failed: {e}", file=sys.stderr); sys.exit(1)
PY
[ "$fail" = 0 ] || { log "aborting: baked tooling incomplete"; exit 3; }
log "baked tooling verified"

# ── pull the pre-built pareto + features to the EXACT paths the config expects ─
# (picker_config_common.paths(CODEC) -> $PP/train/{CODEC}.{PICKER_TARGET}.pareto.parquet
#  + {CODEC}.features.tsv; KEEP_FEATURES is computed from the features TSV at import.)
STATE="fetch-data"
PARQ="$PP/train/${PARETO_STEM}.${PICKER_TARGET}.pareto.parquet"
FEAT="$PP/train/${PARETO_STEM}.features.tsv"
log "fetch pareto + features"
s5 cp "s3://$BUCKET/$RUN_PREFIX/inputs/${PARETO_STEM}.${PICKER_TARGET}.pareto.parquet" "$PARQ" \
  || { log "FATAL pareto fetch failed"; exit 4; }
s5 cp "s3://$BUCKET/$RUN_PREFIX/inputs/${PARETO_STEM}.features.tsv" "$FEAT" \
  || { log "FATAL features fetch failed"; exit 4; }
PARETO_CELLS="$(python3 -c "import pyarrow.parquet as p;print(p.ParquetFile('$PARQ').metadata.num_rows)" 2>/dev/null || echo 0)"
log "pareto cells=$PARETO_CELLS"

# ── pull this box's batch of variants ─────────────────────────────────────────
STATE="fetch-batch"
s5 cp "s3://$BUCKET/$RUN_PREFIX/batches/box-${BOX_ID}.jsonl" "$WORK/batch.jsonl" \
  || { log "FATAL batch fetch failed"; exit 5; }
NJOBS="$(grep -c . "$WORK/batch.jsonl" || echo 0)"
log "batch: $NJOBS variant(s)"

# ── sanity: import the config under this PICKER_TARGET → KEEP_FEATURES count ───
STATE="config-sanity"
NKEEP="$(PICKER_TARGET="$PICKER_TARGET" python3 -c \
  "import importlib;m=importlib.import_module('$CONFIG_MODULE');print(len(m.KEEP_FEATURES))" 2>>"$LOG" || echo 0)"
log "config $CONFIG_MODULE KEEP_FEATURES=$NKEEP"
[ "$NKEEP" -gt 0 ] 2>/dev/null || { log "FATAL config import / KEEP_FEATURES empty"; exit 6; }

# ── parser: extract overheads from one train_hybrid run log (stderr capture) ──
cat > "$WORK/parse_run.py" <<'PY'
import json, re, sys
out_file, tag, drop_csv, secs, rc = sys.argv[1:6]
txt = open(out_file, errors="replace").read() if out_file else ""
def grab(pat, g=1):
    m = re.search(pat, txt)
    return float(m.group(g)) if m else None
# raw val (always printed): "Student metrics: argmin mean overhead X% argmin_acc Y%"
val_raw   = grab(r"Student metrics: argmin mean overhead ([0-9.]+)%")
val_acc   = grab(r"Student metrics: argmin mean overhead [0-9.]+% argmin_acc ([0-9.]+)%")
# deployed val after knob-vetoes (optional): "after N knob-veto(s): val mean overhead X%"
val_veto  = grab(r"knob-veto\(s\): val mean overhead ([0-9.]+)%")
# held-out TEST (7/9 origins): "TEST (7/9 origins): argmin mean X% argmin_acc Y%"
test_ov   = grab(r"TEST \(7/9 origins\): argmin mean ([0-9.]+)%")
test_acc  = grab(r"TEST \(7/9 origins\): argmin mean [0-9.]+% argmin_acc ([0-9.]+)%")
# feature count actually trained on: "Loaded N cells × M features"
nfeat     = grab(r"Loaded [0-9]+ cells\D+([0-9]+) features")
row = {
    "tag": tag,
    "drop": [d for d in drop_csv.split(",") if d],
    "n_dropped": len([d for d in drop_csv.split(",") if d]),
    "n_features": int(nfeat) if nfeat is not None else None,
    "val_overhead": val_raw, "val_argmin_acc": val_acc,
    "val_overhead_vetoed": val_veto,
    "test_overhead": test_ov, "test_argmin_acc": test_acc,
    "train_secs": float(secs), "rc": int(rc),
}
print(json.dumps(row))
PY

# ── the LOO loop ──────────────────────────────────────────────────────────────
STATE="loop"
mapfile -t JOBS < "$WORK/batch.jsonl"
for job in "${JOBS[@]}"; do
  [ -z "$job" ] && continue
  tag="$(echo "$job"  | python3 -c 'import json,sys;print(json.load(sys.stdin)["tag"])' 2>/dev/null)"
  drop="$(echo "$job" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("drop",""))' 2>/dev/null)"
  [ -z "$tag" ] && { log "skip unparseable job: $job"; continue; }
  NRUN=$((NRUN+1))
  drop_args=(); [ -n "$drop" ] && drop_args=(--drop-features "$drop")
  log "[$NRUN/$NJOBS] train_hybrid tag=$tag drop='${drop:-<baseline>}'"
  t0=$(date +%s)
  # No CI env (so --strict is off); --allow-unsafe lets it finish past safety gates and
  # still print the overhead. Per-tag out-json/out-log are throwaway (deleted after).
  # OMP pinned to 1 + loky over all cores (the omp-oversubscription trap).
  PICKER_TARGET="$PICKER_TARGET" CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$OMP_THREADS" \
    OPENBLAS_NUM_THREADS="$OMP_THREADS" NUMEXPR_NUM_THREADS="$OMP_THREADS" \
    LOKY_MAX_CPU_COUNT="$LOKY_WORKERS" \
    timeout "$PER_RUN_TIMEOUT" python3 "$ZENTRAIN/train_hybrid.py" \
      --codec-config "$CONFIG_MODULE" \
      --activation leakyrelu --hidden "$HIDDEN" --seed "$SEED" --allow-unsafe \
      --out-json "$WORK/m_$tag.json" --out-log "$WORK/m_$tag.log" \
      "${drop_args[@]}" > "$WORK/run_$tag.out" 2>&1
  rc=$?
  secs=$(( $(date +%s) - t0 ))
  python3 "$WORK/parse_run.py" "$WORK/run_$tag.out" "$tag" "$drop" "$secs" "$rc" >> "$RESJSONL"
  ov=$(tail -1 "$RESJSONL" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("val_overhead"))' 2>/dev/null)
  if [ "$rc" = 0 ] && [ "$ov" != "None" ] && [ -n "$ov" ]; then NOK=$((NOK+1)); else NFAIL=$((NFAIL+1)); fi
  log "  -> rc=$rc val_overhead=$ov secs=${secs}s (ok=$NOK fail=$NFAIL)"
  # incremental upload (robust to a later kill) + cleanup the throwaway model
  assemble_json 0 >/dev/null 2>&1
  [ -s "$RESJSON" ] && s5 cp "$RESJSON" "s3://$BUCKET/$RUN_PREFIX/results/box-${BOX_ID}.json" 2>>"$LOG" || true
  rm -f "$WORK/m_$tag.json" "$WORK/m_$tag.json.manifest.json" "$WORK/m_$tag.log" "$WORK/run_$tag.out"
done

STATE="done"
log "loo_box_runner done: runs=$NRUN ok=$NOK fail=$NFAIL"
exit 0
