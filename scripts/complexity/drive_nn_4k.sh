#!/usr/bin/env bash
# Per-metric 3840x2160 NN holds on i265 — ONE flock+run-heavy+perf_deltas per metric.
# Lead directive 2026-10-18: each lock hold <20 min; metrics whose measurement exceeds
# that run --repeats 1 (noted in meta); per-call timeout stays 900 s.
# Resume policy: enough samples OR an error row settles an attempt. Error rows
# remain unqualified. Retry only interrupted holds without a terminal row.
# Usage: bash scripts/complexity/drive_nn_4k.sh rten|cuda|webgpu|all
# Each attempt gets a unique log; existing result rows are retained.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${CX4K_REPO:-$SCRIPT_DIR/../..}"
LOCK=${CX4K_LOCK:-$HOME/tmp/i265-heavy.lock}
RUNH=${CX4K_RUN_HEAVY:-$HOME/work/claudehints/scripts/run-heavy}
P=${CX4K_PERF_DRIVER:-$SCRIPT_DIR/perf_deltas.py}
MM=${CX4K_MISSING_DRIVER:-$SCRIPT_DIR/missing_metrics.py}
O=${CX4K_OUTPUT:-$HOME/tmp/cx-4k/perf}
LG=${CX4K_LOGS:-$HOME/tmp/cx-4k/logs}; mkdir -p "$LG" "$O"
HOLD=${CX4K_HOLD_SECONDS:-1140}   # 19-min hard cap on the measured section of each hold
export ZENMETRICS_NN_MODEL_DIR=${ZENMETRICS_NN_MODEL_DIR:-$HOME/tmp/iqa-cx-lead/nn-models}
export ZENMETRICS_NN_ACCEPT_WEIGHTS=ckdn,afine,dreamsim,liqe,hyperiqa,paq2piq,pieapp
export ORT_DISABLE_TELEMETRY=1

M_RTEN=${CX4K_METRICS_RTEN:-"afine,ahiq,arniqa,ckdn,deepdc,dists,dmm,dreamsim,hyperiqa,liqe,lpips-alex,lpips-squeeze,lpips-vgg,maniqa,musiq,paq2piq,stlpips-alex,stlpips-vgg,wadiqam-fr-live,wadiqam-fr-tid,wadiqam-nr-live,wadiqam-nr-tid,wd"}
M_ENG=${CX4K_METRICS_ENG:-"afine,arniqa,ckdn,deepdc,dists,dmm,hyperiqa,liqe,lpips-alex,lpips-squeeze,lpips-vgg,maniqa,musiq,paq2piq,stlpips-alex,stlpips-vgg,wadiqam-fr-live,wadiqam-fr-tid,wadiqam-nr-live,wadiqam-nr-tid,wd,dreamsim"}
BIN_NN=${CX4K_BIN_NN:-$HOME/tmp/iqa-cx-lead/bin/complexity-reps-nn}
BIN_ORT=${CX4K_BIN_ORT:-$HOME/tmp/iqa-cx-lead/bin/complexity-reps-ort}

engine_env() {  # $1 = ort-cuda|ort-webgpu
  export LOCK_SNAP_DIR=$HOME/tmp/iqa-cx-lead/snap
  SITE=$HOME/tmp/nnrt-research/venv-cu/lib/python3.12/site-packages
  export ORT_DYLIB_PATH=$SITE/onnxruntime/capi/libonnxruntime.so.1.31.0
  export LD_LIBRARY_PATH=$(printf '%s:' $SITE/nvidia/*/lib)${LD_LIBRARY_PATH:-}
  export ZENMETRICS_NN_WEBGPU_EP_LIB=$HOME/tmp/webgpu-ep/wheel/onnxruntime_ep_webgpu/libonnxruntime_providers_webgpu.so
  export ZENMETRICS_NN_WEBGPU_DEVICE=1
  export ZENMETRICS_NN_ENGINE=$1
}

hold() {  # bin out metric repeats extra target_s engine modeldir
  local bin=$1 out=$2 m=$3 rep=$4 ex=$5 tgt=$6 eng=$7 mdir=$8
  local log
  log=$(mktemp "$LG/$(basename "$out" .tsv).$m.XXXXXX.log")
  local rc=0
  LAST_HOLD_LOG=$log
  local -a target=()
  if [ -n "$tgt" ]; then target=(--target-s "$tgt"); fi
  {
    echo "== hold start_utc=$(date -u +%FT%TZ) metric=$m repeats=$rep extra=$ex engine=${eng:-rten} =="
    if [ -n "$eng" ]; then engine_env "$eng"; fi
    if [ -n "$mdir" ]; then export ZENMETRICS_NN_MODEL_DIR=$mdir; fi
    flock "$LOCK" "$RUNH" --mem 14G --jobs 8 -- \
      timeout --signal=TERM --kill-after=20s ${HOLD}s python3 "$P" \
      --bin "$bin" --out "$out" --shapes 3840x2160 --metrics "$m" \
      --repeats "$rep" --extra "$ex" "${target[@]}" \
      --cpu 0-7 --threads 8 --timeout-s 900 || rc=$?
    echo "== hold end_utc=$(date -u +%FT%TZ) rc=$rc =="
  } > "$log" 2>&1
  HOLD_RC=$rc
  tail -2 "$log"
  if [ "$rc" = 130 ] || [ "$rc" = 143 ]; then return "$rc"; fi
}

error_row() {  # out metric threads extra reason
  python3 - "$P" "$@" <<'PYERR'
import importlib.util, sys
spec = importlib.util.spec_from_file_location('perf_deltas', sys.argv[1])
perf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf)
perf.append_error(*sys.argv[2:])
PYERR
}

done_check() {  # out metric expected -> empty if enough samples OR terminal failure
  python3 "$MM" --out "$1" --metrics "$2" --threads 8 --repeats "$3"
}

announce_selection() {  # Actual phase list, including explicit overrides.
  local list=$1 metric
  for metric in pieapp pieapp-pyiqa; do
    case ",$list," in
      *,"$metric",*) echo "BENCHMARK EXPLICITLY SELECTED (slow): $metric" >&2 ;;
      *) echo "BENCHMARK OMITTED (slow; explicit CX4K_METRICS_RTEN/CX4K_METRICS_ENG selection required): $metric" >&2 ;;
    esac
  done
}

phase_rten() {
  announce_selection "$M_RTEN"
  local out=$O/i265_nn_3840x2160.tsv
  local pending
  for m in ${M_RTEN//,/ }; do
    pending=$(done_check "$out" "$m" 3)
    [ -z "$pending" ] && continue
    hold "$BIN_NN" "$out" "$m" 3 0 "" "" ""
    pending=$(done_check "$out" "$m" 3)
    if [ -n "$pending" ]; then
      # An interrupted hold produced no terminal row; try a minimal 3-call delta.
      # A recorded error is terminal and never reaches this branch.
      hold "$BIN_NN" "$out" "$m" 1 1 "" "" ""
      pending=$(done_check "$out" "$m" 1)
      if [ -n "$pending" ]; then
        error_row "$out" "$m" 8 1 "incomplete bounded attempts; last exit=$HOLD_RC; log=$LAST_HOLD_LOG"
      fi
    fi
  done
}

phase_engine() {  # $1 = ort-cuda|ort-webgpu
  announce_selection "$M_ENG"
  local e=$1
  local out=$O/i265_nn_${e}_3840x2160.tsv dout=$O/i265_nn_${e}-dreamsim_3840x2160.tsv
  local pending
  for m in ${M_ENG//,/ }; do
    local o=$out mdir="" rep=3 ex=0 exp=3
    if [ "$m" = dreamsim ]; then o=$dout; mdir=$HOME/tmp/nn-phase0/models; fi
    pending=$(done_check "$o" "$m" "$exp")
    [ -z "$pending" ] && continue
    hold "$BIN_ORT" "$o" "$m" "$rep" "$ex" 3 "$e" "$mdir"
    pending=$(done_check "$o" "$m" "$exp")
    if [ -n "$pending" ]; then
      hold "$BIN_ORT" "$o" "$m" 1 1 3 "$e" "$mdir"
      pending=$(done_check "$o" "$m" 1)
      if [ -n "$pending" ]; then
        error_row "$o" "$m" 8 1 "incomplete bounded attempts; last exit=$HOLD_RC; log=$LAST_HOLD_LOG"
      fi
    fi
  done
}

case "${1:-all}" in
  rten)   phase_rten ;;
  cuda)   phase_engine ort-cuda ;;
  webgpu) phase_engine ort-webgpu ;;
  all)    phase_rten; phase_engine ort-cuda; phase_engine ort-webgpu ;;
  *) echo "Unknown phase: $1" >&2; exit 2 ;;
esac
echo "drive ${1:-all} done_utc=$(date -u +%FT%TZ)"
