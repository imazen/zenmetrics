#!/usr/bin/env bash
# cuda-select.sh — pick CUDA 12 or 13 at RUNTIME, from the GPU actually present.
#
# WHY THIS EXISTS
# CUDA Toolkit 13.0 removed offline-compilation support for Maxwell, Pascal and Volta:
# the minimum is now Turing (sm_75). Our fleet straddles that line —
#
#     GTX 1050 / GTX 1060   sm_61  Pascal   -> CUDA 12 ONLY, 13 cannot target them
#     GTX 1660 Ti           sm_75  Turing   -> either
#     RTX 2080              sm_75  Turing   -> either
#     RTX 3070              sm_86  Ampere   -> either
#
# so one baked toolkit cannot serve every node. A CUDA 13 image silently loses the
# Pascal boxes; a CUDA 12 image cannot target future Blackwell parts.
#
# HOW SELECTION WORKS (no rebuild needed)
# cudarc loads NVRTC by dlopen, and its FIRST candidate is the UNVERSIONED `libnvrtc.so`
# (cudarc-0.19.9 `get_lib_name_candidates`, src/lib.rs:204). dlopen resolves that through
# LD_LIBRARY_PATH, and the image already sets LD_LIBRARY_PATH=/usr/local/cuda/lib64 with
# /usr/local/cuda a symlink. So re-pointing that symlink chooses the toolkit, and the
# binary itself is untouched — it stays built with CUDARC_CUDA_VERSION=12000, which is
# fine because the DRIVER api is backward compatible; only NVRTC is version-sensitive.
#
# DECISION
#   no GPU                      -> leave everything alone, exit 0 (CPU nodes are normal)
#   compute cap < 7.5           -> CUDA 12  (hard constraint, 13 cannot compile for it)
#   driver advertises CUDA < 13 -> CUDA 12  (13's PTX would not load)
#   otherwise                   -> CUDA 13 if installed, else CUDA 12
#
# Override with ZEN_CUDA_MAJOR=12|13 when you need to force one.
#
#   source cuda-select.sh     # exports the choice into the current shell
#   bash   cuda-select.sh     # just prints what it would pick
set -uo pipefail

_cs_log() { echo "cuda-select: $*" >&2; }

_cs_pick() {
  local forced="${ZEN_CUDA_MAJOR:-}"
  if [ -n "$forced" ]; then _cs_log "forced by ZEN_CUDA_MAJOR=$forced"; echo "$forced"; return; fi

  command -v nvidia-smi >/dev/null 2>&1 || { echo none; return; }
  local cc drv_cuda
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  [ -n "$cc" ] || { echo none; return; }

  # driver's max supported CUDA api, e.g. "13.0" — authoritative, no hardcoded driver numbers
  drv_cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9]*\).*/\1/p' | head -1)

  local cc_major cc_minor drv_major
  cc_major=${cc%%.*}; cc_minor=${cc##*.}
  drv_major=${drv_cuda%%.*}

  # Turing (7.5) is CUDA 13's floor
  if [ "${cc_major:-0}" -lt 7 ] || { [ "${cc_major:-0}" -eq 7 ] && [ "${cc_minor:-0}" -lt 5 ]; }; then
    _cs_log "compute cap $cc is below CUDA 13's sm_75 floor -> CUDA 12"; echo 12; return
  fi
  if [ -n "$drv_major" ] && [ "$drv_major" -lt 13 ]; then
    _cs_log "driver advertises CUDA $drv_cuda (<13) -> CUDA 12"; echo 12; return
  fi
  if ls -d /usr/local/cuda-13* >/dev/null 2>&1; then
    _cs_log "compute cap $cc, driver CUDA ${drv_cuda:-?} -> CUDA 13"; echo 13; return
  fi
  _cs_log "CUDA 13 not installed in this image -> CUDA 12"; echo 12
}

_cs_apply() {
  local major="$1" dir
  [ "$major" = none ] && { _cs_log "no GPU detected; leaving CUDA env untouched"; return 0; }
  dir=$(ls -d /usr/local/cuda-"$major".* 2>/dev/null | sort -V | tail -1)
  if [ -z "$dir" ]; then
    # DISTINCT from "no GPU". A GPU is present and we know which toolkit it needs, but that
    # toolkit is not in this image — a build/packaging fault, not a benign CPU box. Say so
    # unmistakably, or the caller treats a broken image as "nothing to do".
    export ZEN_CUDA_SELECTED="missing-$major"
    _cs_log "FAULT: GPU needs CUDA $major but no /usr/local/cuda-$major.* exists here."
    _cs_log "       GPU work will fail or silently use the wrong toolkit. Rebuild the image"
    _cs_log "       with both toolkits, or install cuda-nvrtc-$major-x + cuda-nvrtc-dev-$major-x."
    return 1
  fi
  # Prefer LD_LIBRARY_PATH over re-symlinking: works unprivileged and per-process, so two
  # containers on one host can disagree without fighting over a shared symlink.
  export CUDA_PATH="$dir"
  export LD_LIBRARY_PATH="$dir/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export ZEN_CUDA_SELECTED="$major"
  _cs_log "selected CUDA $major at $dir"
  if ! ls "$dir"/lib64/libnvrtc.so >/dev/null 2>&1; then
    _cs_log "WARNING: $dir/lib64/libnvrtc.so (unversioned) missing — cudarc's first dlopen"
    _cs_log "         candidate will not resolve here; install the matching -dev package."
  fi
  return 0
}

_CS_MAJOR="$(_cs_pick)"
_cs_apply "$_CS_MAJOR"
# when executed rather than sourced, report the decision for humans/CI
(return 0 2>/dev/null) || { echo "ZEN_CUDA_SELECTED=${ZEN_CUDA_SELECTED:-none}"; echo "CUDA_PATH=${CUDA_PATH:-unset}"; }
