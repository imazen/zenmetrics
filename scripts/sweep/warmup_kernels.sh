#!/usr/bin/env bash
# warmup_kernels.sh — pre-compile every cubecl-CUDA kernel the Salad
# worker will use, BEFORE the worker starts accepting jobs.
#
# What this does:
#   1. mkdir -p the cubecl PTX cache dir (referenced from cubecl.toml
#      in WORKDIR; the cubecl-cuda context reads that config and
#      writes compiled PTX to <root>/cuda/<ver>/<sha>/sm_<arch>/<driver>/
#      ptx.json.log on the first kernel-launch).
#   2. Run `zenmetrics score-pairs --metric <each-default-GPU-metric>`
#      once on the baked 64×64 (and 256×256 for iwssim) fixtures. The
#      first run for each metric triggers NVRTC compile of every
#      kernel the metric uses; cubecl persists the resulting PTX to
#      the cache dir.
#   3. Report wall-clock time per metric so the entrypoint can log
#      the per-worker boot-cost breakdown.
#
# Why this matters:
#   Fresh containers see a 10–90s NVRTC compile burst on the first
#   real job (it's been measured at ~18s for ssim2-gpu alone on a
#   3060). When this happens IN-LINE with the first sidecar-POST'd
#   job, the worker appears slow / stuck for that window — and the
#   first job's wall time is poisoned. By doing the compile here,
#   under the entrypoint, the cost moves to a bracketed pre-job
#   phase that's diagnosable in the boot log.
#
#   On a node that restarts the same container, the cache dir
#   survives (it's in the container's writable layer); subsequent
#   warmup passes hit the disk cache and complete in <1s/metric.
#
# Why fail-soft on warmup errors:
#   If a kernel fails to compile (driver mismatch, OOM, etc.) the
#   real job would have failed anyway — we'd rather see THAT failure
#   from the real job path, which has the durable error-sidecar.
#   Warmup failures are logged warn-level; the script exits 0 so the
#   entrypoint goes on to launch the worker. If kernels are
#   genuinely broken the worker will surface the error properly.

set -uo pipefail

log() { echo "[warmup] $*" >&2; }

WARMUP_DIR="${WARMUP_DIR:-/opt/zen/warmup}"
CACHE_DIR="${CUBECL_CACHE_DIR:-/var/cache/cubecl}"
METRICS_TO_WARM="${WARMUP_METRICS:-zensim-gpu,ssim2-gpu,butteraugli-gpu,cvvdp,dssim-gpu,iwssim-gpu}"
GPU_RUNTIME="${WARMUP_GPU_RUNTIME:-cuda}"

mkdir -p "${CACHE_DIR}"

if ! command -v zenmetrics >/dev/null 2>&1; then
    log "FATAL: zenmetrics not on PATH"
    exit 0  # fail-soft per docstring
fi

if [[ ! -d "${WARMUP_DIR}" ]]; then
    log "warmup fixtures not found at ${WARMUP_DIR}; skipping"
    exit 0
fi

# Pairs TSV: one row per (ref, dist). Use the noisy 64×64 distorted
# (not identical) so non-trivial code paths fire — some kernels short-
# circuit on identical inputs.
SMALL_PAIRS="${WARMUP_DIR}/small_pairs.tsv"
{
    printf 'ref_path\tdist_path\n'
    printf '%s\t%s\n' "${WARMUP_DIR}/ref_64.png" "${WARMUP_DIR}/dist_noisy_64.png"
} > "${SMALL_PAIRS}"

# iwssim-gpu needs min(W,H) >= 176; use the 256×256 fixture for it.
LARGE_PAIRS="${WARMUP_DIR}/large_pairs.tsv"
{
    printf 'ref_path\tdist_path\n'
    printf '%s\t%s\n' "${WARMUP_DIR}/ref_256.png" "${WARMUP_DIR}/dist_noisy_256.png"
} > "${LARGE_PAIRS}"

# nvidia-smi sanity: log GPU info before warmup so per-node-arch can
# be cross-referenced with the cache dir layout post-warmup.
if command -v nvidia-smi >/dev/null 2>&1; then
    arch_line=$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>&1 | head -1)
    log "GPU: ${arch_line}"
else
    log "nvidia-smi unavailable; proceeding (CPU runtime path)"
fi

total_start_ns=$(date +%s%N)
overall_rc=0

IFS=',' read -ra metrics <<< "${METRICS_TO_WARM}"
for metric in "${metrics[@]}"; do
    metric=$(echo "${metric}" | tr -d ' ')
    [[ -z "${metric}" ]] && continue

    # Pick fixture size: iwssim-gpu needs >=176px, all others fine at 64.
    if [[ "${metric}" == "iwssim-gpu" || "${metric}" == "iwssim" ]]; then
        pairs="${LARGE_PAIRS}"
    else
        pairs="${SMALL_PAIRS}"
    fi
    out_parquet="/tmp/warmup_${metric}.parquet"

    metric_start_ns=$(date +%s%N)
    # 2>&1 to capture cubecl compile messages; we tee to a per-metric
    # log so a crash leaves diagnostic state on disk.
    log_path="/tmp/warmup_${metric}.log"
    # `score-pairs` accepts multi-column metrics (butteraugli emits
    # max+pnorm3, future metrics may emit more) — one parquet column
    # per metric column. Same subcommand for every warmup metric.
    if zenmetrics score-pairs \
            --metric "${metric}" \
            --gpu-runtime "${GPU_RUNTIME}" \
            --pairs-tsv "${pairs}" \
            --out-parquet "${out_parquet}" \
            >"${log_path}" 2>&1
    then
        metric_end_ns=$(date +%s%N)
        elapsed_s=$(awk -v s="${metric_start_ns}" -v e="${metric_end_ns}" 'BEGIN { printf "%.2f", (e-s)/1e9 }')
        log "  ${metric}: OK in ${elapsed_s}s"
    else
        rc=$?
        metric_end_ns=$(date +%s%N)
        elapsed_s=$(awk -v s="${metric_start_ns}" -v e="${metric_end_ns}" 'BEGIN { printf "%.2f", (e-s)/1e9 }')
        # Pull the last line of the metric log for a quick summary.
        tail_line=$(tail -1 "${log_path}" 2>/dev/null | head -c 200)
        log "  ${metric}: FAIL rc=${rc} after ${elapsed_s}s (tail: ${tail_line})"
        overall_rc=1
    fi
    rm -f "${out_parquet}"
done

total_end_ns=$(date +%s%N)
total_s=$(awk -v s="${total_start_ns}" -v e="${total_end_ns}" 'BEGIN { printf "%.2f", (e-s)/1e9 }')

# Stat the cache dir so the boot log shows how much PTX was written.
if [[ -d "${CACHE_DIR}" ]]; then
    cache_size=$(du -sh "${CACHE_DIR}" 2>/dev/null | awk '{print $1}')
    cache_entries=$(find "${CACHE_DIR}" -type f 2>/dev/null | wc -l | tr -d ' ')
    log "cubecl cache: ${cache_size} in ${cache_entries} file(s) at ${CACHE_DIR}"
fi

log "warmup total: ${total_s}s (overall_rc=${overall_rc}; fail-soft, continuing)"
# Always exit 0 — see docstring (fail-soft).
exit 0
