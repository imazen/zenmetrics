#!/usr/bin/env bash
# split_score_worker.sh — vast.ai GPU worker for the HETEROGENEOUS SPLIT.
#
# Hetzner CPU boxes encode + persist the variants (hetzner_cpu_sweep.sh,
# --encoded-out-dir → tar/box on R2). This worker is the vast.ai GPU half:
# it pulls those persisted variants + the source renditions, builds nothing
# (the producer uploads a pairs.tsv), and runs `zenmetrics score-pairs` for
# each GPU metric (butteraugli-gpu, cvvdp, ssim2-gpu, zensim-gpu, dssim-gpu)
# over the (ref, dist) pairs — scoring work already-encoded variants WITHOUT
# re-encoding (encode once on cheap CPU, score many GPU metrics on the GPU).
#
# Entry point of the thin SPLIT image (FROM zenmetrics-sweep:v29 + this);
# wrapped by run_with_error_trap.sh so a crash self-destroys the box.
#
# Env (vast injects into /proc/1/environ):
#   R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY  REQUIRED
#   R2_SESSION_TOKEN     optional (scoped temp creds)
#   ZEN_BUCKET           REQUIRED  R2 bucket (e.g. codec-corpus)
#   ZEN_RUN_PREFIX       REQUIRED  prefix holding variants/ ref/ pairs.tsv; sidecars/ written here
#   ZEN_METRICS          optional  space list; default the 5 GPU metrics
#   ZEN_R2_ENDPOINT      optional  derived from R2_ACCOUNT_ID
set -uo pipefail
if [[ -r /proc/1/environ ]]; then
  while IFS='=' read -r -d '' k v; do case "$k" in R2_*|ZEN_*|AWS_*|CONTAINER_*) export "$k=$v";; esac; done < /proc/1/environ
fi
: "${R2_ACCOUNT_ID:?}"; : "${R2_ACCESS_KEY_ID:?}"; : "${R2_SECRET_ACCESS_KEY:?}"
EP="${ZEN_R2_ENDPOINT:-https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com}"
BUCKET="${ZEN_BUCKET:?ZEN_BUCKET missing}"; PRE="${ZEN_RUN_PREFIX:?ZEN_RUN_PREFIX missing}"
METRICS="${ZEN_METRICS:-butteraugli-gpu cvvdp ssim2-gpu zensim-gpu dssim-gpu}"; METRICS="${METRICS//,/ }"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto
[ -n "${R2_SESSION_TOKEN:-}" ] && export AWS_SESSION_TOKEN="$R2_SESSION_TOKEN"
s5(){ s5cmd --endpoint-url "$EP" "$@"; }
mkdir -p /data/variants /data/ref
echo "[split] worker=$(hostname) pull s3://$BUCKET/$PRE/ metrics='$METRICS'" >&2
# variants may be loose files or per-box tarballs (hetzner_cpu_sweep persists tarballs)
s5 cp "s3://$BUCKET/$PRE/variants/*" /data/variants/ 2>/dev/null || true
for t in /data/variants/*.tar; do [ -f "$t" ] && tar -xf "$t" -C /data/variants/ && rm -f "$t"; done
s5 cp "s3://$BUCKET/$PRE/ref/*" /data/ref/ 2>/dev/null || true
s5 cp "s3://$BUCKET/$PRE/pairs.tsv" /data/pairs.tsv
echo "[split] $(wc -l < /data/pairs.tsv) pair rows; $(ls /data/variants | wc -l) variants; $(ls /data/ref | wc -l) refs" >&2
rc=0
for m in $METRICS; do
  echo "[split] score-pairs --metric $m" >&2
  if zenmetrics score-pairs --metric "$m" --pairs-tsv /data/pairs.tsv --out-parquet "/data/sc_$m.parquet"; then
    s5 cp "/data/sc_$m.parquet" "s3://$BUCKET/$PRE/sidecars/$m.parquet"
  else rc=1; echo "[split] $m FAILED" >&2; fi
done
printf 'rc=%s metrics=%s\n' "$rc" "$METRICS" > /data/DONE
s5 cp /data/DONE "s3://$BUCKET/$PRE/sidecars/DONE"
echo "[split] done rc=$rc" >&2
# best-effort self-destroy on SUCCESS too (run_with_error_trap only fires on failure)
if [ "$rc" = 0 ] && [ -n "${CONTAINER_ID:-}" ] && command -v zenfleet-vastai >/dev/null 2>&1; then
  zenfleet-vastai self-destroy 2>/dev/null || true
fi
exit "$rc"
