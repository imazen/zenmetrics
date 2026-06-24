#!/usr/bin/env bash
# Unpack a codec's variants.tar -> per-sha R2 objects (artifacts/<sha>.<ext>) so jobexec can FETCH the
# pre-encoded distorted image instead of re-encoding it. Re-encoding dominates fleet time: avif rav1e
# is ~134ms median but 5.8s mean / 76s max for big (24MP) cells, paid 6x per cell (once per metric).
# Streams the tar in one pass (sha + write), then bulk-uploads. Progress -> /tmp/unpack_variants.log.
#   usage: unpack_variants.sh <codec_dir> <ext>   e.g. unpack_variants.sh zenavif avif
set -uo pipefail
CODEC="${1:?codec dir e.g. zenavif}"; EXT="${2:?ext e.g. avif}"
DGP="${ZEN_DATAGEN_PREFIX:-picker-sweep-2026-06-22/datagen-2026-06-23}"
LOG=/tmp/unpack_variants.log; : > "$LOG"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto s5cmd --endpoint-url "$EP" "$@"; }
D="/mnt/v/zen/variants-unpack-$CODEC"; rm -rf "$D"; mkdir -p "$D/out"
log "downloading $CODEC variants.tar"
r2 cp "s3://codec-corpus/$DGP/$CODEC/variants.tar" "$D/variants.tar" >/dev/null 2>&1 || { log "FATAL: download failed"; exit 1; }
log "streaming tar -> $D/out/<sha>.$EXT (one pass: read+sha+write)"
python3 - "$D/variants.tar" "$D/out" "$EXT" "$LOG" <<'PY'
import tarfile, hashlib, sys, time
tar, out, ext, logf = sys.argv[1:5]
n=0; t0=time.time()
with tarfile.open(tar, "r") as tf:
    for m in tf:
        if not m.isfile(): continue
        b = tf.extractfile(m).read()
        sha = hashlib.sha256(b).hexdigest()
        with open(f"{out}/{sha}.{ext}", "wb") as f: f.write(b)
        n += 1
        if n % 5000 == 0:
            with open(logf,"a") as L: L.write(f"  streamed {n} ({n/(time.time()-t0):.0f}/s)\n")
with open(logf,"a") as L: L.write(f"  streamed {n} total in {time.time()-t0:.0f}s\n")
print(n)
PY
N=$(ls "$D/out" | wc -l)
log "wrote $N per-sha variants; uploading to s3://codec-corpus/$DGP/$CODEC/artifacts/"
r2 cp "$D/out/" "s3://codec-corpus/$DGP/$CODEC/artifacts/" >/dev/null 2>&1 || { log "FATAL: upload failed"; exit 1; }
UP=$(r2 ls "s3://codec-corpus/$DGP/$CODEC/artifacts/" 2>/dev/null | grep -c .)
log "DONE: $UP variants at s3://codec-corpus/$DGP/$CODEC/artifacts/ (local $N)"
rm -rf "$D"
