#!/usr/bin/env bash
# Watch a running fleet: claims taken + ledger DONE rows grouped by provider (proves which tiers are
# concurrently working the one queue).  Usage: bash scripts/jobsys/watch_fleet.sh <RUN>
set -uo pipefail
RUN="${1:?usage: watch_fleet.sh <RUN>}"
B="${ZEN_FLEET_BUCKET:-zen-tuning-ephemeral}"
set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto
echo "claims taken: $(s5cmd --endpoint-url "$EP" ls "s3://$B/$RUN/claims/" 2>/dev/null | grep -vc spec/)"
d=$(mktemp -d); s5cmd --endpoint-url "$EP" cp "s3://$B/$RUN/ledger/*" "$d/" >/dev/null 2>&1 || true
python3 - "$d" <<'PY'
import glob, sys, collections, pyarrow.parquet as pq
c=collections.Counter()
for f in glob.glob(sys.argv[1]+"/*.parquet"):
    try:
        t=pq.read_table(f)
        if "provider" in t.schema.names:
            for p in t.column("provider").to_pylist(): c[p]+=1
    except Exception: pass
print("ledger DONE rows by provider:", dict(c) or "(none yet)")
print("distinct tiers active:", len(c))
PY
rm -rf "$d"
