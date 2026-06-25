#!/usr/bin/env bash
# Watch a running fleet: claims taken + ledger DONE rows grouped by provider (proves which tiers are
# concurrently working the one queue).  Usage: bash scripts/jobsys/watch_fleet.sh <RUN>
set -uo pipefail
RUN="${1:?usage: watch_fleet.sh <RUN>}"
B="${ZEN_FLEET_BUCKET:-zen-tuning-ephemeral}"
set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto
CLAIMS=$(s5cmd --endpoint-url "$EP" ls "s3://$B/$RUN/claims/" 2>/dev/null | grep -vc spec/ || true)
echo "claims taken: $CLAIMS"
d=$(mktemp -d); s5cmd --endpoint-url "$EP" cp "s3://$B/$RUN/ledger/*" "$d/" >/dev/null 2>&1 || true
CLAIMS="$CLAIMS" python3 - "$d" <<'PY'
import glob, sys, os, collections, pyarrow.parquet as pq
c=collections.Counter()
for f in glob.glob(sys.argv[1]+"/*.parquet"):
    try:
        t=pq.read_table(f)
        if "provider" in t.schema.names:
            for p in t.column("provider").to_pylist(): c[p]+=1
    except Exception: pass
done=sum(c.values()); claims=int(os.environ.get("CLAIMS","0") or 0)
print("ledger DONE rows by provider:", dict(c) or "(none yet)")
print("distinct tiers active:", len(c))
# Idle/stall flag: boxes claimed work but the ledger isn't filling = stalled, or all mid-long-job.
if claims > 0 and done == 0:
    print(f"⚠  {claims} claim(s) taken but 0 DONE — workers stalled or all mid-(long)-job; boxes may be burning $/hr idle.")
    print("   per-box util:  scripts/sweep/fleet_util_snapshot.sh")
elif claims > 0 and done < claims // 2:
    print(f"⚠  {claims} claims but only {done} done — low throughput; check scripts/sweep/fleet_util_snapshot.sh")
PY
rm -rf "$d"
