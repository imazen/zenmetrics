#!/usr/bin/env bash
# Prove the LAN-staged 372 corpora (scripts/jobsys/stage_eval372_corpora_lan.py)
# are usable by the REAL executor with NO code change: declare + run one
# Feature job against s3:// paths, declare + run the SAME pairs against their
# local-path originals, and assert the feature vectors are bit-identical.
#
# This is the reachability half of docs/PLAN_REV2_RECALC_2026-09-06.md's LAN
# staging prerequisite. It is an operator-run gate (needs real LAN-store
# network access + credentials), not a `cargo test` — matching
# rev2_bitexact_gate.py's own shape, for the same reason: CI has neither.
#
# Usage: scripts/jobsys/verify_lan_stage_reachability.sh [corpus]
#   corpus defaults to csiq (smallest staged corpus, 3 pairs from one ref).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORPUS="${1:-csiq}"
STAGE_ROOT="/mnt/v/zen/zensim-training/rev2-lan-stage-2026-09-06"
WORK="$(mktemp -d "${HOME}/tmp/lanverify-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

EXEC="$REPO/target/release/zenmetrics"
JOBCTL="$REPO/target/release/zenfleet-ctl"
for b in "$EXEC" "$JOBCTL"; do
  [ -x "$b" ] || { echo "FATAL: missing binary $b (cargo build --release -p zenmetrics-cli --bin zenmetrics -p zenfleet-ctl --features ...,bmp,cpu-metrics,feature-jobs first)" >&2; exit 2; }
done

# shellcheck source=../lib/s3env.sh
set -a; . "$REPO/scripts/lib/s3env.sh"; set +a
export ZEN_R2_ENDPOINT="$EP"

lan_tsv="$STAGE_ROOT/pairs/${CORPUS}_pairs_lan.tsv"
[ -f "$lan_tsv" ] || { echo "FATAL: no staged pairs TSV for '$CORPUS' at $lan_tsv" >&2; exit 2; }
manifest_json="$STAGE_ROOT/manifests/${CORPUS}_MANIFEST.json"
[ -f "$manifest_json" ] || { echo "FATAL: no manifest for '$CORPUS' at $manifest_json" >&2; exit 2; }

# Take the first 3 rows (one reference, its first few distorted variants) —
# enough to exercise both the reference-decode path and the inputs loop.
head -4 "$lan_tsv" > "$WORK/s3_pairs.tsv"
python3 - "$WORK/s3_pairs.tsv" "$manifest_json" <<'PY'
import csv, json, os, sys
s3_pairs_path, manifest_path = sys.argv[1], sys.argv[2]
m = json.load(open(manifest_path))
root, prefix = m["local_root"], m["s3_prefix"]
rows = list(csv.DictReader(open(s3_pairs_path), delimiter="\t"))
local_rows = []
for r in rows:
    lr = dict(r)
    for col in ("ref_path", "dist_path"):
        assert r[col].startswith(prefix + "/"), r[col]
        rel = r[col][len(prefix) + 1:]
        lr[col] = f"{root}/{rel}"
    local_rows.append(lr)
with open(sys.argv[1].replace("s3_pairs.tsv", "local_pairs.tsv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
    w.writeheader()
    w.writerows(local_rows)
PY

"$JOBCTL" declare-features --pairs "$WORK/s3_pairs.tsv" --out "$WORK/s3_manifest.json" --regime 372 --chunk 16 >&2
"$JOBCTL" declare-features --pairs "$WORK/local_pairs.tsv" --out "$WORK/local_manifest.json" --regime 372 --chunk 16 >&2

python3 -c "import json; print(json.dumps(json.load(open('$WORK/s3_manifest.json'))[0]))" \
  | "$EXEC" jobexec > "$WORK/s3_out.jsonl" 2>"$WORK/s3_err.log" \
  || { echo "FATAL: S3-path job failed:"; cat "$WORK/s3_err.log" >&2; exit 1; }
python3 -c "import json; print(json.dumps(json.load(open('$WORK/local_manifest.json'))[0]))" \
  | "$EXEC" jobexec > "$WORK/local_out.jsonl" 2>"$WORK/local_err.log" \
  || { echo "FATAL: local-path job failed:"; cat "$WORK/local_err.log" >&2; exit 1; }

python3 - "$WORK/s3_out.jsonl" "$WORK/local_out.jsonl" <<'PY'
import json, os, struct, sys

def load(path):
    rows = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[os.path.basename(r["encode_sha"])] = r["features"]
    return rows

def bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]

a, b = load(sys.argv[1]), load(sys.argv[2])
if set(a) != set(b):
    print(f"FATAL: row sets differ: s3-only={set(a)-set(b)} local-only={set(b)-set(a)}", file=sys.stderr)
    sys.exit(1)
ncmp = ndiff = 0
for k in a:
    for x, y in zip(a[k], b[k]):
        ncmp += 1
        if bits(x) != bits(y):
            ndiff += 1
print(f"compared {ncmp} cells across {len(a)} pairs: {ndiff} differ")
if ndiff:
    sys.exit(1)
print("RESULT: LAN-STAGE REACHABLE — s3:// fetch produces bit-identical features to the local path, no executor code change needed")
PY
