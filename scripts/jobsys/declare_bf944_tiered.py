#!/usr/bin/env python3
# declare_bf944_tiered.py — SIMD-TIER-MATCHED declare of the 944 backfill (SOTA-944 P1
# bigcodec leg). Supersedes the flat declare_bf944.py pool after the smoke-stage G-BF1
# finding (2026-08-02):
#
#   The bf924 baseline (tbig_924_full + the 21 views) is SIMD-TIER-MIXED row-wise —
#   each row carries the accumulation order of whichever box extracted it (AVX-512
#   Zen4 lianli/wsl vs AVX2-only tower/node-2/i265/node-3 vs NEON mac; deltas ~1e-10 rel,
#   ~25% of slots). MEASURED: a tower re-extraction of tower-made rows is BITWISE
#   IDENTICAL with the new binary, while an AVX-512 box re-extracting the same rows is
#   not (tier_proof instrument, benchmarks doc). So G-BF1 (f0..f923 bitwise vs the 924
#   parquet, row-for-row) is achievable ONLY by re-extracting every cell on a box of
#   the SAME SIMD tier as its bf924 extractor.
#
# This declare therefore splits every bf924 run's cells into up to three tier runs,
# keyed by the bf924 ledger's per-cell worker (attribution.parquet, built by
# attribution.py from the kept rows of matched_ledger.parquet):
#
#   bf944v4x-<x>  -> jobs/_pool944v4x/runlist.tsv   (AVX-512: lianli, wsl)
#   bf944v4-<x>   -> jobs/_pool944v4/runlist.tsv    (AVX2: tower, node-2, node-3, i265)
#   bf944neon-<x> -> jobs/_pool944neon/runlist.tsv  (Apple Silicon: mac, native build)
#
# Cell identity: the bf924 manifests are read directly and the metric rewritten
# (zensim-foldapp -> zensim-foldapp2), so the tier split is by MANIFEST INDEX against
# `zenfleet-ctl ids` output — the owner's own JobId hash, never re-implemented here.
# HARD GATES before any upload: per run, the manifest's job_id set must equal the
# attribution's set for that pool (every cell attributed, none extra), and the global
# per-tier totals must match the attribution histogram exactly.
#
# Usage: python3 scripts/jobsys/declare_bf944_tiered.py [--dry-run] \
#            [--attribution ~/tmp/bigcodec944/attribution.parquet]
import argparse
import gzip
import io
import json
import os
import subprocess
import sys
import tempfile

import pyarrow.parquet as pq

BUCKET = "zentrain"
OLD_METRIC = "zensim-foldapp"
METRIC = "zensim-foldapp2"
TIERS = ("v4", "v4x", "neon")
# tag -> (sweep, ntars) — mirrors declare_bf924.py / pool_launch.sh gen_runlist.
TAR_RUNS = [
    ("zavif", "mandfix4-zenavif-1782593621", 8),
    ("zjxll", "jxl-lossy-vardct-1782609551", 24),
    ("zwebp", "mandfix2-zenwebp-1782584881", 9),
    ("zjxlm", "jxl-modular-1782596759", 10),
    ("zpng", "mandfix2-zenpng-1782584881", 2),
]
ENC_RUN = ("zjl2", "canonical/2026-06-27/zenjpeg_lossy/encodes")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CTL = os.environ.get("ZEN_CTL_BIN", os.path.join(ROOT, "target", "release", "zenfleet-ctl"))


def envs():
    e = dict(os.environ)
    for line in open(os.path.expanduser("~/.config/cloudflare/r2-credentials")):
        line = line.strip()
        if line.startswith("R2_") and "=" in line:
            k, v = line.split("=", 1)
            e[k] = v.strip().strip('"').strip("'")
    e["AWS_ACCESS_KEY_ID"] = e["R2_ACCESS_KEY_ID"]
    e["AWS_SECRET_ACCESS_KEY"] = e["R2_SECRET_ACCESS_KEY"]
    e["AWS_REGION"] = "auto"
    e["EP"] = "https://%s.r2.cloudflarestorage.com" % e["R2_ACCOUNT_ID"]
    return e


E = envs()


def s5(*args, stdin=None):
    r = subprocess.run(
        ["s5cmd", "--endpoint-url", E["EP"], *args],
        env=E, capture_output=True, input=stdin,
    )
    r.err = r.stderr.decode(errors="replace")[:200]
    return r


def manifest_ids(cells) -> list[str]:
    """Owner-hashed job ids, one per cell, in manifest order (zenfleet-ctl ids)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cells, f)
        path = f.name
    try:
        r = subprocess.run([CTL, "ids", "--manifest", path], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"zenfleet-ctl ids: {r.stderr.decode()[:300]}")
        out = []
        for ln in r.stdout.decode().splitlines():
            idx, jid, _ip = ln.split("\t", 2)
            assert int(idx) == len(out), "ids output out of order"
            out.append(jid)
        return out
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--attribution",
        default=os.path.expanduser("~/tmp/bigcodec944/attribution.parquet"),
    )
    a = ap.parse_args()

    at = pq.read_table(a.attribution, columns=["pool", "job_id", "tier"])
    tier_of = {}  # (pool, job_id) -> tier
    for p, j, t in zip(
        at.column("pool").to_pylist(), at.column("job_id").to_pylist(),
        at.column("tier").to_pylist(),
    ):
        if t not in TIERS:
            print(f"ABORT: attribution row with unknown tier {t!r}")
            sys.exit(1)
        tier_of[(p, j)] = t
    print(f"attribution: {len(tier_of)} (pool, job_id) rows")

    runs = []  # (old, x, src, mode)
    for tag, sweep, ntars in TAR_RUNS:
        for i in range(ntars):
            runs.append((
                f"bf924-{tag}-t{i}", f"{tag}-t{i}",
                f"s3://{BUCKET}/jxl-lossy/runs/{sweep}/variants/box-{i}.tar", "tar",
            ))
    runs.append((f"bf924-{ENC_RUN[0]}", ENC_RUN[0], ENC_RUN[1], "enc"))

    runlists = {t: [] for t in TIERS}
    totals = {t: 0 for t in TIERS}
    fail = 0
    for old, x, src, mode in runs:
        r = s5("cat", f"s3://{BUCKET}/jobs/{old}/manifest.json.gz")
        if r.returncode != 0:
            print(f"FAIL fetch {old}: {r.err}")
            fail += 1
            continue
        cells = json.load(gzip.open(io.BytesIO(r.stdout)))
        ok = True
        for c in cells:
            k = c.get("kind")
            if not isinstance(k, dict) or k.get("kind") != "score_file" \
                    or k.get("metrics") != [OLD_METRIC]:
                print(f"FAIL {old}: unexpected cell kind/metrics {str(k)[:80]}")
                fail += 1
                ok = False
                break
        if not ok:
            continue
        ids = manifest_ids(cells)
        # HARD GATE: manifest ids must exactly equal the attributed set for this pool.
        attr_ids = {j for (p, j) in tier_of if p == old}
        if set(ids) != attr_ids or len(ids) != len(attr_ids):
            print(f"FAIL {old}: manifest ids != attributed ids "
                  f"({len(ids)} cells vs {len(attr_ids)} attributed; "
                  f"missing={len(attr_ids - set(ids))} extra={len(set(ids) - attr_ids)})")
            fail += 1
            continue
        by_tier = {t: [] for t in TIERS}
        for i, jid in enumerate(ids):
            by_tier[tier_of[(old, jid)]].append(i)
        parts = []
        for t in TIERS:
            idxs = by_tier[t]
            if not idxs:
                continue
            totals[t] += len(idxs)
            new = f"bf944{t}-{x}"
            sub = []
            for i in idxs:
                c = json.loads(json.dumps(cells[i]))  # deep copy
                c["kind"]["metrics"] = [METRIC]
                sub.append(c)
            if not a.dry_run:
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                    gz.write(json.dumps(sub).encode())
                u = s5("pipe", f"s3://{BUCKET}/jobs/{new}/manifest.json.gz",
                       stdin=buf.getvalue())
                if u.returncode != 0:
                    print(f"FAIL upload {new}: {u.err}")
                    fail += 1
                    continue
                if mode == "tar":
                    u = s5("cp", f"s3://{BUCKET}/jobs/{old}/variant_index.tsv",
                           f"s3://{BUCKET}/jobs/{new}/variant_index.tsv")
                    if u.returncode != 0:
                        print(f"FAIL index copy {new}: {u.err}")
                        fail += 1
                        continue
            runlists[t].append(f"{new}\t{src}\t{mode}")
            parts.append(f"{t}:{len(idxs)}")
        print(f"ok {old} -> {' '.join(parts)} (total {len(cells)})", flush=True)

    print(f"tier totals: {totals} (sum {sum(totals.values())})")
    if fail:
        print(f"ABORT: {fail} failures — runlists NOT uploaded")
        sys.exit(1)
    # Global gate: totals must equal the attribution histogram exactly.
    import collections
    want = collections.Counter(at.column("tier").to_pylist())
    if {t: totals[t] for t in TIERS} != {t: want.get(t, 0) for t in TIERS}:
        print(f"ABORT: tier totals {totals} != attribution histogram {dict(want)}")
        sys.exit(1)
    for t in TIERS:
        rl = "\n".join(runlists[t]) + "\n"
        uri = f"s3://{BUCKET}/jobs/_pool944{t}/runlist.tsv"
        if a.dry_run:
            print(f"[dry] {uri}: {len(runlists[t])} runs")
        else:
            u = s5("pipe", uri, stdin=rl.encode())
            if u.returncode != 0:
                print(f"FAIL runlist upload {uri}: {u.err}")
                sys.exit(1)
            print(f"runlist: {len(runlists[t])} runs -> {uri}")
    print("DONE")


if __name__ == "__main__":
    main()
