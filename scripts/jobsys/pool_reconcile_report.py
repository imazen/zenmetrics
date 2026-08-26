#!/usr/bin/env python3
"""Per-run reconcile accounting for a pool wave: distinct done/failed job_ids
from the LIVE ledgers vs the manifest's declared cells. The completion answer
the drain beacons only approximate (G-W3-style gate: gap==0 on every run).

  python3 pool_reconcile_report.py [runlist_uri]   # default jobs/_pool944v4/runlist.tsv
"""
import gzip
import io
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs
import pyarrow.fs as fs

# THE resolver (scripts/lib/zen_s3env.py) — never re-derive endpoint/creds in a new script.
# FIXED 2026-08-24: this used to hardcode a read of ~/.config/cloudflare/r2-credentials
# regardless of ZEN_STORE — see pool_progress.py's matching fix for the full story.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lib"))
from zen_s3env import resolve

_endpoint, _access_key, _secret_key = resolve()
S3 = fs.S3FileSystem(
    access_key=_access_key, secret_key=_secret_key,
    endpoint_override=_endpoint, region="auto")
BUCKET = os.environ.get("ZEN_BUCKET", "zentrain")
# --auto-pause: after reconciling, set control.json {"paused":true} on every run
# whose gap==0. A completed job whose workers are still attached re-scores it
# every pass (coarse claims never let it idle), which is how one done job burned
# 37.8x rescore-tax on the LAN while the real gap sat unworked (2026-08-26). This
# makes the reconcile loop self-heal that: a completed run is paused so no worker
# wastes another pass on it. Idempotent (skips runs already paused).
AUTO_PAUSE = "--auto-pause" in sys.argv
_pos = [a for a in sys.argv[1:] if not a.startswith("--")]
RUNLIST = _pos[0] if _pos else "jobs/_pool944v4/runlist.tsv"

with S3.open_input_stream(f"{BUCKET}/{RUNLIST}") as f:
    runs = [ln.split("\t")[0] for ln in f.read().decode().splitlines() if ln.strip()]


def one(run: str):
    try:
        with S3.open_input_stream(f"{BUCKET}/jobs/{run}/manifest.json.gz") as f:
            raw = f.read()
        try:
            declared = len(json.load(gzip.open(io.BytesIO(raw))))
        except gzip.BadGzipFile:
            declared = len(json.loads(raw))  # transport already inflated it
        # Read per-file with tolerance: a worker mid-pass writes chunks that are
        # briefly footerless on SeaweedFS; one in-flight file must not kill the
        # whole report (2026-08-26). Skipped files are counted loudly.
        _sel = fs.FileSelector(f"{BUCKET}/jobs/{run}/ledger/")
        _frags, _skipped = [], 0
        for _fi in S3.get_file_info(_sel):
            if _fi.size == 0: _skipped += 1; continue
            try:
                _frags.append(pq.read_table(_fi.path, filesystem=S3, columns=["job_id", "status"]))
            except Exception:
                _skipped += 1
        if _skipped: print(f"  [{run}] WARNING: skipped {_skipped} unreadable/in-flight ledger chunk(s)")
        t = pa.concat_tables(_frags, promote_options="permissive") if _frags else None
        st = t.column("status").to_pylist()
        jid = t.column("job_id").to_pylist()
        done = {j for j, s in zip(jid, st) if s == "done"}
        failed = {j for j, s in zip(jid, st) if s == "failed"} - done
        gap = declared - len(done)
        return (run, declared, len(done), len(failed), gap, t.num_rows)
    except Exception as e:  # noqa: BLE001 — report and keep accounting
        return (run, -1, -1, -1, -1, str(e)[:80])


t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    rows = list(ex.map(one, runs))
tot_decl = tot_done = tot_fail = tot_gap = tot_rows = 0
bad = []
complete = []  # runs with gap==0 (declared > 0), for --auto-pause
for run, decl, done, fail, gap, raw in sorted(rows):
    if decl < 0:
        print(f"{run}: ERROR {raw}")
        bad.append(run)
        continue
    tot_decl += decl
    tot_done += done
    tot_fail += fail
    tot_gap += max(gap, 0)
    tot_rows += raw
    flag = "" if gap == 0 else f"  <-- GAP {gap}"
    print(f"{run}: declared={decl} done={done} failed-only={fail} raw_rows={raw}{flag}")
    if gap <= 0:  # all declared cells done (gap can go negative on re-declared runs)
        complete.append((run, raw, done))
print(f"\nTOTAL declared={tot_decl} distinct_done={tot_done} failed-only={tot_fail} "
      f"gap={tot_gap} raw_ledger_rows={tot_rows} (rescore tax {tot_rows / max(tot_done, 1):.2f}x) "
      f"errors={len(bad)} ({time.time() - t0:.0f}s)")
print("VERDICT:", "COMPLETE — every run gap==0" if tot_gap == 0 and not bad else "NOT COMPLETE")

if AUTO_PAUSE and complete:
    import json as _json
    print(f"\n--auto-pause: {len(complete)} run(s) at gap<=0 — pausing so no worker re-scores them")
    for run, raw, done in complete:
        ckey = f"{BUCKET}/jobs/{run}/control.json"
        try:
            with S3.open_input_stream(ckey) as f:
                cur = _json.loads(f.read().decode() or "{}")
        except Exception:  # noqa: BLE001 — missing/unreadable control.json => treat as unpaused
            cur = {}
        if cur.get("paused") is True:
            print(f"  {run}: already paused (rescore tax {raw / max(done, 1):.1f}x)")
            continue
        cur["paused"] = True
        try:
            with S3.open_output_stream(ckey) as f:
                f.write(_json.dumps(cur).encode())
            print(f"  AUTO-PAUSED {run}: control.json paused:true "
                  f"(was re-scoring {done} done cells at {raw / max(done, 1):.1f}x tax)")
        except Exception as e:  # noqa: BLE001 — report, don't abort the other pauses
            print(f"  {run}: FAILED to pause ({str(e)[:80]})")
elif AUTO_PAUSE:
    print("\n--auto-pause: no run at gap<=0; nothing to pause")
