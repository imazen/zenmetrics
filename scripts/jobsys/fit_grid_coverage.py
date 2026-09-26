#!/usr/bin/env python3
"""Grid-coverage gate for a fit grid that spans several jobsets.

Proves that the union of DONE cells across the jobsets equals the registered grid
(960 P0 / 960 P2 / 210 D2), and lists every cell that is neither DONE nor sitting in a jobset
that some worker still serves, with its claim owner and age. This is the precondition for
`FLEET_FITS_DONE.md`: a cell claimed in a drained jobset, or in one whose workers have all moved
to a newer jobset, is never re-claimed and would otherwise vanish without any error.

    fit_grid_coverage.py --grid-manifest fit-manifest-full.json \\
        --jobset fitp0-20260925 --jobset fitp0v8-20260925 \\
        [--serving fitp0v8-20260925 ...] [--endpoint URL] [--bucket zentrain] [--prefix jobs]

Exit 0 iff every registered cell is DONE in some jobset and no DONE cell lies outside the grid.
Exit 1 otherwise; the report says which cells are STRANDED (not DONE, and every jobset that could
run them is drained/paused/unserved), which are IN_FLIGHT (claimed by a live worker), which are
FAILED (a failed ledger row, e.g. `disk_full` / `upload_fail` / `timeout`, and no served jobset
declares them without a failed row: a failed cell's claim stays "done" until the TTL, so nothing
re-runs it until it is re-declared) and which are merely pending. `--serving` names the jobsets
that have live workers; by default every jobset whose control.json is neither drained nor paused
counts as served. `--live-workers FILE` (JSON list or one name per line) names the workers that are
running now: a live worker holds exactly one cell, its newest non-DONE claim, so that cell is
IN_FLIGHT wherever it is declared (an older claim under the same worker name belongs to a dead
earlier incarnation of that container name), and a cell whose only claim is not live is STRANDED
unless a served jobset can re-claim it. Without `--live-workers` every claim is taken as live. Cell -> claim mapping: single-cell chunk claims are `claims/chunk-<sha256(job_id
+ "\\n")>`; job ids are recomputed from the manifest (sha256 of the canonical
`{"inputs":[sorted shas],"kind":...}` JSON, exactly zenfleet-core's `JobId::of`).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pyarrow.parquet as pq


def job_id(job: dict) -> str:
    canon = {"inputs": sorted(set(job["inputs"])), "kind": job["kind"]}
    return hashlib.sha256(json.dumps(canon, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def chunk_key(jid: str) -> str:
    return "chunk-" + hashlib.sha256((jid + "\n").encode()).hexdigest()


def parse_claim(body: str, now: int) -> tuple:
    """`<epoch> <worker> [done/total]` -> (worker, age_seconds, finished); ("?", None, False) if
    unreadable. `finished` is a chunk whose progress reads done >= total: the worker has written its
    result (DONE or a failed row), so the claim no longer means a fit is running."""
    tok = body.split()
    if len(tok) >= 2 and tok[0].isdigit():
        finished = False
        if len(tok) >= 3 and "/" in tok[2]:
            d, _, t = tok[2].partition("/")
            finished = d.isdigit() and t.isdigit() and int(t) > 0 and int(d) >= int(t)
        return tok[1], max(0, now - int(tok[0])), finished
    return "?", None, False


def coverage(grid: list, jobsets: list, serving: set, now: int, live: set | None = None) -> dict:
    """Pure core. `grid`: registered cell names. `jobsets`: dicts with `name`, `declared` (set of
    cell names), `done` (set), `claims` ({cell: (worker, age)}), `drained` (bool), optional `failed`
    ({cell: error_class} for cells whose latest row is not DONE). `live`: names of the
    workers running now, or None to treat every claim as live."""
    grid_set = set(grid)
    done = {}
    for js in jobsets:
        for cell in js["done"]:
            done.setdefault(cell, []).append(js["name"])
    outside = sorted(set(done) - grid_set)
    # A live worker holds one cell: its newest claim on a not-yet-DONE cell of the grid.
    newest = {}
    if live is not None:
        for js in jobsets:
            for cell, (worker, age, *rest) in js["claims"].items():
                if rest and rest[0]:
                    continue          # finished chunk (a failed cell's claim), not a running fit
                if cell in grid_set and cell not in done and worker in live and age is not None:
                    if worker not in newest or age < newest[worker][0]:
                        newest[worker] = (age, cell, js["name"])
    missing = []
    for cell in grid:
        if cell in done:
            continue
        holders = [js for js in jobsets if cell in js["declared"]]
        live_serving = [js["name"] for js in holders if js["name"] in serving]
        claims = {js["name"]: js["claims"][cell] for js in holders if cell in js["claims"]}
        # Claims in jobsets that do not declare the cell cannot exist; claims are keyed per jobset.
        in_flight = [name for name, (w, age, *rest) in claims.items() if not (rest and rest[0])
                     and (live is None or (w in live and newest.get(w, (0, None, None))[1:] == (cell, name)))]
        failed_in = {js["name"]: js.get("failed", {})[cell] for js in holders if cell in js.get("failed", {})}
        served_holders = [js for js in holders if js["name"] in serving]
        if in_flight:
            status = "IN_FLIGHT"
        elif not served_holders:
            status = "FAILED" if failed_in else "STRANDED"
        elif all(cell in js.get("failed", {}) for js in served_holders):
            status = "FAILED"
        else:
            status = "pending"
        missing.append({
            "cell": cell,
            "declared_in": [js["name"] for js in holders],
            "served_by_jobsets": live_serving,
            "failed_in": failed_in,
            "claims": {name: {"worker": w, "age_s": age, "finished": bool(rest and rest[0]),
                              "live": name in in_flight}
                       for name, (w, age, *rest) in claims.items()},
            "status": status,
        })
    return {"grid": len(grid_set), "done": len(grid_set & set(done)),
            "done_in_several_jobsets": {c: j for c, j in done.items() if len(j) > 1 and c in grid_set},
            "done_outside_grid": outside, "missing": missing,
            "in_flight": [m["cell"] for m in missing if m["status"] == "IN_FLIGHT"],
            "failed": [m["cell"] for m in missing if m["status"] == "FAILED"],
            "stranded": [m["cell"] for m in missing if m["status"] == "STRANDED"],
            "complete": not missing and not outside}


def s5(endpoint: str, *args: str, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["s5cmd", "--endpoint-url", endpoint, *args], capture_output=True, text=True, check=check)


def load_jobset(name: str, endpoint: str, bucket: str, prefix: str, scratch: Path, now: int) -> dict:
    base = f"s3://{bucket}/{prefix}/{name}"
    manifest = json.loads(s5(endpoint, "cat", f"{base}/manifest.json").stdout)
    control = s5(endpoint, "cat", f"{base}/control.json", check=False)
    ctl = json.loads(control.stdout) if control.returncode == 0 and control.stdout.strip() else {}
    ledger_dir = scratch / name
    ledger_dir.mkdir(parents=True, exist_ok=True)
    s5(endpoint, "sync", f"{base}/ledger/*", str(ledger_dir) + "/", check=False)
    done, latest_bad = set(), {}
    for path in sorted(ledger_dir.glob("*.parquet")):
        for row in pq.read_table(path, columns=["status", "image_path", "error_class", "ts"]).to_pylist():
            if row["status"] == "done":
                done.add(row["image_path"])
            elif row["image_path"] not in latest_bad or row["ts"] >= latest_bad[row["image_path"]][0]:
                latest_bad[row["image_path"]] = (row["ts"], row["error_class"] or row["status"])
    failed = {cell: err for cell, (_, err) in latest_bad.items() if cell not in done}
    declared = {job["cell"]["image_path"] for job in manifest}
    claims = {}
    listing = s5(endpoint, "ls", f"{base}/claims/*", check=False).stdout
    held = {line.split()[-1].split("/")[-1] for line in listing.splitlines() if line.strip()}
    for job in manifest:
        cell = job["cell"]["image_path"]
        if cell in done:
            continue
        key = chunk_key(job_id(job))
        if key in held:
            body = s5(endpoint, "cat", f"{base}/claims/{key}", check=False).stdout
            claims[cell] = parse_claim(body, now)
    return {"name": name, "declared": declared, "done": done, "claims": claims, "failed": failed,
            "drained": bool(ctl.get("drain")) or bool(ctl.get("paused"))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-manifest", type=Path, required=True, help="manifest whose cell names ARE the registered grid")
    p.add_argument("--jobset", action="append", required=True)
    p.add_argument("--serving", action="append", help="jobsets with live workers (default: all not drained/paused)")
    p.add_argument("--live-workers", type=Path, help="running workers: JSON list or one name per line")
    p.add_argument("--endpoint", default=os.environ.get("EP"))
    p.add_argument("--bucket", default="zentrain")
    p.add_argument("--prefix", default="jobs")
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()
    if not args.endpoint:
        sys.exit("fit_grid_coverage: set --endpoint or EP")
    now = int(time.time())
    grid = [job["cell"]["image_path"] for job in json.loads(args.grid_manifest.read_text())]
    if len(set(grid)) != len(grid):
        sys.exit("fit_grid_coverage: grid manifest has duplicate cell names")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or str(Path.home() / "tmp")) as tmp:
        jobsets = [load_jobset(n, args.endpoint, args.bucket, args.prefix, Path(tmp), now) for n in args.jobset]
    serving = set(args.serving) if args.serving else {js["name"] for js in jobsets if not js["drained"]}
    live = None
    if args.live_workers:
        text = args.live_workers.read_text()
        live = set(json.loads(text)) if text.lstrip().startswith("[") else set(text.split())
    report = coverage(grid, jobsets, serving, now, live)
    print(f"grid {report['grid']}: DONE {report['done']}, missing {len(report['missing'])} "
          f"({len(report['in_flight'])} IN_FLIGHT, {len(report['failed'])} FAILED, {len(report['stranded'])} STRANDED), done outside grid {len(report['done_outside_grid'])}, "
          f"done in several jobsets {len(report['done_in_several_jobsets'])}")
    for m in report["missing"]:
        claims = "; ".join(f"{js}: {c['worker']} age {c['age_s']}s{' (finished chunk)' if c['finished'] else '' if c['live'] else ' (not live)'}"
                           for js, c in m["claims"].items()) or "unclaimed"
        failed = "  failed: " + ", ".join(f"{js}={e}" for js, e in m["failed_in"].items()) if m["failed_in"] else ""
        print(f"  {m['status']:8s} {m['cell']}  declared in {m['declared_in'] or 'NO jobset'}  {claims}{failed}")
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True))
    sys.exit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
