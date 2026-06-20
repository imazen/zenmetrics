#!/usr/bin/env python3
"""fleet_top — real-time health monitor for a zen job-system fleet run.

A `top`/`watch` for the fleet: reads the R2 ledger + claims + manifest for a
RUN and renders, on a refresh loop, everything you need to stay on top of a
live fleet:

  • PROGRESS  — done / poison / in-flight / gap vs the manifest, rate, ETA.
  • WORKERS   — per-worker throughput + liveness (ACTIVE / IDLE / STALLED by
                last-seen ledger ts), claims held, provider.
  • STALLS    — claims whose lease has expired (no heartbeat-renew within TTL)
                but whose job is NOT terminal → stuck/abandoned, reclaimable;
                and workers holding claims with no recent ledger activity.
  • PROBLEM CELLS (deactivate) — POISON rows (the reconciler gave up on these:
                deterministic failure, or transient over the attempt cap) and
                Failed rows at the edge of the cap (about to poison), grouped
                by error_class so a systemic-bad config is obvious. `--poison-out`
                exports the list so you can exclude it from the next declare.
  • RECENT FAILURES — a live tail of the newest Failed/Poison rows.

Data model (matches `zenfleet-ledger::ledger_schema` + `zenfleet-core`):
  ledger row = job_id, image_path, codec, q, knob_tuple_json, output_sha,
  status {Done,Failed,Poison}, error_class, attempts, ts (unix secs), worker,
  provider, kind_json. Latest-wins per job_id on ts. Done|Poison are terminal;
  Failed is retried until `max_attempts` (default 3) then poisoned.

Sources:
  default        s3://$ZEN_FLEET_BUCKET/<RUN>/{ledger/*.parquet,claims/,manifest.json}
                 (R2 creds from ~/.config/cloudflare/r2-credentials, via s5cmd)
  --local DIR    read DIR/{ledger/*.parquet,claims/,manifest.json} (offline/test)

Usage:
  fleet_top.py <RUN> [--interval 5] [--once] [--stall 180] [--ttl 600]
               [--max-attempts 3] [--poison-out poison.tsv] [--local DIR]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

# ── ANSI ────────────────────────────────────────────────────────────────────
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GRN, YEL, CYN, MAG = ("\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[35m")


def c(s, color, on=True):
    return f"{color}{s}{RESET}" if on else str(s)


def hms(secs):
    secs = int(max(0, secs))
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def bar(frac, width=28):
    frac = max(0.0, min(1.0, frac))
    n = int(frac * width)
    return "[" + "#" * n + "-" * (width - n) + f"] {frac*100:5.1f}%"


# ── data loading ──────────────────────────────────────────────────────────--
def r2_env():
    cred = os.path.expanduser("~/.config/cloudflare/r2-credentials")
    env = dict(os.environ)
    if os.path.exists(cred):
        for line in open(cred):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"')
    env.setdefault("AWS_ACCESS_KEY_ID", env.get("R2_ACCESS_KEY_ID", ""))
    env.setdefault("AWS_SECRET_ACCESS_KEY", env.get("R2_SECRET_ACCESS_KEY", ""))
    env["AWS_REGION"] = "auto"
    return env


def s5(env, *args):
    ep = f"https://{env.get('R2_ACCOUNT_ID','')}.r2.cloudflarestorage.com"
    return subprocess.run(["s5cmd", "--endpoint-url", ep, *args],
                          env=env, capture_output=True, text=True)


def fetch_r2(bucket, run, tmp):
    """Pull ledger + manifest; list claims (key + mtime). Returns (ledger_dir, manifest_path, claims)."""
    env = r2_env()
    base = f"s3://{bucket}/{run}"
    ld = os.path.join(tmp, "ledger"); os.makedirs(ld, exist_ok=True)
    s5(env, "cp", f"{base}/ledger/*", ld + "/")            # tolerate "no object found"
    # cp to the dir (robust s5cmd idiom for a single object) → tmp/manifest.json
    s5(env, "cp", f"{base}/manifest.json", tmp + "/")
    mf = os.path.join(tmp, "manifest.json")
    claims = []
    out = s5(env, "ls", f"{base}/claims/").stdout
    for line in out.splitlines():
        # s5cmd ls: "2026/06/18 12:00:00  1234  key"
        parts = line.split()
        if len(parts) >= 4 and "/" in parts[0]:
            ts = parse_s5_time(parts[0], parts[1])
            key = parts[-1].rstrip("/")
            if key and not key.endswith("/"):
                claims.append((os.path.basename(key), ts))
    return ld, (mf if os.path.exists(mf) else None), claims


def parse_s5_time(d, t):
    try:
        return int(time.mktime(time.strptime(f"{d} {t}", "%Y/%m/%d %H:%M:%S")))
    except Exception:
        return 0


def load_local(d):
    ld = os.path.join(d, "ledger")
    mf = os.path.join(d, "manifest.json")
    claims = []
    cdir = os.path.join(d, "claims")
    if os.path.isdir(cdir):
        for name in os.listdir(cdir):
            p = os.path.join(cdir, name)
            claims.append((name, int(os.path.getmtime(p))))
    return ld, (mf if os.path.exists(mf) else None), claims


def read_ledger(ledger_dir):
    """All rows, latest-wins per job_id on ts. Returns dict job_id -> row(dict)."""
    import pyarrow.parquet as pq
    latest = {}
    for f in sorted(glob.glob(os.path.join(ledger_dir, "*.parquet"))):
        try:
            t = pq.read_table(f)
        except Exception:
            continue
        cols = {n: t.column(n).to_pylist() for n in t.schema.names}
        for i in range(t.num_rows):
            row = {n: cols[n][i] for n in cols}
            jid = row.get("job_id")
            ts = row.get("ts") or 0
            if jid is None:
                continue
            if jid not in latest or ts >= latest[jid].get("ts", 0):
                latest[jid] = row
    return latest


def read_manifest(path):
    if not path or not os.path.exists(path):
        return None
    try:
        d = json.load(open(path))
    except Exception:
        return None
    jobs = d if isinstance(d, list) else d.get("jobs", d.get("desired", []))
    return len(jobs) if isinstance(jobs, list) else None


# ── analysis ─────────────────────────────────────────────────────────────---
def analyze(latest, claims, total, now, args, run_start):
    rows = list(latest.values())
    by_status = {"Done": 0, "Failed": 0, "Poison": 0}
    workers = {}        # name -> stats
    err_classes = {}
    poison_cells, atrisk_cells, recent = [], [], []
    for r in rows:
        st = r.get("status", "?")
        by_status[st] = by_status.get(st, 0) + 1
        w = r.get("worker", "?")
        ws = workers.setdefault(w, {"worker": w, "provider": r.get("provider", "?"),
                                    "done": 0, "failed": 0, "poison": 0,
                                    "last_ts": 0, "claims": 0})
        if st == "Done": ws["done"] += 1
        elif st == "Failed": ws["failed"] += 1
        elif st == "Poison": ws["poison"] += 1
        ws["last_ts"] = max(ws["last_ts"], r.get("ts", 0))
        ws["provider"] = r.get("provider", ws["provider"])
        if st in ("Failed", "Poison"):
            ec = r.get("error_class") or "(none)"
            err_classes[ec] = err_classes.get(ec, 0) + 1
            recent.append(r)
        cell = {"job_id": r.get("job_id"), "codec": r.get("codec"), "q": r.get("q"),
                "knob_tuple_json": r.get("knob_tuple_json"),
                "error_class": r.get("error_class") or "(none)",
                "attempts": r.get("attempts", 0)}
        if st == "Poison":
            poison_cells.append(cell)
        elif st == "Failed" and r.get("attempts", 0) >= max(1, args.max_attempts - 1):
            atrisk_cells.append(cell)

    terminal = by_status.get("Done", 0) + by_status.get("Poison", 0)
    # claims held that are NOT terminal in the ledger = in-flight; expired = stalled
    claim_jobs = {jid: ts for jid, ts in claims}
    inflight, stalled_jobs = 0, []
    for jid, cts in claim_jobs.items():
        r = latest.get(jid)
        if r and r.get("status") in ("Done", "Poison"):
            continue  # finished, claim is just a leftover lease marker
        inflight += 1
        if cts and (now - cts) > args.ttl:
            stalled_jobs.append((jid, now - cts))
        # attribute a held claim to its worker if we can see one
        if r and r.get("worker") in workers:
            workers[r["worker"]]["claims"] += 1

    gap = (total - terminal - inflight) if total else None

    # worker liveness
    for ws in workers.values():
        age = now - ws["last_ts"] if ws["last_ts"] else 1 << 30
        ws["age"] = age
        if age <= args.stall:
            ws["state"] = "ACTIVE"
        elif ws["claims"] > 0:
            ws["state"] = "STALLED"   # holding work but gone quiet
        else:
            ws["state"] = "IDLE"

    # throughput / ETA from terminal rate over the run window
    elapsed = max(1, now - run_start) if run_start else None
    rate_min = (terminal / elapsed * 60.0) if elapsed else None
    eta = (gap / rate_min * 60.0) if (gap and rate_min and rate_min > 0) else None

    recent.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return {
        "total": total, "done": by_status.get("Done", 0),
        "failed": by_status.get("Failed", 0), "poison": by_status.get("Poison", 0),
        "inflight": inflight, "gap": gap, "terminal": terminal,
        "workers": sorted(workers.values(), key=lambda w: (-w["done"], w["worker"])),
        "stalled_jobs": sorted(stalled_jobs, key=lambda x: -x[1]),
        "poison_cells": poison_cells, "atrisk_cells": atrisk_cells,
        "err_classes": sorted(err_classes.items(), key=lambda x: -x[1]),
        "recent": recent[:8], "rate_min": rate_min, "eta": eta, "elapsed": elapsed,
    }


# ── render ───────────────────────────────────────────────────────────────---
def render(run, st, args, now):
    color = sys.stdout.isatty()
    L = []
    L.append(c(f"fleet_top  run={run}  {time.strftime('%H:%M:%S')}"
               f"  elapsed={hms(st['elapsed']) if st['elapsed'] else '?'}", BOLD, color))
    tot = st["total"]
    if tot:
        frac = st["terminal"] / tot
        L.append("  " + bar(frac) + f"  {st['terminal']}/{tot} terminal")
    rate = f"{st['rate_min']:.1f}/min" if st["rate_min"] else "?"
    eta = hms(st["eta"]) if st["eta"] else "?"
    L.append("  " + "  ".join([
        c(f"done {st['done']}", GRN, color),
        c(f"inflight {st['inflight']}", CYN, color),
        c(f"failed {st['failed']}", YEL, color),
        c(f"POISON {st['poison']}", RED, color),
        f"gap {st['gap'] if st['gap'] is not None else '?'}",
        f"rate {rate}", f"eta {eta}",
    ]))

    L.append(c("\nWORKERS", BOLD, color))
    if st["workers"]:
        L.append(f"  {'worker':<22}{'provider':<10}{'state':<9}{'done':>6}{'fail':>5}"
                 f"{'pois':>5}{'claims':>7}  last-seen")
        for w in st["workers"][:14]:
            col = {"ACTIVE": GRN, "STALLED": RED, "IDLE": DIM}.get(w["state"], "")
            seen = hms(w["age"]) + " ago" if w["age"] < (1 << 29) else "never"
            state = c(f"{w['state']:<9}", col, color)
            L.append(f"  {w['worker'][:21]:<22}{w['provider'][:9]:<10}{state}"
                     f"{w['done']:>6}{w['failed']:>5}{w['poison']:>5}{w['claims']:>7}  {seen}")
    else:
        L.append("  (no worker rows yet)")

    if st["stalled_jobs"]:
        L.append(c(f"\nSTALLS  ({len(st['stalled_jobs'])} claims past TTL {args.ttl}s, not terminal — reclaimable)", BOLD + RED if color else "", color))
        for jid, age in st["stalled_jobs"][:6]:
            L.append(f"  {jid[:16]}…  stuck {hms(age)}")

    npois, nrisk = len(st["poison_cells"]), len(st["atrisk_cells"])
    if npois or nrisk or st["err_classes"]:
        L.append(c(f"\nPROBLEM CELLS  poison(deactivated) {npois}  at-risk(near cap) {nrisk}", BOLD, color))
        for ec, n in st["err_classes"][:6]:
            L.append(f"  {c('×'+str(n), RED, color)}  {ec}")
        if npois:
            L.append(c(f"  → {npois} cells already POISON (won't retry). --poison-out to export the deactivate list.", DIM, color))

    if st["recent"]:
        L.append(c("\nRECENT FAILURES", BOLD, color))
        for r in st["recent"]:
            ago = hms(now - r.get("ts", now)) + " ago"
            tag = c("POISON", RED, color) if r.get("status") == "Poison" else c("fail", YEL, color)
            L.append(f"  {ago:>10}  {tag}  {r.get('codec','?')}/q{r.get('q','?')}  "
                     f"{(r.get('error_class') or '(none)')}  a{r.get('attempts',0)}  [{r.get('worker','?')[:14]}]")
    return "\n".join(L)


def export_poison(path, st):
    with open(path, "w") as f:
        f.write("job_id\tcodec\tq\tknob_tuple_json\terror_class\tattempts\tkind\n")
        for cell in st["poison_cells"]:
            f.write(f"{cell['job_id']}\t{cell['codec']}\t{cell['q']}\t"
                    f"{cell['knob_tuple_json']}\t{cell['error_class']}\t{cell['attempts']}\tpoison\n")
        for cell in st["atrisk_cells"]:
            f.write(f"{cell['job_id']}\t{cell['codec']}\t{cell['q']}\t"
                    f"{cell['knob_tuple_json']}\t{cell['error_class']}\t{cell['attempts']}\tat-risk\n")


def snapshot(run, args, run_start_holder):
    now = int(time.time())
    if args.local:
        ledger_dir, mf, claims = load_local(args.local)
        latest = read_ledger(ledger_dir)
        total = read_manifest(mf)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir, mf, claims = fetch_r2(args.bucket, run, tmp)
            latest = read_ledger(ledger_dir)
            total = read_manifest(mf)   # must read before the tmp dir is cleaned up
    if run_start_holder[0] is None:
        seen = [r.get("ts", 0) for r in latest.values() if r.get("ts")]
        run_start_holder[0] = min(seen) if seen else now
    st = analyze(latest, claims, total, now, args, run_start_holder[0])
    if args.poison_out:
        export_poison(args.poison_out, st)
    return st, now


def main():
    ap = argparse.ArgumentParser(description="real-time zen fleet health monitor")
    ap.add_argument("run", help="RUN id (R2 prefix under the bucket)")
    ap.add_argument("--bucket", default=os.environ.get("ZEN_FLEET_BUCKET", "zen-tuning-ephemeral"))
    ap.add_argument("--interval", type=float, default=5.0, help="refresh seconds")
    ap.add_argument("--once", action="store_true", help="single snapshot, then exit")
    ap.add_argument("--stall", type=int, default=180, help="worker quiet secs → STALLED")
    ap.add_argument("--ttl", type=int, default=600, help="claim age secs past which a non-terminal claim is stalled")
    ap.add_argument("--max-attempts", type=int, default=3, help="retry cap (matches reconciler); attempts≥cap-1 = at-risk")
    ap.add_argument("--poison-out", help="write the poison + at-risk deactivate list to this TSV each refresh")
    ap.add_argument("--local", help="read from a local DIR/{ledger,claims,manifest.json} instead of R2")
    args = ap.parse_args()

    run_start = [None]
    if args.once:
        st, now = snapshot(args.run, args, run_start)
        print(render(args.run, st, args, now))
        return 0
    try:
        while True:
            st, now = snapshot(args.run, args, run_start)
            sys.stdout.write("\033[2J\033[H")  # clear + home
            sys.stdout.write(render(args.run, st, args, now) + "\n")
            sys.stdout.write(c(f"\n  refresh {args.interval:.0f}s · Ctrl-C to exit", DIM, sys.stdout.isatty()) + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
