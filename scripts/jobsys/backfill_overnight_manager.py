#!/usr/bin/env python3
# Autonomous overnight backfill manager. Runs on the dev box (nohup), independent of the chat session.
# Every cycle: discover declared byte-range runs (bf-z*-t*), measure per-run done/total, keep the fleet
# at <= CAP boxes (~$2/hr) by launching ONE oversubscribed byte-range box per undone run without a live
# box, log a progress report. Boxes self-destruct on drain, so the fleet is self-bounding; this only
# ROLLS coverage across the ~53 tar-runs and refills as boxes finish. Exits when every run is complete
# (or DEADLINE), then tears down the index box. CAP counts ALL hzsf-bf boxes (incl. the draining
# zenjpeg fleet), so total spend stays under the cap even while zenjpeg finishes.
import os, sys, json, gzip, time, subprocess, re
import pyarrow.dataset as ds, pyarrow.fs as fs

REPO = "/home/lilith/work/zen/zenmetrics"
CAP = int(os.environ.get("CAP", "56"))              # total hzsf-bf boxes (cpx42 ~= $0.033/hr -> ~$1.85/hr)
EXPECTED = int(os.environ.get("EXPECTED_RUNS", "53"))  # don't declare "all complete" until ~all tars indexed
CYCLE = int(os.environ.get("CYCLE", "420"))
DEADLINE = time.time() + float(os.environ.get("HOURS", "7")) * 3600
LOG = os.path.expanduser("~/tmp/hz720/overnight.log")
TOTALS = os.path.expanduser("~/tmp/hz720/totals.json")
DONEF = os.path.expanduser("~/tmp/hz720/done_runs.txt")
SWEEP = {"zavif": "mandfix4-zenavif-1782593621", "zjxll": "jxl-lossy-vardct-1782609551",
         "zwebp": "mandfix2-zenwebp-1782584881", "zjxlm": "jxl-modular-1782596759",
         "zpng": "mandfix2-zenpng-1782584881"}
CODEC = {"zavif": "zenavif", "zjxll": "zenjxl", "zwebp": "zenwebp", "zjxlm": "zenjxl", "zpng": "zenpng"}

def envs():
    e = dict(os.environ)
    for line in open(os.path.expanduser("~/.config/cloudflare/r2-credentials")):
        line = line.strip()
        if line.startswith("R2_") and "=" in line:
            k, v = line.split("=", 1); e[k] = v.strip().strip('"').strip("'")
    e["R2_ACCOUNT_ID"] = e.get("R2_ACCOUNT_ID", "")
    tok = None
    p = os.path.expanduser("~/.config/hetzner/credentials")
    for line in open(p):
        if line.startswith("api_token="):
            tok = line.split("=", 1)[1].strip()
    e["HCLOUD_TOKEN"] = tok or ""
    e["EP"] = "https://%s.r2.cloudflarestorage.com" % e["R2_ACCOUNT_ID"]
    return e

E = envs()
S3 = fs.S3FileSystem(access_key=E["R2_ACCESS_KEY_ID"], secret_key=E["R2_SECRET_ACCESS_KEY"],
                     endpoint_override=E["EP"], region="auto")

def log(m):
    line = "[%s] %s" % (time.strftime("%H:%M:%SZ", time.gmtime()), m)
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")

def s5(*a):
    return subprocess.run(["s5cmd", "--endpoint-url", E["EP"], *a], env=dict(E, AWS_ACCESS_KEY_ID=E["R2_ACCESS_KEY_ID"], AWS_SECRET_ACCESS_KEY=E["R2_SECRET_ACCESS_KEY"], AWS_REGION="auto"), capture_output=True, text=True)

def runs():
    r = s5("ls", "s3://zentrain/jobs/")
    out = []
    for l in r.stdout.splitlines():
        m = re.search(r'\bbf-(zavif|zjxll|zwebp|zjxlm|zpng)-t\d+/', l)
        if m: out.append(l.split()[-1].rstrip("/"))
    return sorted(set(out))

def total_jobs(run):
    tot = json.load(open(TOTALS)) if os.path.exists(TOTALS) else {}
    if run in tot: return tot[run]
    try:
        with S3.open_input_file("zentrain/jobs/%s/manifest.json.gz" % run) as f:
            n = len(json.loads(gzip.decompress(f.read())))
    except Exception:
        return None
    tot[run] = n; json.dump(tot, open(TOTALS, "w")); return n

def done_jobs(run):
    try:
        t = ds.dataset("zentrain/jobs/%s/ledger/" % run, filesystem=S3, format="parquet").to_table(columns=["job_id", "status"])
        return len(set(j for j, s in zip(t.column("job_id").to_pylist(), t.column("status").to_pylist()) if s == "done"))
    except Exception:
        return 0

def live_boxes():
    r = subprocess.run(["hcloud", "server", "list", "-o", "columns=name"], env=E, capture_output=True, text=True)
    return [l.strip() for l in r.stdout.splitlines() if "hzsf-" in l]

def tar_of(run):
    tag = run[3:run.index("-t")]; n = run[run.rindex("-t") + 2:]
    return "s3://zentrain/jxl-lossy/runs/%s/variants/box-%s.tar" % (SWEEP[tag], n), CODEC[tag]

def launch(run):
    tar, _codec = tar_of(run)
    env = dict(E, ZEN_BUCKET="zentrain", ZEN_TAR_OVERRIDE=tar,
               ZEN_CORPUS_PREFIX_OVERRIDE="refs/clean-picker-corpus-2026-06-26",
               ZEN_RAYON_THREADS="1", ZEN_CHUNK_WALL_SEC="60", ZEN_CORE_OVERSUBSCRIBE="3",
               ZEN_IDLE_PASSES="10", RESUME="1", TYPES="cpx42 cx33 cpx52 cpx41",
               LOCATIONS="fsn1 nbg1 hel1", HCLOUD_TOKEN=E["HCLOUD_TOKEN"])
    subprocess.run(["bash", "scripts/jobsys/hetzner_scorefile_launch.sh", run, "1"],
                   cwd=REPO, env=env, capture_output=True, text=True, timeout=180)

def mark_done():
    done = set(open(DONEF).read().split()) if os.path.exists(DONEF) else set()
    return done

def main():
    log("overnight manager START cap=%d cycle=%ds deadline=%.1fh" % (CAP, CYCLE, (DEADLINE - time.time()) / 3600))
    while time.time() < DEADLINE:
        rs = runs()
        done = mark_done()
        live = live_boxes()
        nlive = sum(1 for b in live if b.startswith("hzsf-bf"))
        tot_done = tot_all = 0
        undone = []
        for run in rs:
            tj = total_jobs(run)
            if tj is None: continue
            dj = done_jobs(run)
            tot_done += dj; tot_all += tj
            complete = dj >= tj * 0.999
            if complete:
                if run not in done:
                    open(DONEF, "a").write(run + "\n"); done.add(run)
                continue
            has_box = any(("hzsf-%s-" % run) in b for b in live)
            if not has_box: undone.append((run, tj - dj))
        # launch undone (biggest remaining first) up to CAP
        undone.sort(key=lambda x: -x[1])
        launched = 0
        for run, rem in undone:
            if nlive >= CAP: break
            launch(run); nlive += 1; launched += 1
            log("  launch %s (rem~%d jobs)" % (run, rem))
        pct = 100 * tot_done / tot_all if tot_all else 0
        log("runs=%d done=%d undone=%d live=%d launched=%d | variants %d/%d (%.1f%%) ~$%.2f/hr" %
            (len(rs), len(done), len(undone), nlive, launched, tot_done * 12, tot_all * 12, pct, nlive * 0.033))
        # Only declare victory once ~all tars are indexed (len(rs) >= EXPECTED) AND every declared run
        # is complete AND no boxes remain — otherwise an early lull (first runs drain before later tars
        # finish indexing) would exit + tear down the index box, abandoning the rest.
        if len(rs) >= EXPECTED and len(done) >= len(rs) and nlive == 0:
            log("ALL RUNS COMPLETE (%d runs)" % len(rs)); break
        time.sleep(CYCLE)
    # teardown index box
    subprocess.run(["hcloud", "server", "delete", "hz-tar-index"], env=E, capture_output=True)
    log("overnight manager EXIT (index box torn down)")

if __name__ == "__main__":
    main()
