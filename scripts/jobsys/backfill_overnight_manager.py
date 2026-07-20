#!/usr/bin/env python3
# Autonomous overnight backfill manager. Runs on the dev box (nohup), independent of the chat session.
# Keeps the whole scoring fleet under a PRICE cap (EUR/hr, not a box count — box prices vary 8x), on
# CHEAP cx43 boxes (8 vCPU @ EUR0.0296/hr, 4.4x cheaper than cpx42). Every cycle: discover runs (the 6
# byte-range tar-runs bf-z*-t* + the direct-object zenjpeg run bf-zjl2), measure done/total, and launch
# ONE oversubscribed box per undone run without a live box while the projected fleet price stays under
# MAX_EUR. Boxes self-destruct on drain, so the fleet is self-bounding; this only rolls coverage and
# refills as boxes finish. Exits when every run (>= EXPECTED) is complete, then tears down the index box.
import os, sys, json, gzip, time, subprocess, re
import pyarrow.dataset as ds, pyarrow.fs as fs

REPO = "/home/lilith/work/zen/zenmetrics"
MAX_EUR = float(os.environ.get("MAX_EUR", "1.6"))     # scoring-fleet ceiling EUR/hr (+ ~0.13 index box < $2)
CYCLE = int(os.environ.get("CYCLE", "300"))
EXPECTED = int(os.environ.get("EXPECTED_RUNS", "54")) # 53 tar-runs + bf-zjl2
DEADLINE = time.time() + float(os.environ.get("HOURS", "8")) * 3600
LOG = os.path.expanduser("~/tmp/hz720/overnight.log")
TOTALS = os.path.expanduser("~/tmp/hz720/totals.json")
DONEF = os.path.expanduser("~/tmp/hz720/done_runs.txt")
TYPES = os.environ.get("TYPES", "cx43 cx33 cx23")     # cheap-first; launcher falls back down the list
PRICE = {"cx23": 0.0104, "cx33": 0.0160, "cx43": 0.0296, "cx41": 0.0, "cpx42": 0.1314, "cpx52": 0.19}
NEXT_EUR = PRICE.get(TYPES.split()[0], 0.03)          # price assumed for the next box launched
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
    tok = None
    for line in open(os.path.expanduser("~/.config/hetzner/credentials")):
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
    print(line, flush=True); open(LOG, "a").write(line + "\n")

def s5(*a):
    return subprocess.run(["s5cmd", "--endpoint-url", E["EP"], *a],
                          env=dict(E, AWS_ACCESS_KEY_ID=E["R2_ACCESS_KEY_ID"], AWS_SECRET_ACCESS_KEY=E["R2_SECRET_ACCESS_KEY"], AWS_REGION="auto"),
                          capture_output=True, text=True)

def runs():
    r = s5("ls", "s3://zentrain/jobs/")
    out = set()
    for l in r.stdout.splitlines():
        m = re.search(r'\b(bf-(?:zavif|zjxll|zwebp|zjxlm|zpng)-t\d+|bf-zjl2)/', l)
        if m: out.add(m.group(1))
    return sorted(out)

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

def live():
    r = subprocess.run(["hcloud", "server", "list", "-o", "columns=name,type"], env=E, capture_output=True, text=True)
    boxes = []
    for l in r.stdout.splitlines()[1:]:
        p = l.split()
        if len(p) >= 2 and p[0].startswith("hzsf-"):
            boxes.append((p[0], p[1]))
    return boxes

def fleet_eur(boxes):
    return sum(PRICE.get(t, 0.03) for _, t in boxes)

def launch(run):
    env = dict(E, ZEN_BUCKET="zentrain", ZEN_CORPUS_PREFIX_OVERRIDE="refs/clean-picker-corpus-2026-06-26",
               ZEN_RAYON_THREADS="1", ZEN_CHUNK_WALL_SEC="60", ZEN_CORE_OVERSUBSCRIBE="3",
               ZEN_IDLE_PASSES="10", RESUME="1", TYPES=TYPES, LOCATIONS="fsn1 nbg1 hel1",
               HCLOUD_TOKEN=E["HCLOUD_TOKEN"])
    if run == "bf-zjl2":  # zenjpeg = direct-object (individual encodes exist), not byte-range
        env["ZEN_TAR_OVERRIDE"] = "s3://zentrain/x/none.tar"
        env["ZEN_ENCODES_PREFIX"] = "canonical/2026-06-27/zenjpeg_lossy/encodes"
        env["ZEN_ENCODES_BUCKET"] = "zentrain"
    else:                 # byte-range: reconstruct the tar URI from the run name
        tag = run[3:run.index("-t")]; n = run[run.rindex("-t") + 2:]
        env["ZEN_TAR_OVERRIDE"] = "s3://zentrain/jxl-lossy/runs/%s/variants/box-%s.tar" % (SWEEP[tag], n)
    subprocess.run(["bash", "scripts/jobsys/hetzner_scorefile_launch.sh", run, "1"],
                   cwd=REPO, env=env, capture_output=True, text=True, timeout=180)

def main():
    log("overnight START max_eur=%.2f types='%s' cycle=%ds deadline=%.1fh" % (MAX_EUR, TYPES, CYCLE, (DEADLINE - time.time()) / 3600))
    while time.time() < DEADLINE:
        rs = runs()
        done = set(open(DONEF).read().split()) if os.path.exists(DONEF) else set()
        boxes = live()
        eur = fleet_eur(boxes)
        tot_done = tot_all = 0; undone = []
        for run in rs:
            tj = total_jobs(run)
            if tj is None: continue
            dj = done_jobs(run); tot_done += dj; tot_all += tj
            if dj >= tj * 0.999:
                if run not in done: open(DONEF, "a").write(run + "\n"); done.add(run)
                continue
            if not any(("hzsf-%s-" % run) in b for b, _ in boxes):
                undone.append((run, tj - dj))
        undone.sort(key=lambda x: -x[1])
        launched = 0
        for run, rem in undone:
            if eur + NEXT_EUR > MAX_EUR: break
            launch(run); eur += NEXT_EUR; launched += 1
            log("  launch %s (rem~%d)" % (run, rem))
        pct = 100 * tot_done / tot_all if tot_all else 0
        log("runs=%d done=%d undone=%d boxes=%d launched=%d | %d/%d var (%.1f%%) EUR%.2f/hr" %
            (len(rs), len(done), len(undone), len(boxes), launched, tot_done * 12, tot_all * 12, pct, eur))
        if len(rs) >= EXPECTED and len(done) >= len(rs) and not boxes:
            log("ALL RUNS COMPLETE (%d)" % len(rs)); break
        time.sleep(CYCLE)
    subprocess.run(["hcloud", "server", "delete", "hz-tar-index"], env=E, capture_output=True)
    log("overnight EXIT (index box torn down)")

if __name__ == "__main__":
    main()
