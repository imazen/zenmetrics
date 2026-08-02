#!/usr/bin/env python3
# declare_bf944.py — declare the 944-feature (folded+append2 streaming zensim) backfill runs by
# cloning the 54 drained bf924-* runs with the metric swapped to "zensim-foldapp2".
#
# SOTA-944 campaign P1 bigcodec leg (zensim docs/PLAN_SOTA944_CAMPAIGN_2026-08-01.md; recipe in
# zensim benchmarks/backfill944_2026-08-01.md "bigcodec decision"). Extraction math is the P1.5
# ADJUDICATED default: append2_block ON, append2_dst_activity OFF (zensim >= d0616362).
#
# Per run bf924-<x> (53 tar-mode + the direct-object zenjpeg run bf924-zjl2):
#   * fetch  jobs/bf924-<x>/manifest.json.gz, rewrite every cell's kind.metrics ->
#     ["zensim-foldapp2"] (the ONLY metric — same single-metric discipline as the 924 wave),
#     upload as jobs/bf944-<x>/manifest.json.gz
#   * server-side copy jobs/bf924-<x>/variant_index.tsv -> jobs/bf944-<x>/variant_index.tsv
#     (tar-mode runs only; same TARs, same byte-range index — nothing re-encodes)
#   * emit the runlist row `bf944-<x>\t<same src>\t<same mode>`
# Then upload the new-wave runlist to jobs/_pool944/runlist.tsv — a SEPARATE pool from
# jobs/_pool924/runlist.tsv, so a stale exec-zensim924 image can never claim 944 cells (it would
# fail loud on the unknown metric anyway, but it should never see them).
#
# Stale-image safety is layered exactly like bf924: new runlist URI (workers must be repointed
# deliberately) + new metric string (old binaries error, never emit 924-wide rows into a 944 run).
#
# Usage: python3 scripts/jobsys/declare_bf944.py [--dry-run]
import gzip
import io
import json
import os
import subprocess
import sys

BUCKET = "zentrain"
OLD_METRIC = "zensim-foldapp"
METRIC = "zensim-foldapp2"
# tag -> (sweep, ntars) — mirrors declare_bf924.py / pool_launch.sh gen_runlist (the 720 wave).
TAR_RUNS = [
    ("zavif", "mandfix4-zenavif-1782593621", 8),
    ("zjxll", "jxl-lossy-vardct-1782609551", 24),
    ("zwebp", "mandfix2-zenwebp-1782584881", 9),
    ("zjxlm", "jxl-modular-1782596759", 10),
    ("zpng", "mandfix2-zenpng-1782584881", 2),
]
ENC_RUN = ("zjl2", "canonical/2026-06-27/zenjpeg_lossy/encodes")


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


def main():
    dry = "--dry-run" in sys.argv
    runs = []  # (old, new, src, mode)
    for tag, sweep, ntars in TAR_RUNS:
        for i in range(ntars):
            runs.append((
                f"bf924-{tag}-t{i}", f"bf944-{tag}-t{i}",
                f"s3://{BUCKET}/jxl-lossy/runs/{sweep}/variants/box-{i}.tar", "tar",
            ))
    runs.append((f"bf924-{ENC_RUN[0]}", f"bf944-{ENC_RUN[0]}", ENC_RUN[1], "enc"))
    print(f"{len(runs)} runs to declare (dry_run={dry})")

    runlist, total_cells, fail = [], 0, 0
    for old, new, src, mode in runs:
        r = s5("cat", f"s3://{BUCKET}/jobs/{old}/manifest.json.gz")
        if r.returncode != 0:
            print(f"FAIL fetch {old}: {r.err}")
            fail += 1
            continue
        raw = r.stdout
        cells = json.load(gzip.open(io.BytesIO(raw)))
        for c in cells:
            k = c.get("kind")
            if not isinstance(k, dict) or k.get("kind") != "score_file":
                print(f"FAIL {old}: unexpected cell kind {str(k)[:80]} — refusing to clone")
                fail += 1
                cells = None
                break
            if k.get("metrics") != [OLD_METRIC]:
                print(f"FAIL {old}: expected metrics=[{OLD_METRIC!r}], got {k.get('metrics')!r} — refusing to clone")
                fail += 1
                cells = None
                break
            k["metrics"] = [METRIC]
        if cells is None:
            continue
        total_cells += len(cells)
        if not dry:
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                gz.write(json.dumps(cells).encode())
            r = s5("pipe", f"s3://{BUCKET}/jobs/{new}/manifest.json.gz", stdin=buf.getvalue())
            if r.returncode != 0:
                print(f"FAIL upload {new}: {r.err}")
                fail += 1
                continue
            if mode == "tar":
                r = s5("cp", f"s3://{BUCKET}/jobs/{old}/variant_index.tsv",
                       f"s3://{BUCKET}/jobs/{new}/variant_index.tsv")
                if r.returncode != 0:
                    print(f"FAIL index copy {new}: {r.err}")
                    fail += 1
                    continue
        runlist.append(f"{new}\t{src}\t{mode}")
        print(f"ok {new}: {len(cells)} cells")

    if fail:
        print(f"ABORT: {fail} failures — runlist NOT uploaded")
        sys.exit(1)
    rl = "\n".join(runlist) + "\n"
    if dry:
        print(rl[:400])
    else:
        r = s5("pipe", f"s3://{BUCKET}/jobs/_pool944/runlist.tsv", stdin=rl.encode())
        if r.returncode != 0:
            print(f"FAIL runlist upload: {r.err}")
            sys.exit(1)
        print(f"runlist: {len(runlist)} runs -> s3://{BUCKET}/jobs/_pool944/runlist.tsv")
    print(f"total cells: {total_cells}")


if __name__ == "__main__":
    main()
