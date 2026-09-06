#!/usr/bin/env python3
"""Stage the 372 eval root's RE-EXTRACTABLE corpora's pixels to the LAN store.

Prerequisite for REV2 RECALCULATION wave 2 (docs/PLAN_REV2_RECALC_2026-09-06.md
§2, §7.3): "the 372 corpora are NOT on the LAN store" — their pixels live at
local WSL paths (`/mnt/v/dataset*`) no fleet node can see. This script:

  1. syncs each corpus's local pixel directory to the LAN store (whole
     directory, path-mirrored, via `s5cmd sync` through the endpoint
     `scripts/lib/s3env.sh` resolves — never a hardcoded endpoint);
  2. rewrites that corpus's `ref_path`/`dist_path` pairs TSV (the shape
     `zenfleet-ctl declare-features --pairs` already reads, see
     `crates/zenfleet-ctl/src/lib.rs::parse_feature_pairs`) to the `s3://`
     keys the executor's `resolve_source` / `resolve_feature_input` already
     fetch in-process (`crates/zenmetrics-cli/src/jobexec.rs` — NO code
     change needed there; both functions handle `s3://` today);
  3. writes a per-corpus `_MANIFEST.json` (rows, referenced-file count,
     sha256 per REFERENCED file, source path, sync date, corpus root, s3
     prefix) under the manifest dir.

ONLY the 8 corpora the postC 372 root's own manifest marks `"source":
"fresh:*"` are staged (cid22, kadid, tid, csiq, live, pipal, konjnd, aic3).
The other 6 (aic4, nonphoto, imazen26, sdr25, hfnlproxy, hf_nearlossless) are
`"source": "stored"` — COPIED FROM THE OLD ROOT, not re-extractable on this
box — so there is nothing to stage for them; see
`/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC/_MANIFEST.json`.

PIPAL CAVEAT (read before trusting row-for-row parity): the postC root's
pipal table has 21,800 rows; the full PIPAL train set this script reconstructs
from `Train_Label/*.txt` has 23,200 (200 refs x 116 distortions each, verified
by direct enumeration). Every group-00..06 loses EXACTLY 7 entries per
reference in the stored table (200 x 7 = 1,400 = 23,200 - 21,800) with no
documented selection rule found in `zensim-validate::load_pipal` (which
returns the full un-truncated list) or in `--max-images` (default 0 = all,
and `build_eval372_root.sh` does not pass it). This script stages the FULL
23,200-pair PIXEL set (a superset - nothing is missing) and writes the FULL
23,200-row pairs TSV, flagged `"row_parity_vs_stored_root": "NOT VERIFIED
(superset)"` in its manifest. Reconciling the exact 21,800-row historical
selection is wave-2 DECLARE-time work (this lane's own plan doc, not this
script's job to guess).

cid22/kadid/tid/csiq/live/aic3/konjnd are staged from PAIRS TSVs verified
row-for-row (order + values) against the postC root's own extraction CSVs
(csiq/live are literally the same file the postC build read; kadid/tid/cid22
were spot-checked by ref-group boundary; aic3/konjnd were checked cell-by-cell
against every row) -- see benchmarks/ for the checked commands.

Usage:
  scripts/jobsys/stage_eval372_corpora_lan.py [--dry-run] [--corpus NAME ...]
  scripts/jobsys/stage_eval372_corpora_lan.py --verify-only   # re-hash + diff, no sync
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

S3_BUCKET = "codec-corpus"
S3_PREFIX_BASE = "eval372-rev2-2026-09-06"
# Both the per-corpus manifests (up to ~2.9 MB each, thousands of sha256
# entries) and the rewritten pairs TSVs are block-storage artifacts, not repo
# content (ML pipeline discipline #7b — anything >30 KB that isn't source
# goes to /mnt/v with a git-tracked pointer, not into the repo). The repo
# carries only `docs/rev2_lan_stage.pointer.md`.
STAGE_ROOT = Path("/mnt/v/zen/zensim-training/rev2-lan-stage-2026-09-06")
MANIFEST_DIR = STAGE_ROOT / "manifests"
LAN_TSV_DIR = STAGE_ROOT / "pairs"

# name -> (local_root, local_pairs_tsv, extra_cols_kept, note)
CORPORA = {
    "cid22": (
        "/mnt/v/dataset/cid22/CID22_validation_set",
        "/mnt/v/dataset/cid22/CID22_validation_set/cid22val_pairs_ab.tsv",
        "verified: literal ab.tsv used elsewhere in the pipeline; rows=4292 matches postC",
    ),
    "kadid": (
        "/mnt/v/dataset/kadid10k",
        "/mnt/v/dataset/kadid10k/kadid_pairs_ab.tsv",
        "verified: rows=10125 matches postC; ref-group boundaries (I01..I81) match order",
    ),
    "tid": (
        "/mnt/v/dataset/tid2013",
        "/mnt/v/dataset/tid2013/tid_pairs_ab.tsv",
        "verified: rows=3000 matches postC; ref-group boundaries match order",
    ),
    "csiq": (
        "/mnt/v/dataset/csiq",
        "/mnt/v/dataset/csiq/csiq_pairs.tsv",
        "EXACT: this is literally the file build_eval372_root.sh fed to `ex csiq pairs-tsv`",
    ),
    "live": (
        "/mnt/v/datasets/LIVE",
        "/mnt/v/datasets/LIVE/live_r2_pairs.tsv",
        "EXACT: this is literally the file build_eval372_root.sh fed to `ex live pairs-tsv` "
        "(.bmp paths -- NOT live_r2_pairs_png.tsv, which the plan doc's G7.2 blocker names as "
        "misaligned against the stored table)",
    ),
    "aic3": (
        "/mnt/v/dataset/aic3_ctc_epfl",
        "/mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv",
        "verified row-for-row (ref_basename,human_score) identical, in order, vs postC's aic3.csv",
    ),
    "konjnd": (
        "/mnt/v/datasets/KonJND-1k/KonJND-1k",
        None,  # generated below by build_konjnd_pairs()
        "verified row-for-row (image_id,mean) identical, in order, vs postC's konjnd.csv",
    ),
    "pipal": (
        "/mnt/v/dataset/pipal",
        None,  # generated below by build_pipal_pairs()
        "NOT VERIFIED (superset): full 23200-pair PIPAL train set staged; postC root has 21800 "
        "(200 refs x -7 each, mechanism undocumented) -- see module docstring",
    ),
}


def build_konjnd_pairs(out_path: Path) -> int:
    base = "/mnt/v/datasets/KonJND-1k/KonJND-1k"
    rows = []
    with open(f"{base}/subjective_ratings.csv", newline="") as f:
        r = csv.reader(f)
        next(r)
        for rec in r:
            if len(rec) < 5:
                continue
            image_id, comp, _n, mean_s = rec[0], rec[1], rec[2], rec[3]
            try:
                mean_threshold = float(mean_s)
            except ValueError:
                continue
            stem = image_id[:-4] if image_id.endswith(".png") else image_id
            if not stem:
                continue
            level = int(round(min(max(mean_threshold, 1.0), 100.0)))
            if comp == "JPEG":
                subdir, ext = "jpeg", "jpg"
            elif comp == "BPG":
                subdir, ext = "bpg", "png"
            else:
                continue
            dist_path = f"{base}/{subdir}/{stem}_{comp}_{level:03d}.{ext}"
            if not os.path.exists(dist_path):
                continue
            rows.append((f"{base}/source_image/{image_id}", dist_path, f"{mean_threshold}"))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score"])
        w.writerows(rows)
    return len(rows)


def build_pipal_pairs(out_path: Path) -> int:
    base = "/mnt/v/dataset/pipal"
    label_dir, ref_dir = f"{base}/Train_Label", f"{base}/Train_Ref"
    dist_dirs = [f"{base}/Distortion_{i}" for i in range(1, 5) if os.path.isdir(f"{base}/Distortion_{i}")]
    rows = []
    for lf in sorted(f for f in os.listdir(label_dir) if f.endswith(".txt")):
        ref_stem = lf[:-4]
        ref_path = f"{ref_dir}/{ref_stem}.bmp"
        if not os.path.exists(ref_path):
            continue
        with open(f"{label_dir}/{lf}") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                dist_name = parts[0].strip()
                dist_path = next((p for d in dist_dirs if os.path.exists(p := f"{d}/{dist_name}")), None)
                if dist_path is None:
                    continue
                rows.append((ref_path, dist_path))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path"])
        w.writerows(rows)
    return len(rows)


def read_pairs_tsv(path: Path):
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return list(r), r.fieldnames


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def s3env():
    """Resolve EP/creds via the canonical resolver (never re-derive an endpoint)."""
    zm_root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        ["bash", "-c", f". {zm_root}/scripts/lib/s3env.sh && env"],
        capture_output=True, text=True, check=True,
    )
    env = dict(os.environ)
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k in ("EP", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "ZEN_S3_STORE"):
                env[k] = v
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--corpus", action="append", default=None, help="restrict to these corpora")
    ap.add_argument("--verify-only", action="store_true", help="skip sync, just (re)hash + rewrite TSV + manifest")
    args = ap.parse_args()

    names = args.corpus or list(CORPORA.keys())
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    LAN_TSV_DIR.mkdir(parents=True, exist_ok=True)
    env = s3env()
    ep = env.get("EP")
    print(f"[s3env] EP={ep} STORE={env.get('ZEN_S3_STORE')}", file=sys.stderr)

    summary = []
    for name in names:
        root, pairs_tsv, note = CORPORA[name]
        s3_prefix = f"s3://{S3_BUCKET}/{S3_PREFIX_BASE}/{name}"
        print(f"\n=== {name} ===  root={root}  ->  {s3_prefix}", file=sys.stderr)

        if pairs_tsv is None:
            gen_path = LAN_TSV_DIR / f"{name}_pairs_src.tsv"
            n = {"konjnd": build_konjnd_pairs, "pipal": build_pipal_pairs}[name](gen_path)
            pairs_tsv = str(gen_path)
            print(f"  generated {n} pairs -> {gen_path}", file=sys.stderr)

        rows, fieldnames = read_pairs_tsv(Path(pairs_tsv))
        assert "ref_path" in fieldnames and "dist_path" in fieldnames, fieldnames

        # 1. sync whole directory (idempotent; re-run-safe)
        if not args.dry_run and not args.verify_only:
            t0 = time.time()
            cmd = ["s5cmd", "--endpoint-url", ep, "sync", f"{root}/", f"{s3_prefix}/"]
            print(f"  $ {' '.join(cmd)}", file=sys.stderr)
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            dt = time.time() - t0
            if r.returncode != 0:
                print(r.stdout[-4000:], file=sys.stderr)
                print(r.stderr[-4000:], file=sys.stderr)
                print(f"  FAILED sync ({dt:.1f}s, rc={r.returncode})", file=sys.stderr)
                return 1
            n_copied = r.stdout.count("cp ")
            print(f"  synced in {dt:.1f}s ({n_copied} objects copied/updated)", file=sys.stderr)
        else:
            print("  [skip sync: dry-run/verify-only]", file=sys.stderr)

        # 2. hash every REFERENCED file (dedup ref+dist), rewrite TSV to s3:// paths
        referenced = {}
        for row in rows:
            for col in ("ref_path", "dist_path"):
                p = row[col]
                if p not in referenced:
                    referenced[p] = None
        missing = [p for p in referenced if not os.path.isfile(p)]
        if missing:
            print(f"  FATAL: {len(missing)} referenced files missing locally, e.g. {missing[:3]}", file=sys.stderr)
            return 1

        t0 = time.time()
        for p in referenced:
            referenced[p] = sha256_file(p)
        print(f"  hashed {len(referenced)} referenced files in {time.time()-t0:.1f}s", file=sys.stderr)

        out_rows = []
        for row in rows:
            new_row = dict(row)
            for col in ("ref_path", "dist_path"):
                rel = os.path.relpath(row[col], root)
                new_row[col] = f"{s3_prefix}/{rel}"
            out_rows.append(new_row)

        lan_tsv_path = LAN_TSV_DIR / f"{name}_pairs_lan.tsv"
        with open(lan_tsv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            w.writeheader()
            w.writerows(out_rows)

        total_bytes = sum(os.path.getsize(p) for p in referenced)
        manifest = {
            "corpus": name,
            "sync_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "local_root": root,
            "s3_prefix": s3_prefix,
            "s3_endpoint_store": env.get("ZEN_S3_STORE"),
            "local_pairs_tsv_source": str(pairs_tsv),
            "lan_pairs_tsv": str(lan_tsv_path),
            "rows": len(rows),
            "referenced_files": len(referenced),
            "referenced_bytes": total_bytes,
            "note": note,
            "sha256": referenced,
        }
        manifest_path = MANIFEST_DIR / f"{name}_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        print(f"  wrote {manifest_path} ({len(referenced)} files, {total_bytes/1e6:.1f} MB referenced)", file=sys.stderr)

        summary.append((name, len(rows), len(referenced), total_bytes))

    print("\n=== SUMMARY ===", file=sys.stderr)
    tot_rows = tot_files = tot_bytes = 0
    for name, rows_n, files_n, bytes_n in summary:
        print(f"  {name:10s} rows={rows_n:6d}  referenced_files={files_n:6d}  bytes={bytes_n/1e6:8.1f} MB", file=sys.stderr)
        tot_rows += rows_n
        tot_files += files_n
        tot_bytes += bytes_n
    print(f"  {'TOTAL':10s} rows={tot_rows:6d}  referenced_files={tot_files:6d}  bytes={tot_bytes/1e6:8.1f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
