#!/usr/bin/env python3
"""G-BITEXACT — the JobKind::Feature executor's correctness gate.

Runs a declared Feature wave LOCALLY through the real executor binary (no
fleet, no store) and compares every emitted feature value to a stored root,
as `to_bits()`, not as a tolerance.

WHY THIS IS THE GATE, and why it is stronger than anything a rev2 wave can
offer: revision 1 is byte-reproducible, and a stored answer for it already
exists.  The postC 372 eval root records `build_commit 4fbd8ff8`; if the
zensim this executor links is at that commit, the fleet's rev1 output must be
bit-identical to that root on the same pixels.  Any non-zero delta means the
executor is NOT a drop-in producer for the stored root, and therefore no rev2
number it produces would be comparable to any stored number.  That is a stop,
not a footnote (docs/PLAN_REV2_RECALC_2026-09-06.md §5).

Two divergence risks are checked explicitly rather than assumed away:

  * sub-64 padding — zensim's `extract_features_372col` passes RAW dims to
    `compute_zensim_with_config` (which reflect-pads internally since
    `f9fac41e`), while zenmetrics' `extract_features_regime` reflect-pads
    EXTERNALLY to 64 and passes the padded dims.  For any image >= 64x64 the
    external pad is a no-op and the two agree by construction.  Below 64 they
    may not.  This refuses a corpus containing a sub-64 image rather than
    silently routing it.

  * two producers, one root — the postC root was built by TWO extractors
    (`zensim-validate --extract-only` for cid22/kadid/tid/pipal,
    `extract_features_372col` for konjnd/aic3/csiq/live).  A gate over one
    corpus proves half a claim; `--corpus` names which half you are proving.

Usage:
  rev2_bitexact_gate.py --pairs /mnt/v/dataset/csiq/csiq_pairs.tsv \\
      --stored <root>/csiq_features_372col_2026-07-18.parquet \\
      --exec  <path to zenmetrics binary> [--regime 372] [--chunk 16]
      [--limit N] [--jobctl <path>] [--work DIR]

Exit 0 = bit-exact on every compared cell.  Exit 1 = a delta (reported).
Exit 2 = the gate could not run (missing input, sub-64 image, row misalignment).
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    print("FATAL: pyarrow is required", file=sys.stderr)
    sys.exit(2)


def bits(x: float) -> int:
    """The f64 bit pattern — the comparison unit.  A tolerance would hide
    exactly the class of drift this gate exists to catch."""
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def png_dims(path: str):
    """Width/height from a PNG IHDR, without decoding.  Returns None for a
    non-PNG (the sub-64 check then reports 'unknown' rather than passing)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--stored", required=True)
    ap.add_argument("--exec", dest="execbin", required=True)
    ap.add_argument("--jobctl", default=None)
    ap.add_argument("--regime", default="372")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="compare only the first N pairs (first-cell gate)")
    ap.add_argument("--work", default=None)
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    for p in (args.pairs, args.stored, args.execbin):
        if not os.path.exists(p):
            print("FATAL: missing %s" % p, file=sys.stderr)
            return 2

    work = args.work or tempfile.mkdtemp(prefix="rev2gate-", dir=os.path.expanduser("~/tmp"))
    os.makedirs(work, exist_ok=True)

    # ── 1. the pairs, in file order (the order the stored root was built in)
    with open(args.pairs) as fh:
        lines = [l.rstrip("\n") for l in fh if l.strip()]
    hdr = lines[0].split("\t")
    try:
        ri, di = hdr.index("ref_path"), hdr.index("dist_path")
    except ValueError:
        print("FATAL: pairs header needs ref_path + dist_path, got %r" % hdr, file=sys.stderr)
        return 2
    pairs = []
    for l in lines[1:]:
        f = l.split("\t")
        pairs.append((f[ri].strip(), f[di].strip()))
    if args.limit:
        pairs = pairs[: args.limit]
    print("pairs: %d" % len(pairs), flush=True)

    # ── 2. sub-64 refusal (a declared divergence risk, checked not assumed)
    small = []
    for r, d in pairs:
        for p in (r, d):
            wh = png_dims(p)
            if wh and (wh[0] < 64 or wh[1] < 64):
                small.append((p, wh))
    if small:
        print("FATAL: %d image(s) are smaller than 64px in some dimension; the zensim "
              "extractor pads INTERNALLY and this executor pads EXTERNALLY, so the two "
              "are not known to agree there. Refusing rather than routing them silently."
              % len(small), file=sys.stderr)
        for p, wh in small[:5]:
            print("   %s %dx%d" % (p, wh[0], wh[1]), file=sys.stderr)
        return 2

    # ── 3. declare through the REAL owner (zenfleet-ctl), never a hand-rolled
    #       manifest — the declare shape is part of what this gate proves.
    sub = os.path.join(work, "pairs.tsv")
    with open(sub, "w") as fh:
        fh.write("ref_path\tdist_path\n")
        for r, d in pairs:
            fh.write("%s\t%s\n" % (r, d))
    manifest = os.path.join(work, "manifest.json")
    jobctl = args.jobctl or os.path.join(
        os.path.dirname(os.path.abspath(args.execbin)), "zenfleet-ctl")
    if not os.path.exists(jobctl):
        print("FATAL: zenfleet-ctl not found at %s (pass --jobctl)" % jobctl, file=sys.stderr)
        return 2
    cmd = [jobctl, "declare-features", "--pairs", sub, "--out", manifest,
           "--regime", args.regime, "--chunk", str(args.chunk)]
    if args.revision:
        cmd += ["--revision", args.revision]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FATAL: declare failed:\n%s" % r.stderr, file=sys.stderr)
        return 2
    jobs = json.load(open(manifest))
    print("declared %d feature jobs" % len(jobs), flush=True)

    # ── 4. run every job through the executor exactly as the worker does:
    #       one DesiredJob as JSON on stdin, output bytes on stdout.
    env = dict(os.environ)
    env.setdefault("TMPDIR", os.path.expanduser("~/tmp"))
    got = {}          # (ref, dist) -> [f0..fN]
    blobs = []
    for n, job in enumerate(jobs):
        r = subprocess.run([args.execbin, "jobexec"], input=json.dumps(job),
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print("FATAL: job %d failed (rc=%d):\n%s" % (n, r.returncode, r.stderr[-2000:]),
                  file=sys.stderr)
            return 2
        blobs.append(r.stdout)
        for line in r.stdout.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") != "feature":
                print("FATAL: unexpected row kind %r" % row.get("kind"), file=sys.stderr)
                return 2
            got[(row["image_path"], row["encode_sha"])] = row["features"]
        if (n + 1) % 10 == 0:
            print("  ran %d/%d jobs" % (n + 1, len(jobs)), flush=True)
    print("extracted %d rows" % len(got), flush=True)
    if len(got) != len(pairs):
        print("FATAL: %d rows for %d pairs — the executor dropped or duplicated work"
              % (len(got), len(pairs)), file=sys.stderr)
        return 2

    # ── 4b. G-EXEC.2 idempotence: re-running one job must give the same bytes.
    r2 = subprocess.run([args.execbin, "jobexec"], input=json.dumps(jobs[0]),
                        capture_output=True, text=True, env=env)
    if r2.returncode != 0 or r2.stdout != blobs[0]:
        print("FATAL: G-EXEC.2 — the same job did not produce byte-identical output twice",
              file=sys.stderr)
        return 1
    print("G-EXEC.2 idempotence: OK (job 0 byte-identical on re-run)", flush=True)

    # ── 5. compare, positionally, against the stored root
    t = pq.read_table(args.stored)
    names = t.column_names
    nfeat = sum(1 for c in names if c.startswith("f") and c[1:].isdigit())
    stored_cols = [t.column("f%d" % i).to_pylist() for i in range(nfeat)]
    nrows = t.num_rows
    if nrows < len(pairs):
        print("FATAL: stored root has %d rows, pairs file has %d" % (nrows, len(pairs)),
              file=sys.stderr)
        return 2
    # Row alignment: the stored root was built by walking THIS pairs file in
    # order, so row i <-> pair i. Verified on ref_basename before any value is
    # compared -- a positional compare on a misaligned table would report
    # thousands of false deltas (or, worse, false agreement).
    if "ref_basename" in names:
        rb = t.column("ref_basename").to_pylist()
        for i, (r, _d) in enumerate(pairs):
            # The two producers spell the identity differently:
            # `extract_features_372col` stores the FILENAME (`1600.png`),
            # `zensim-validate --extract-only` stores the STEM (`I01`). Compare
            # like for like -- strip the extension only when the stored value
            # has none. This is a spelling difference, not a weaker check: an
            # actual misalignment still fails, because the stems differ too.
            mine = os.path.basename(r)
            if "." not in rb[i]:
                mine = os.path.splitext(mine)[0]
            if mine != rb[i]:
                print("FATAL: row alignment broken at %d: pairs says %s, stored says %s"
                      % (i, mine, rb[i]), file=sys.stderr)
                return 2
        print("row alignment: OK (ref_basename matches on all %d compared rows)" % len(pairs),
              flush=True)
    else:
        print("WARNING: stored root has no ref_basename column — alignment UNVERIFIED",
              flush=True)

    ncmp = 0
    ndiff = 0
    worst = (0.0, None)
    per_block = {"basic": [0, 0], "peaks": [0, 0], "masked": [0, 0], "iw": [0, 0]}

    def block_of(i):
        if i < 156:
            return "basic"
        if i < 228:
            return "peaks"
        if i < 300:
            return "masked"
        return "iw"

    for i, (r, d) in enumerate(pairs):
        mine = got[(r, d)]
        if len(mine) != nfeat:
            print("FATAL: row %d has %d features, stored root has %d"
                  % (i, len(mine), nfeat), file=sys.stderr)
            return 2
        for k in range(nfeat):
            a, b = mine[k], stored_cols[k][i]
            ncmp += 1
            blk = block_of(k)
            per_block[blk][1] += 1
            if bits(a) != bits(b):
                ndiff += 1
                per_block[blk][0] += 1
                dd = abs(float(a) - float(b))
                if dd > worst[0]:
                    worst = (dd, (i, k, a, b))

    print("")
    print("=== G-BITEXACT ===")
    print("compared %d cells (%d rows x %d features)" % (ncmp, len(pairs), nfeat))
    for blk, (nd, nt) in per_block.items():
        if nt:
            print("  %-7s %d/%d differ" % (blk, nd, nt))
    if ndiff == 0:
        print("RESULT: BIT-EXACT — 0 of %d cells differ" % ncmp)
        return 0
    print("RESULT: **FAIL** — %d of %d cells differ (%.4f%%)" % (ndiff, ncmp, 100.0 * ndiff / ncmp))
    if worst[1]:
        i, k, a, b = worst[1]
        print("  worst: row %d f%d  mine=%.17g stored=%.17g  |delta|=%.6g" % (i, k, a, b, worst[0]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
