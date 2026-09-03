#!/usr/bin/env python3
"""avifdoe_harvest.py — reduce the AVIF-DOE score blobs + encode ledgers to ONE tidy table.

The DOE's four encode runs (a0r/a1/a2/ag) and its score run
(avifdoe-svt-sf-cpu-20260902) are joined here into a single parquet keyed on
`encode_sha`, which is the content address of the encoded AVIF bitstream and
therefore the only stable identity across the two sides.

WHAT THIS DOES NOT DO: statistics. Every SROCC/PLCC/CI in the Stage-A report
comes from the canonical owner (`zenstats` via `panel` / `scripts/lib/zen_stats.py`).
This script only reshapes bytes into rows.

HETEROGENEOUS ROWS ARE FIRST-CLASS. The score run changed metric sets mid-flight
(DOE plan §13.5: 3-metric before the 907-blob boundary at 2026-09-02T05:42Z,
2-metric after), so a row carries `metrics_present` and every metric column is
nullable. NEVER filter on a fixed metric set or a fixed row count.

`zensim` is emitted as a 720-wide FEATURE vector (`kind:"feature"`), not a
scalar — so `ssim2` is the only corpus-wide scalar quality response, exactly as
the plan's §7.1 designates it. `--with-features` keeps the vectors.

Usage:
  avifdoe_harvest.py --score-dir DIR --sizes sizes.tsv --pairs a1=P1.tsv ... --out out.parquet
"""
import argparse, collections, glob, json, os, sys

def sha_key(x):
    """Normalise an encode identity to its bare content address.

    The two sides disagree on spelling: `zenfleet-ctl pairs` writes a BARE sha in
    its `encode_sha` column (the full object URI is in `dist_path`), while the
    score blobs write the full `s3://.../blobs/<sha>` URI into their own
    `encode_sha` field. Keying on the basename joins them; keying on the raw
    string silently produces ZERO rows (measured 2026-09-02 — the join returned
    0 of 111,870 ndjson lines before this was added).
    """
    return x.rsplit("/", 1)[-1] if x else x

def parse_label(label):
    """'s4-svt-420-acb1-mtx32' -> (speed=4, chroma='420', devs=['acb1','mtx32'])."""
    if not label:
        return None, None, None
    parts = label.split("-")
    if len(parts) < 3 or not parts[0].startswith("s"):
        return None, None, None
    try:
        speed = int(parts[0][1:])
    except ValueError:
        return None, None, None
    return speed, parts[2], [p for p in parts[3:] if p]

# The naive preset x q sweeps (`avifsub-{svt,aom}-enc-*`) predate the DOE plan
# vocabulary and spell their knob tuple `{"backend":"svt-rs","speed":N}` -- no
# `cell`, no chroma. They are the DEFAULT-KNOB control at their own size, which
# is exactly what a Stage-B native wave needs to difference against, so they
# must be harvestable by the same tool rather than by a fork of it.
#
# The synthesized label asserts CHROMA, so it is backed by measurement, not by
# convention: on run `avifdoe-svt-b6-20260902` the in-run control `sN-svt-420`
# is byte-identical to `avifsub-svt-enc-20260901` speed N on 928/928 shared
# (image, q) cells at each of N = 4, 6, 7, and byte-DISTINCT from every other
# naive speed (2026-09-02, benchmarks/avif_doe_stageB6_analysis_2026-09-02.md
# section 3). If a future backend's naive sweep is not 4:2:0 by default, that
# identity check is what will catch it -- run it before trusting this path.
NAIVE_BACKEND_CHROMA = {"svt-rs": "svt-420", "aom-rs": "aom-420"}

def synth_label(kt, codec=None):
    """Naive-sweep knob tuple -> a DOE-vocabulary control label, or None."""
    sp, be = kt.get("speed"), kt.get("backend")
    if sp is not None and be in NAIVE_BACKEND_CHROMA:
        return f"s{int(sp)}-{NAIVE_BACKEND_CHROMA[be]}"
    return hdr_label(kt, codec)


# The HDR sweep lane (`sweep --hdr`, avif_hdr_arm_plan_2026-09-02.md §4.3)
# emits a knob tuple with the ONE wired dial and no `backend` / `cell` key:
# `{"preset":N}` for the zenav1-svt arm, `{"speed":N}` for the zenavif arm.
# The backend is the pairs table's `codec` column there. The label deliberately
# carries NO chroma token: the two HDR arms differ in chroma (svt 4:2:0 vs
# zenavif 4:4:4 GBR) but this function has no evidence of it in hand, and the
# existing chroma assertion above is backed by a byte-identity measurement.
# `parse_label` returns (None, None, None) for these, which is correct — a
# preset is not a speed — so the dial is carried by the explicit `dial` /
# `dial_kind` columns instead of being smuggled into the label grammar.
HDR_CODEC_TAG = {"zenav1-svt": "zenav1svt", "zenavif": "zenavif"}


def hdr_label(kt, codec):
    """HDR-lane knob tuple -> `<dial><N>-<codec>-hdr10`, or None."""
    tag = HDR_CODEC_TAG.get(codec)
    if tag is None or kt.get("backend") is not None or kt.get("cell") is not None:
        return None
    for key, pfx in (("preset", "p"), ("speed", "s")):
        if kt.get(key) is not None and len(kt) == 1:
            return f"{pfx}{int(kt[key])}-{tag}-hdr10"
    return None


def dial_of(kt):
    """(dial_value, dial_kind) for a single-dial HDR tuple, else (None, None)."""
    if kt.get("cell") is not None or kt.get("backend") is not None or len(kt) != 1:
        return (None, None)
    for key in ("preset", "speed"):
        if kt.get(key) is not None:
            return (int(kt[key]), key)
    return (None, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-dir", required=True, help="dir of score blobs (NDJSON, one per chunk)")
    ap.add_argument("--sizes", required=True, help="TSV: <encode_sha_uri>\\t<bytes>")
    ap.add_argument("--pairs", action="append", default=[], help="run=path.tsv (zenfleet-ctl pairs output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-features", action="store_true", help="retain the 720-wide zensim vectors")
    a = ap.parse_args()

    # --- cell rows, from the canonical `zenfleet-ctl pairs` tables ----------
    # ONE ROW PER CELL, not per bitstream. Encoding is content-addressed, so two
    # arms whose knobs make no difference to the output share an encode_sha —
    # 40,896 A1+A2 cells collapse to ~28k distinct blobs. Keying rows on
    # encode_sha would silently drop one of every such pair and destroy exactly
    # the "this knob is inert here" signal the DOE is measuring. So cells are the
    # row set and scores are attached to them, never the other way round.
    cells = []
    for spec in a.pairs:
        run, path = spec.split("=", 1)
        with open(path) as f:
            hdr = f.readline().rstrip("\n").split("\t")
            ix = {c: i for i, c in enumerate(hdr)}
            for line in f:
                r = line.rstrip("\n").split("\t")
                kt = json.loads(r[ix["knob_tuple_json"]])
                codec = r[ix["codec"]] if "codec" in ix else None
                lbl = kt.get("cell") or synth_label(kt, codec)
                speed, chroma, devs = parse_label(lbl)
                dial, dial_kind = dial_of(kt)
                esha = sha_key(r[ix["encode_sha"]])
                cells.append(dict(
                    run=run, image=r[ix["image_path"]], q=int(r[ix["q"]]),
                    arm=lbl, plan=kt.get("plan"), fp=kt.get("fp"),
                    codec=codec, dial=dial, dial_kind=dial_kind,
                    speed=speed, chroma=chroma,
                    devs="|".join(sorted(devs)) if devs is not None else None,
                    n_dev=len(devs) if devs is not None else None,
                    encode_sha=esha,
                ))
    print(f"cell rows from pairs tables: {len(cells)}", file=sys.stderr)
    print("  distinct encode_sha:", len({c["encode_sha"] for c in cells}), file=sys.stderr)

    # --- encoded bytes, from the object listing ------------------------------
    size = {}
    with open(a.sizes) as f:
        for line in f:
            k, v = line.rstrip("\n").split("\t")
            size[sha_key(k)] = int(v)
    print(f"encode sizes: {len(size)}", file=sys.stderr)

    # --- scores, keyed by bitstream ------------------------------------------
    sc = {}
    nblob = nline = 0
    for p in sorted(glob.glob(os.path.join(a.score_dir, "*"))):
        if not os.path.isfile(p):
            continue
        nblob += 1
        with open(p, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                nline += 1
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                raw = d.get("encode_sha")
                if not raw:
                    continue
                e = sha_key(raw)
                r = sc.setdefault(e, {"_m": set()})
                if d.get("kind") == "metric":
                    m = d.get("metric")
                    if m:
                        r["_m"].add(m); r[f"m_{m}"] = d.get("score")
                    for k, v in (d.get("scores") or {}).items():
                        r["_m"].add(k); r.setdefault(f"m_{k}", v)
                elif d.get("kind") == "feature":
                    r["_m"].add("zensim_features")
                    r["feature_regime"] = d.get("regime")
                    if a.with_features:
                        r["features"] = d.get("features")
    print(f"score blobs read: {nblob}  ndjson lines: {nline}  scored bitstreams: {len(sc)}", file=sys.stderr)

    metric_cols = sorted({k for r in sc.values() for k in r if k.startswith("m_")})
    out = []
    for c in cells:
        s_ = sc.get(c["encode_sha"])
        c["bytes"] = size.get(c["encode_sha"])
        c["metrics_present"] = "|".join(sorted(s_["_m"])) if s_ else None
        c["feature_regime"] = (s_ or {}).get("feature_regime")
        for k in metric_cols:
            c[k] = (s_ or {}).get(k)
        if a.with_features:
            c["features"] = (s_ or {}).get("features")
        out.append(c)

    import pyarrow as pa, pyarrow.parquet as pq
    tbl = pa.Table.from_pylist(out)
    pq.write_table(tbl, a.out, compression="zstd")
    print(f"wrote {a.out}: {tbl.num_rows} rows x {tbl.num_columns} cols", file=sys.stderr)
    print("metrics_present histogram:", file=sys.stderr)
    for k, v in collections.Counter(r["metrics_present"] for r in out).most_common():
        print(f"   {v:>7}  {k}", file=sys.stderr)
    print("rows per run:", dict(collections.Counter(r["run"] for r in out)), file=sys.stderr)
    print("rows missing bytes:", sum(1 for r in out if r["bytes"] is None), file=sys.stderr)
    print("rows UNSCORED (no ssim2):", sum(1 for r in out if r.get("m_ssim2") is None), file=sys.stderr)

if __name__ == "__main__":
    main()
