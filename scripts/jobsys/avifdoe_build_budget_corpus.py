#!/usr/bin/env python3
"""Build the AVIF-DOE *budget* corpus: a 1024x1024 content-aware crop of every
subsample pick above the pixel budget, native passthrough below it.

Registered in `benchmarks/avif_doe_plan_2026-09-01.md` section 2.4. The pixel
budget is a free variable set by the permutation count the design wants, and
this script is what spends it.

WHY CROP AND NOT DOWNSCALE (plan doc 2.4, short version): the knobs under test
are largely high-frequency machinery — ac_bias is an RD bias toward
high-frequency error, variance boost keys on per-superblock variance, sharpness
biases deblocking, CDEF is a directional de-ringing filter, and SVT's
screen-content detector keys on crisp edges and flat palette-able regions. A
resampling kernel is a low-pass filter, i.e. it removes exactly the signal those
knobs act on, by a knob-specific amount. A native-resolution crop keeps grain,
sharpness and the noise floor intact.

WHY 1024 AND NOT 512: tiles are an axis in this design and AV1's minimum tile
width is 256px, so a 2x2 tiling needs >= 512px in both axes to be a measurement
rather than a silent geometry degradation — 1024 gives 2x headroom. 1024 is also
8x128 and 16x64, so the screening arm carries no partial-superblock cells at
either SVT superblock size, and it sits in the slope-dominated regime of the
cost model (at 0.25 MP an svt speed-7 encode is 13ms and fixed overhead leads).

CROP SELECTION is cluster-preserving, not center and not max-activity. A center
crop lands on flat background often enough to matter; a max-activity crop biases
the whole wave toward busy content and inflates every knob's measured effect.
Instead: the imazen26 feature parquet already carries per-position crop rows
(c50/c25 x {center,tl,tr,bl,br}) alongside the whole-image `full` row, so the
position whose feature vector is NEAREST the parent's — in the same z-scored
84-feature content space the k-means subsample used — is the position whose
content is most representative of the parent. The 1024 window is then centred on
that position's centre.

PASSTHROUGH RULE (plan doc 2.4, CORRECTED 2026-09-02): a source is left native
when `mp <= BUDGET_MP` OR either dimension is already <= `--side`. The second
clause is deliberate and is NOT "already at or under budget": cropping an image
whose WIDTH is already 1024 can only trim height, i.e. it discards content
without reducing scale, on a design whose stated purpose for the crop is to buy
permutations with pixels. The registered 23-cropped/9-native split assumed the
first clause alone; the rule as written yields 13 cropped / 19 native and 37.02
MP against the 33.55 MP the budget implies (+10.3%, 10 of 32 references). That
overshoot was AMENDED into the plan rather than rebuilt away -- see section 12.2
for the arithmetic and the reason (13,532 cells were already encoded against
these references).

Everything is recorded: source sha256, crop rect, crop sha256, the chosen
position and its feature distance, the parent's cluster id. Per the plan's G-CROP
gate the crop's OWN features must be re-extracted and its cluster assignment
checked against the parent's. Pass BOTH `--extractor` (the canonical zenanalyze
`extract_features_imazen26_crops` binary) and `--cluster-model` (the fitted
geometry from `imazen26_recluster_even.py --out-model`) to run that check;
without them the manifest carries `feature_recheck=PENDING` loudly rather than
silently inheriting the parent's features.

THE GATE VALIDATES ITSELF (added 2026-09-02). The 19 native references are
symlinks -- BIT-IDENTICAL to the parents that were clustered -- so re-extracting
them must reproduce the parent's cluster exactly and must land within
`--native-drift-tol` of the parent's stored feature vector in the clustering's
own z-space. Those 19 rows are therefore a control the extractor cannot fake: a
missing, no-op, wrong-schema or garbage extractor fails them and the gate exits
non-zero. This replaces the previous `--features-cmd` flag, which was documented
as satisfying G-CROP but was never executed (the script imported no
`subprocess`), so passing it flipped the manifest to a passing value without
extracting a single feature.

Usage:
  avifdoe_build_budget_corpus.py \
      --picks   /mnt/v/output/avifsvt-subsample-2026-09-01/avif_subsample_picks_2026-09-01.tsv \
      --parquet /mnt/v/output/imazen-26-features/imazen26_features_2026-06-13.parquet \
      --sources /mnt/v/output/avifsvt-subsample-2026-09-01/sources \
      --out     /mnt/v/output/avif-doe-1024-2026-09-01 \
      --extractor     ~/work/zen/zenanalyze/target/release/examples/extract_features_imazen26_crops \
      --cluster-model ~/tmp/doegaps/kmeans_K32_full_even_2026-09-02.npz
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

BUDGET_PX = 1024
BUDGET_MP = (BUDGET_PX * BUDGET_PX) / 1e6  # 1.048576

# Same exclusion list as scripts/imazen26_recluster_even.py — geometry features
# are not content, and cropping changes them by construction, so including them
# would make every crop look far from its parent for a trivial reason.
GEOM = ("pixel_count", "log_pixels", "bitmap_bytes", "min_dim", "max_dim",
        "aspect", "block_misalign", "log_padded", "channel_count")

POSITIONS = ("center", "tl", "tr", "bl", "br")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_feature_rows(parquet):
    """{(image_path, crop_label): np.ndarray} over the z-scored content features,
    plus the standardisation the whole native population defines."""
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet)
    feats = [n for n in pf.schema.names if n.startswith("feat_")]
    content = [n for n in feats if not any(k in n for k in GEOM)]
    t = pq.read_table(parquet,
                      columns=["image_path", "crop_label", "size_class"] + content)
    t = t.filter(pc.equal(t["size_class"], "native"))
    paths = t["image_path"].to_pylist()
    crops = t["crop_label"].to_pylist()
    M = np.empty((len(paths), len(content)), np.float64)
    for j, name in enumerate(content):
        col = t[name].to_numpy(zero_copy_only=False).astype(np.float64)
        med = np.nanmedian(col)
        col = np.nan_to_num(col, nan=med if np.isfinite(med) else 0.0)
        M[:, j] = col
    mu, sd = M.mean(0), M.std(0)
    sd[sd == 0] = 1.0
    Z = (M - mu) / sd
    return {(p, c): Z[i] for i, (p, c) in enumerate(zip(paths, crops))}, content


def crop_rect_for(position, w, h, side):
    """Top-left of a `side`x`side` window centred on `position`'s centre,
    clamped into the frame. Even coordinates: an odd x/y offset shifts the 4:2:0
    chroma phase relative to the parent, which would be a silent content change
    on top of the intended one."""
    cx, cy = {
        "center": (w / 2, h / 2),
        "tl": (w / 4, h / 4),
        "tr": (3 * w / 4, h / 4),
        "bl": (w / 4, 3 * h / 4),
        "br": (3 * w / 4, 3 * h / 4),
    }[position]
    x = int(max(0, min(w - side, round(cx - side / 2))))
    y = int(max(0, min(h - side, round(cy - side / 2))))
    return (x - (x & 1), y - (y & 1), side, side)


def stored_parent_vectors(parquet, names, source_paths):
    """The clustered population's OWN stored vector for each parent, in `names`
    order. This is the reference the native control is measured against, and it
    must come from the parquet the clustering was fit on -- not from a fresh
    extraction, or the control would compare a thing to itself."""
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(parquet,
                      columns=["image_path", "crop_label", "size_class"] + names)
    t = t.filter(pc.and_(pc.equal(t["size_class"], "native"),
                         pc.equal(t["crop_label"], "full")))
    idx = {p: i for i, p in enumerate(t["image_path"].to_pylist())}
    cols = {n: t[n].to_numpy(zero_copy_only=False).astype(np.float64) for n in names}
    out = {}
    for sp in source_paths:
        if sp in idx:
            out[sp] = np.array([cols[n][idx[sp]] for n in names], np.float64)
    return out


def run_extractor(extractor, out_dir, rows, workdir):
    """Run the CANONICAL zenanalyze extractor over the built corpus, once.

    `--sizes ""` + `--crop-fractions ""` make it emit exactly one row per input
    (`crop_label=full`, `size_class=native`) with no resize grid and no sub-crop
    variants. Non-zero rc, a missing output, or a short row count is fatal: the
    whole point of this gate is that it cannot be cleared without features.
    """
    os.makedirs(workdir, exist_ok=True)
    man = os.path.join(workdir, "extract_manifest.tsv")
    with open(man, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["path", "sha256", "split", "content_class", "source"])
        for r in rows:
            w.writerow([os.path.join(out_dir, "sources", r["corpus_key"]),
                        r["crop_sha256"], "budget", r["content_class"],
                        r["corpus_key"]])
    tsv = os.path.join(workdir, "budget_features.tsv")
    argv = [extractor, "--manifest", man, "--output", tsv,
            "--sizes", "", "--crop-fractions", ""]
    print("G-CROP: running " + " ".join(repr(a) for a in argv), file=sys.stderr)
    try:
        cp = subprocess.run(argv, capture_output=True, text=True)
    except OSError as e:
        sys.exit(f"*** G-CROP FAILED *** cannot execute --extractor {extractor!r}: {e}")
    if cp.returncode != 0:
        sys.stderr.write(cp.stdout or "")
        sys.stderr.write(cp.stderr or "")
        sys.exit(f"*** G-CROP FAILED *** extractor exited rc={cp.returncode}")
    if not os.path.exists(tsv):
        sys.exit(f"*** G-CROP FAILED *** extractor rc=0 but wrote no {tsv}")
    with open(tsv) as f:
        got = list(csv.DictReader(f, delimiter="\t"))
    if len(got) != len(rows):
        sys.exit(f"*** G-CROP FAILED *** extractor emitted {len(got)} rows for "
                 f"{len(rows)} references (a silently-dropped decode is not a pass)")
    return tsv, got, argv


def feature_vector(row, names):
    """Pull `names` out of an extraction row, accepting the bare `feat_<name>`
    header form and the qualified one. Missing/blank -> NaN (filled by the
    model's own medians upstream, exactly as the clustering did)."""
    import numpy as np
    v = np.empty(len(names))
    for j, n in enumerate(names):
        raw = row.get("feat_" + n)
        if raw is None:
            raw = row.get(n, "")
        v[j] = float(raw) if raw not in ("", None) else float("nan")
    return v


def check_clusters(model_npz, extracted, rows, parquet, native_tol):
    """Assign every built reference to its nearest k-means centroid IN THE
    CLUSTERING'S OWN Z-SPACE, and validate the native control.

    Returns {corpus_key: dict(...)}. Exits non-zero if any native reference
    fails -- see the module header for why that is the anti-no-op property.
    """
    import numpy as np
    m = np.load(model_npz, allow_pickle=True)
    names = [str(x) for x in m["content_names"]]
    fill, kept, mu, sd, C = m["fill"], m["kept_cols"], m["mu"], m["sd"], m["centroids"]

    by_key = {r["source"]: r for r in extracted}
    parents = stored_parent_vectors(parquet, names,
                                    [r["source_path"] for r in rows])

    def z_of(vec):
        vec = np.where(np.isfinite(vec), vec, fill)
        return (vec[kept] - mu) / sd

    out, native_fail = {}, []
    for r in rows:
        key = r["corpus_key"]
        if key not in by_key:
            sys.exit(f"*** G-CROP FAILED *** no extracted features for {key}")
        z = z_of(feature_vector(by_key[key], names))
        d = np.linalg.norm(C - z, axis=1)
        assigned = int(d.argmin())
        srt = np.sort(d)
        parent_v = parents.get(r["source_path"])
        pz = (float(np.linalg.norm(z - z_of(parent_v)))
              if parent_v is not None else float("nan"))
        preserved = assigned == int(r["parent_cluster"])
        out[key] = dict(assigned_cluster=assigned,
                        cluster_dist=round(float(srt[0]), 4),
                        cluster_margin=round(float(srt[1] - srt[0]), 4),
                        parent_z_dist=("" if parent_v is None else round(pz, 4)),
                        preserved=preserved)
        if r["transform"] == "native":
            # A native reference IS the parent, byte for byte. Anything but an
            # exact cluster match, or a z-displacement beyond extractor drift,
            # means the features are not what they claim to be.
            if not preserved:
                native_fail.append(f"{key}: cluster {r['parent_cluster']} -> {assigned}")
            elif parent_v is None:
                native_fail.append(f"{key}: parent not found in {parquet}")
            elif not (pz <= native_tol):
                native_fail.append(f"{key}: z-displacement {pz:.4f} > {native_tol}")

    if native_fail:
        print("\n*** G-CROP FAILED — NATIVE CONTROL BROKEN ***\n"
              "The native references are symlinks to the clustered parents, so "
              "re-extracting them MUST reproduce the parents' own vectors and "
              "clusters. These did not, which means the extraction is not "
              "measuring what it claims:", file=sys.stderr)
        for f in native_fail:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", type=int, default=BUDGET_PX)
    ap.add_argument("--extractor", default=None,
                    help="path to the CANONICAL zenanalyze extractor binary "
                         "(examples/extract_features_imazen26_crops, built with "
                         "--features experimental,hdr and WITHOUT `api` so the "
                         "headers are the bare feat_<name> form the population "
                         "parquet uses). Required, with --cluster-model, to "
                         "satisfy G-CROP; it is RUN, and a non-zero exit, a "
                         "missing output or a short row count is fatal.")
    ap.add_argument("--cluster-model", default=None,
                    help="the fitted clustering geometry (.npz) from "
                         "`imazen26_recluster_even.py --out-model` — feature "
                         "names, NaN-fill medians, kept-column mask, mu/sd and "
                         "the k-means centroids. Assignment happens in THAT "
                         "space; re-standardising here would answer a different "
                         "question.")
    ap.add_argument("--native-drift-tol", type=float, default=0.5,
                    help="max z-space displacement allowed between a NATIVE "
                         "reference (a symlink, so bit-identical to its parent) "
                         "and the parent's stored vector. Default 0.5 is ~20x "
                         "the measured 2026-09-02 extractor drift (max 0.0253 "
                         "over the 32 picks), so it passes real drift and fails "
                         "a wrong/no-op extractor.")
    args = ap.parse_args()
    if bool(args.extractor) != bool(args.cluster_model):
        ap.error("--extractor and --cluster-model must be given together: "
                 "features without the fitted geometry cannot be checked, and "
                 "the geometry without features has nothing to check.")

    import numpy as np
    from PIL import Image

    os.makedirs(os.path.join(args.out, "sources"), exist_ok=True)
    Z, content_names = load_feature_rows(args.parquet)

    picks = list(csv.DictReader(open(args.picks), delimiter="\t"))
    rows = []
    for p in picks:
        key = p["corpus_key"]
        src = os.path.join(args.sources, key)
        if not os.path.exists(src):
            sys.exit(f"missing corpus file: {src}")
        w, h = int(p["final_width"]), int(p["final_height"])
        mp = w * h / 1e6
        dst = os.path.join(args.out, "sources", key)

        if mp <= BUDGET_MP or w <= args.side or h <= args.side:
            # NEVER upscale; leave native anything already under the pixel
            # budget OR already at/below `--side` on either axis. The second
            # clause is why the split is 13/19 and not the registered 23/9:
            # cropping a 1024x1536 pick can only trim height, discarding
            # content without reducing scale. See the module header and plan
            # section 12.2 — AMENDED, not a bug to rebuild away.
            # Symlink so no pixel is copied and the native corpus stays the
            # single source of truth.
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
            rows.append(dict(
                origin_id=p["origin_id"], corpus_key=key, transform="native",
                source_path=p["source_path"],
                crop_rect="", position="", feat_distance="",
                parent_cluster=p["cluster_id"], content_class=p["content_class"],
                width=w, height=h, source_sha256=sha256(src),
                crop_sha256=sha256(src), feature_recheck="n/a-native",
                below_tile_floor=str(min(w, h) < 512).lower()))
            continue

        # Cluster-preserving position: the crop row nearest the `full` row.
        parent = Z.get((p["source_path"], "full"))
        best, best_d = "center", None
        if parent is not None:
            for pos in POSITIONS:
                for label in (f"c50_{pos}", f"c25_{pos}", pos):
                    v = Z.get((p["source_path"], label))
                    if v is None:
                        continue
                    d = float(np.linalg.norm(v - parent))
                    if best_d is None or d < best_d:
                        best, best_d = pos, d
                    break
        else:
            print(f"WARN {key}: no parent feature row; falling back to center",
                  file=sys.stderr)

        x, y, cw, ch = crop_rect_for(best, w, h, args.side)
        with Image.open(src) as im:
            im.convert("RGB").crop((x, y, x + cw, y + ch)).save(dst, "PNG",
                                                                optimize=False)
        rows.append(dict(
            origin_id=p["origin_id"], corpus_key=key, transform="crop-native",
            source_path=p["source_path"],
            crop_rect=f"{x},{y},{cw},{ch}", position=best,
            feat_distance=("" if best_d is None else f"{best_d:.4f}"),
            parent_cluster=p["cluster_id"], content_class=p["content_class"],
            width=cw, height=ch, source_sha256=sha256(src),
            crop_sha256=sha256(dst),
            feature_recheck="PENDING",
            below_tile_floor="false"))

    # The corpus key is UNCHANGED (`<id>.scale<W>x<H>.png`) even for crops, so
    # `avifsvt_cells.py`'s filename regex and every existing consumer still
    # parse; the transform lives in the manifest, which is the thing that must
    # be read to know what a reference is.
    man = os.path.join(args.out, "crop_manifest_2026-09-01.tsv")

    n_crop = sum(1 for r in rows if r["transform"] == "crop-native")
    total_mp = sum(int(r["width"]) * int(r["height"]) for r in rows) / 1e6
    budget_mp = len(rows) * BUDGET_MP
    print(f"{len(rows)} references -> {args.out}/sources  "
          f"({n_crop} cropped, {len(rows) - n_crop} native)")
    print(f"pixels: {total_mp:.2f} MP vs {budget_mp:.2f} MP at the budget "
          f"({100 * (total_mp / budget_mp - 1):+.1f}%) — see plan section 12.2")

    # ---- G-CROP ----------------------------------------------------------
    prov = None
    if args.extractor:
        tsv, extracted, argv = run_extractor(args.extractor, args.out, rows,
                                             os.path.join(args.out, "_gcrop"))
        feats_sha = sha256(tsv)
        verdicts = check_clusters(args.cluster_model, extracted, rows,
                                  args.parquet, args.native_drift_tol)
        shifted = []
        for r in rows:
            v = verdicts[r["corpus_key"]]
            r.update(assigned_cluster=v["assigned_cluster"],
                     cluster_dist=v["cluster_dist"],
                     cluster_margin=v["cluster_margin"],
                     parent_z_dist=v["parent_z_dist"],
                     features_sha256=feats_sha)
            if r["transform"] == "native":
                r["feature_recheck"] = "n/a-native"
            elif v["preserved"]:
                r["feature_recheck"] = "preserved"
            else:
                r["feature_recheck"] = (
                    f"SHIFTED:{r['parent_cluster']}->{v['assigned_cluster']}")
                shifted.append(r)
        prov = dict(features_tsv=tsv, features_sha256=feats_sha,
                    extractor=args.extractor, extractor_argv=argv,
                    cluster_model=args.cluster_model,
                    cluster_model_sha256=sha256(args.cluster_model),
                    population_parquet=args.parquet,
                    native_drift_tol=args.native_drift_tol,
                    n_references=len(rows), n_cropped=n_crop,
                    n_native=len(rows) - n_crop,
                    total_megapixels=round(total_mp, 3),
                    n_cluster_shifts=len(shifted),
                    cluster_shifts=[r["corpus_key"] for r in shifted])
        with open(os.path.join(args.out, "gcrop_provenance.json"), "w") as f:
            json.dump(prov, f, indent=2, sort_keys=True)
        print(f"G-CROP: features {feats_sha[:16]}… over {len(extracted)} "
              f"references; native control PASSED "
              f"({len(rows) - n_crop}/{len(rows) - n_crop})")
        if shifted:
            # NOT a failure: section 2.4 asks for the shift to be RECORDED per
            # pick, not forbidden. A crop is genuinely different content.
            print(f"G-CROP: {len(shifted)} of {n_crop} crops changed cluster "
                  f"(recorded, not fatal):", file=sys.stderr)
            for r in shifted:
                print(f"  {r['corpus_key']}  cluster {r['parent_cluster']} -> "
                      f"{r['assigned_cluster']}  (parent_z_dist "
                      f"{r['parent_z_dist']}, margin {r['cluster_margin']})",
                      file=sys.stderr)

    # Append-only column order: the original 14 keep their positions so any
    # positional reader still parses; everything new lands at the tail.
    base = ["origin_id", "corpus_key", "transform", "crop_rect", "position",
            "feat_distance", "parent_cluster", "content_class", "width",
            "height", "source_sha256", "crop_sha256", "feature_recheck",
            "below_tile_floor"]
    extra = [k for k in rows[0] if k not in base]
    with open(man, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=base + extra, delimiter="\t",
                             lineterminator="\n")
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"manifest: {man}")

    if any(r["feature_recheck"] == "PENDING" for r in rows):
        print("\n*** G-CROP NOT SATISFIED ***\n"
              "The crops' OWN features were not re-extracted, so the "
              "cluster-assignment check the plan requires has not run. The "
              "tuning model must condition on features of what was actually "
              "encoded, never the parent's. Re-run with --extractor and "
              "--cluster-model (see the module header) before publishing any "
              "A1/A2/A3 conclusion.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
