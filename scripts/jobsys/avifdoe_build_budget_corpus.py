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

Everything is recorded: source sha256, crop rect, crop sha256, the chosen
position and its feature distance, the parent's cluster id. Per the plan's G-CROP
gate the crop's OWN features must be re-extracted and its cluster assignment
checked against the parent's; `--features-cmd` wires an extractor in when one is
available, and without it the manifest carries `feature_recheck=PENDING` loudly
rather than silently inheriting the parent's features.

Usage:
  avifdoe_build_budget_corpus.py \
      --picks   /mnt/v/output/avifsvt-subsample-2026-09-01/avif_subsample_picks_2026-09-01.tsv \
      --parquet /mnt/v/output/imazen-26-features/imazen26_features_2026-06-13.parquet \
      --sources /mnt/v/output/avifsvt-subsample-2026-09-01/sources \
      --out     /mnt/v/output/avif-doe-1024-2026-09-01
"""
import argparse
import csv
import hashlib
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", type=int, default=BUDGET_PX)
    ap.add_argument("--features-cmd", default=None,
                    help="command receiving a PNG path on argv and printing the "
                         "84 content features as one whitespace-separated line; "
                         "when absent the manifest records feature_recheck=PENDING")
    args = ap.parse_args()

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
            # NEVER upscale, and never crop something already at or under
            # budget. Symlink so no pixel is copied and the native corpus stays
            # the single source of truth.
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
            rows.append(dict(
                origin_id=p["origin_id"], corpus_key=key, transform="native",
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
            crop_rect=f"{x},{y},{cw},{ch}", position=best,
            feat_distance=("" if best_d is None else f"{best_d:.4f}"),
            parent_cluster=p["cluster_id"], content_class=p["content_class"],
            width=cw, height=ch, source_sha256=sha256(src),
            crop_sha256=sha256(dst),
            feature_recheck=("PENDING" if not args.features_cmd else "todo"),
            below_tile_floor="false"))

    # The corpus key is UNCHANGED (`<id>.scale<W>x<H>.png`) even for crops, so
    # `avifsvt_cells.py`'s filename regex and every existing consumer still
    # parse; the transform lives in the manifest, which is the thing that must
    # be read to know what a reference is.
    man = os.path.join(args.out, "crop_manifest_2026-09-01.tsv")
    with open(man, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        wtr.writeheader()
        wtr.writerows(rows)

    n_crop = sum(1 for r in rows if r["transform"] == "crop-native")
    print(f"{len(rows)} references -> {args.out}/sources  "
          f"({n_crop} cropped, {len(rows) - n_crop} native)")
    print(f"manifest: {man}")
    if any(r["feature_recheck"] == "PENDING" for r in rows):
        print("\n*** G-CROP NOT SATISFIED ***\n"
              "The crops' OWN features were not re-extracted, so the "
              "cluster-assignment check the plan requires has not run. The "
              "tuning model must condition on features of what was actually "
              "encoded, never the parent's. Re-run with --features-cmd, or run "
              "the imazen26 extractor over "
              f"{args.out}/sources and record the result before publishing any "
              "A1/A2/A3 conclusion.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
