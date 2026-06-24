#!/usr/bin/env python3
# Munge a CPU-sweep omni.tsv (image_path, q, knob_tuple_json{cell}, encoded_bytes, score_ssim2,
# score_zensim) into the zentrain train_hybrid Pareto schema:
#   config_id, config_name, image_path, size_class, width, height, target_zq, bytes, ssim2, zensim
# config_id = sorted-unique index of config_name. size_class/width/height come from the re-extracted
# feature TSV (keyed on image basename). Output parquet.
#   usage: omni_to_pareto.py <omni.tsv> <features.tsv> <out.parquet>
import sys, csv, json, os
import pyarrow as pa, pyarrow.parquet as pq
csv.field_size_limit(1 << 24)
omni, feat_tsv, out = sys.argv[1], sys.argv[2], sys.argv[3]

meta = {}  # basename -> (size_class, width, height)
with open(feat_tsv) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        meta[os.path.basename(row["image_path"])] = (
            row["image_path"],  # the feature TSV's image_path — emit THIS so the Pareto join key matches
            row.get("size_class", "native"),
            int(float(row.get("width", 0) or 0)),
            int(float(row.get("height", 0) or 0)),
        )

raw = []
with open(omni) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        try:
            cell = json.loads(row["knob_tuple_json"])["cell"]
        except Exception:
            continue
        raw.append((cell, row["image_path"], int(float(row["q"])),
                    int(float(row["encoded_bytes"])), float(row["score_ssim2"]), float(row["score_zensim"])))

cfgs = sorted({r[0] for r in raw})
cfgid = {c: i for i, c in enumerate(cfgs)}
C = ("config_id", "config_name", "image_path", "size_class", "width", "height",
     "target_zq", "bytes", "ssim2", "zensim")
cols = {k: [] for k in C}
miss = 0
for cell, ip, q, b, s2, zs in raw:
    m = meta.get(os.path.basename(ip))
    if m is None:
        miss += 1
        m = (ip, "native", 0, 0)
    cols["config_id"].append(cfgid[cell]); cols["config_name"].append(cell)
    cols["image_path"].append(m[0]); cols["size_class"].append(m[1])
    cols["width"].append(m[2]); cols["height"].append(m[3])
    cols["target_zq"].append(q); cols["bytes"].append(b)
    cols["ssim2"].append(s2); cols["zensim"].append(zs)
pq.write_table(pa.table(cols), out, compression="zstd")
print("wrote %s : %d rows, %d configs, %d images, %d feature-misses"
      % (out, len(cols["config_id"]), len(cfgs), len(set(cols["image_path"])), miss), flush=True)
