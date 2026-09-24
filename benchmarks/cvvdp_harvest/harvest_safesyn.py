#!/usr/bin/env python3
"""cvvdp-safesyn harvest: join a ScoreFile run's JSONL blobs back to the
SafeSyn cache row keys, emit the sidecar parquet + _MANIFEST.json the brief
names. Keys on the pairs TSV's `row_id` (== the cache parquet's row_id), via
encode_sha (full dist s3 URI) -> dist_path.

  harvest_safesyn.py --run <run> --bucket zentrain \
      --out /var/tmp/cvvdp-safesyn/safesyn_cvvdp_sidecar.parquet

Completeness gates — every one is a loud failure, no partial harvest:
  * exactly 196,086 distinct pairs, row_id unique, every metric non-null;
  * every emitted row's reference/distorted_pixels_sha256 equals the
    Sept-14 safesyn-train audit hash (the fleet-review r2 condition: the
    ZEN_JOBEXEC_AVIF_DECODE opt-in does not enter the JobId, so the stamps
    are the per-row proof the worker ran the bound decode route);
  * every row carries the pixel-hash stamps (a worker built without
    ZEN_JOBEXEC_PIXEL_HASH emits none and fails here).
Codec family is derived from the encode_sha path — NOT the row's `codec`
label (upstream labels each per-ref job with its first pair's codec).

Blobs are parsed per file, line by line; raw job stdout is never
concatenated (jobexec emits no trailing newline on the last row).
"""
import argparse, csv, json, math, os, struct, subprocess, sys
from collections import defaultdict

PAIRS = "/var/tmp/zensim-validation-2026-09-14/baseline-recovery/safesyn-train-pairs.tsv"
S3_PREFIX = "s3://codec-corpus/safesyn-rev2-2026-09-06/"
IMG_PREFIX = "/mnt/v/input/zensim/images/"
AUDIT = "/var/tmp/zensim-validation-2026-09-14/baseline-recovery/safesyn-train-audit.jsonl"
AUDIT_FOR_PROVENANCE = AUDIT
# metric string -> output column (brief: cvvdp_jod_<display>)
METRIC_COL = {
    "cvvdp@standard_fhd": "cvvdp_jod_standard_fhd",
    "cvvdp@sdr_fhd_24": "cvvdp_jod_sdr_fhd_24",
    "cvvdp": "cvvdp_jod_standard_4k",
    "ssim2": "ssim2_fresh",
}


def load_pair_keys():
    """dist s3 URI -> (row_id, ref_basename, source_path-ish)."""
    m = {}
    with open(PAIRS) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            uri = S3_PREFIX + "images/" + "/".join(r["dist_path"].split("/")[-3:])
            m[uri] = (int(r["row_id"]), r["ref_path"].rsplit("/", 1)[-1],
                      r["dist_path"])
    return m


def load_audit_hashes():
    """dist key (images/<ref>/<family>/<file>) -> audit row; also ref_basename
    -> reference_pixels_sha256 for the per-ref check."""
    by_dist, by_ref = {}, {}
    with open(AUDIT) as f:
        for line in f:
            a = json.loads(line)
            dkey = a["distorted"].replace(IMG_PREFIX, "images/")
            by_dist[dkey] = a
            by_ref[a["ref_basename"]] = a["reference_pixels_sha256"]
    return by_dist, by_ref


def fam_of_uri(uri):
    """codec family from the encode_sha path: .../images/<ref>/<family>/<file>."""
    parts = uri.split("/")
    return parts[-2] if len(parts) >= 2 else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bucket", default="zentrain")
    ap.add_argument("--endpoint", default=os.environ.get("ZEN_S3_ENDPOINT", ""))
    ap.add_argument("--blob-dir", default="/var/tmp/cvvdp-safesyn/harvest_blobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-out", default=None)
    ap.add_argument("--build-meta", default=None,
                    help="path to build_meta.json written by "
                         "path_dep_provenance.py at image build")
    a = ap.parse_args()

    pair = load_pair_keys()
    by_dist, _by_ref = load_audit_hashes()
    print(f"pairs index: {len(pair)} dist uris; audit: {len(by_dist)} dist keys",
          file=sys.stderr)

    # 1) ledger -> DONE output shas
    os.makedirs(a.blob_dir, exist_ok=True)
    env = dict(os.environ)
    r = subprocess.run(
        ["s5cmd", "--endpoint-url", a.endpoint, "cp",
         f"s3://{a.bucket}/jobs/{a.run}/blobs/*", a.blob_dir + "/"],
        env=env, capture_output=True, text=True)
    if r.returncode != 0:
        # empty prefix (run not started) or a real fetch error — the
        # completeness gate below still fails loudly either way
        print(f"WARN blob fetch rc={r.returncode}: {r.stderr.strip()[:200]}",
              file=sys.stderr)
    blobs = sorted(os.path.join(a.blob_dir, f) for f in os.listdir(a.blob_dir)
                   if os.path.isfile(os.path.join(a.blob_dir, f)))
    print(f"blobs: {len(blobs)}", file=sys.stderr)

    # 2) parse JSONL rows; collect per-(ref,dist) metric scores. Every row is
    # hash-checked against the Sept-14 audit — the env-baked decode route does
    # not enter the JobId, so these stamps are the per-row proof of route.
    cells = defaultdict(dict)   # dist_uri -> {metric: score}
    nrows = nerr = 0
    px_missing = px_bad = dup_bad = duplicate_rows = unknown_metric_rows = 0
    for b in blobs:
        with open(b) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                nrows += 1
                r = json.loads(line)
                if "error" in r:
                    nerr += 1
                    continue
                uri = r["encode_sha"]
                dkey = "images/" + "/".join(uri.split("/")[-3:])
                au = by_dist.get(dkey)
                rph, dph = (r.get("reference_pixels_sha256"),
                            r.get("distorted_pixels_sha256"))
                if rph is None or dph is None:
                    px_missing += 1
                elif au is None:
                    px_bad += 1
                    print(f"NO AUDIT KEY {dkey}", file=sys.stderr)
                elif (rph != au["reference_pixels_sha256"]
                      or dph != au["distorted_pixels_sha256"]):
                    px_bad += 1
                    if px_bad <= 10:
                        print(f"PX MISMATCH {dkey} metric={r.get('metric')}",
                              file=sys.stderr)
                col = METRIC_COL.get(r.get("metric"))
                if col is None or r.get("score") is None:
                    unknown_metric_rows += 1
                    continue
                v = float(r["score"])
                if not math.isfinite(v):
                    raise ValueError(f"non-finite score: {dkey} {col}")
                prev = cells[uri].get(col)
                if prev is not None:
                    duplicate_rows += 1
                    if struct.pack("!d", prev) != struct.pack("!d", v):
                        # lease-retry duplicate rows must agree bit for bit —
                        # a disagreement means non-determinism, fail loudly
                        # (GO condition 3).
                        print(f"DUPLICATE DISAGREE {dkey} {col}: "
                              f"{prev} vs {v}", file=sys.stderr)
                        dup_bad += 1
                cells[uri][col] = v
    print(f"rows={nrows} err_rows={nerr} cells={len(cells)} "
          f"px_missing={px_missing} px_bad={px_bad} dup_bad={dup_bad} "
          f"duplicate_rows={duplicate_rows} unknown_metric_rows={unknown_metric_rows}",
          file=sys.stderr)
    assert nerr == 0, f"{nerr} error rows"
    assert px_missing == 0, f"{px_missing} rows lack pixel-hash stamps"
    assert px_bad == 0, f"{px_bad} rows fail the Sept-14 audit pixel hash"
    assert dup_bad == 0, f"{dup_bad} duplicate rows disagree"
    assert unknown_metric_rows == 0, f"{unknown_metric_rows} unknown or unscored rows"

    # 3) emit parquet keyed on row_id
    import pyarrow as pa, pyarrow.parquet as pq
    cols = list(METRIC_COL.values())
    out_rows, missing = [], []
    for uri, scores in cells.items():
        key = pair.get(uri)
        if key is None:
            missing.append(uri); continue
        row_id, refbase, dist = key
        out_rows.append((row_id, refbase, dist, fam_of_uri(uri),
                        *[scores.get(c) for c in cols]))
    out_rows.sort()
    print(f"mapped={len(out_rows)} unmapped={len(missing)}", file=sys.stderr)
    if missing[:5]:
        print("unmapped sample:", missing[:5], file=sys.stderr)
    assert not missing, f"{len(missing)} rows cannot map to a SafeSyn row_id"

    t = pa.table({
        "row_id": pa.array([r[0] for r in out_rows], type=pa.int64()),
        "ref_basename": pa.array([r[1] for r in out_rows]),
        "source_path": pa.array([r[2] for r in out_rows]),
        # family from the encode_sha path, NOT the ledger row's `codec` label
        # (upstream labels every row with the job's first-pair codec).
        "codec_family": pa.array([r[3] for r in out_rows]),
        **{c: pa.array([r[4 + i] for r in out_rows], type=pa.float64())
           for i, c in enumerate(cols)},
    })
    # completeness gate
    assert t.num_rows == 196086, f"expected 196086 rows, got {t.num_rows}"
    ids = t.column("row_id").to_pylist()
    assert len(set(ids)) == len(ids), "duplicate row_id"
    for c in cols:
        assert t.column(c).null_count == 0, f"nulls in {c}"
    pq.write_table(t, a.out)
    print(f"wrote {a.out}", file=sys.stderr)

    if a.manifest_out:
        import hashlib
        def sha256_file(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        # Build-time provenance captured by path_dep_provenance.py at image
        # build (fleet-review r2 item 3: the musl binary links sibling PATH
        # crates at working-copy state, so the zenmetrics commit alone does
        # not reproduce it — record each sibling's HEAD + dirty-diff hash).
        build_meta = {}
        for cand in (a.build_meta, "/var/tmp/cvvdp-safesyn/build_meta.json"):
            if cand and os.path.exists(cand):
                build_meta = json.load(open(cand))
                break
        assert build_meta.get("build", {}).get("zenmetrics_commit") == (
            "9f36f88b8a23e645bd791c1193842d950eccf4be"
        ), "frozen zenmetrics build commit missing or wrong"
        assert len(build_meta.get("path_dep_repos", {})) == 13, (
            "incomplete sibling build provenance"
        )
        man = {
            "lane": "cvvdp-safesyn",
            "brief": "CVVDP_SAFESYN_brief.md Part 2 + CVVDP_SAFESYN_FLEET_addendum.md",
            "created_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
            "run": a.run, "bucket": a.bucket,
            "build_commit": build_meta["build"]["zenmetrics_commit"],
            "ledger_prefix": f"s3://{a.bucket}/jobs/{a.run}/",
            "rows": t.num_rows, "blob_count": len(blobs),
            "error_rows": nerr,
            "duplicate_rows": duplicate_rows,
            "sidecar_sha256": sha256_file(a.out),
            "metrics": METRIC_COL,
            "display_models": {
                "cvvdp_jod_standard_4k": "standard_4k (default; 75.4 px/deg — the historical teacher geometry)",
                "cvvdp_jod_standard_fhd": "standard_fhd (37.84 px/deg, 200 nits SDR)",
                "cvvdp_jod_sdr_fhd_24": "sdr_fhd_24 (37.84 px/deg, 100 nits SDR — E2b-selected)",
                "ssim2_fresh": "fast-ssim2, same decoded buffers (binding control)",
            },
            "build": build_meta.get("build", "MISSING build_meta.json"),
            "path_dep_provenance": build_meta.get("path_deps", "MISSING"),
            "path_dep_repos": build_meta.get("path_dep_repos", "MISSING"),
            # GO condition 4: the linked zensim source commit comes from
            # build_meta.json, NOT the image's ZEN_ZENSIM_BUILD_COMMIT env
            # (which holds a lane doc commit and only feeds `feature`-job rows).
            "zensim_linked_commit": build_meta.get("zensim_linked_commit"),
            "decode_route_env": build_meta.get(
                "decode_route_env",
                {"ZEN_JOBEXEC_AVIF_DECODE": "zencodec",
                 "ZEN_JOBEXEC_PIXEL_HASH": "1"}),
            "decoder_identities": build_meta.get(
                "decoder_identities", "MISSING"),
            "inputs": {
                "pairs_tsv": PAIRS,
                "pairs_tsv_sha256": sha256_file(PAIRS),
                "pairs_parquet": "/var/tmp/cvvdp-safesyn/safesyn_pairs_uri.parquet",
                "blob_prefix": S3_PREFIX,
                "audit_jsonl": AUDIT_FOR_PROVENANCE,
            },
            "completeness": {
                "expected_rows": 196086,
                "row_id_unique": True,
                "metric_columns_nonnull": list(METRIC_COL.values()),
                "pixel_hash_rows_checked": nrows - nerr,
                "pixel_hash_missing_stamps": px_missing,
                "pixel_hash_audit_mismatches": px_bad,
                "duplicate_disagreements": dup_bad,
                "unknown_metric_rows": unknown_metric_rows,
            },
        }
        json.dump(man, open(a.manifest_out, "w"), indent=1)
        print(f"wrote {a.manifest_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
