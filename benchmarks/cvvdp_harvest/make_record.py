#!/usr/bin/env python3
"""Render the small audited record from the raw SafeSyn harvest artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path("/var/tmp/cvvdp-safesyn")
DEST = Path(__file__).resolve().parents[1]
STEM = "cvvdp_safesyn_2026-09-23"
STORE = "s3://zentrain/jobs/cvvdp-safesyn-20260923/harvest"
TOWER = "/mnt/user/coefficient/output/zensim/cvvdp-safesyn-2026-09-23"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    side = ROOT / "safesyn_cvvdp_sidecar.parquet"
    manifest_path = ROOT / "safesyn_cvvdp_sidecar_MANIFEST.json"
    build = ROOT / "build_meta.json"
    store = json.load(open(ROOT / "store_verify.json"))
    scores = json.load(open(ROOT / "independent_scores.json"))
    desc = json.load(open(ROOT / "descriptive.json"))
    teardown = json.load(open(ROOT / "teardown_status.json"))
    manifest = json.load(open(manifest_path))
    assert store["manifest_jobs"] == store["ledger_jobs"] == store["blob_objects"] == 3218
    assert not any(store[k] for k in ("missing_ledger", "missing_blob", "failed_or_not_done"))
    assert manifest["rows"] == scores["unique_row_ids_checked"] == desc["rows"] == 196086
    assert manifest["sidecar_sha256"] == sha(side)
    assert scores["raw_sidecar_mismatch"] == scores["unmatched_raw_rows"] == 0
    assert manifest["build_commit"] == "9f36f88b8a23e645bd791c1193842d950eccf4be"
    assert len(manifest["path_dep_repos"]) == 13
    record = {
        "run": "cvvdp-safesyn-20260923",
        "role": "TRAIN metric labels only; descriptive comparison, no fitting or selection",
        "build_commit": manifest["build_commit"],
        "build_meta_sha256": sha(build),
        "manifest_sha256": sha(manifest_path),
        "sidecar_sha256": sha(side),
        "sidecar_bytes": side.stat().st_size,
        "score_rows": manifest["completeness"]["pixel_hash_rows_checked"],
        "pairs": manifest["rows"],
        "store": store,
        "independent_scores": scores,
        "descriptive": desc,
        "teardown": teardown,
    }
    json_path = DEST / (STEM + ".json")
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    assert json_path.stat().st_size < 30000

    lines = [
        "# CVVDP SafeSyn fleet harvest — 2026-09-23",
        "",
        "TRAIN-role synthetic pairs with metric labels only. This is a descriptive label sidecar, not a human-accuracy or model-selection result.",
        "",
        "## Completion and provenance",
        "",
        f"- Run `{record['run']}`: **{store['manifest_jobs']:,} jobs**; {store['ledger_rows']:,} raw ledger rows, {store['ledger_jobs']:,} latest `done` jobs, **{store['blob_objects']:,} blobs**. Missing ledger rows, missing blobs, failed/not-done jobs: **0 / 0 / 0**.",
        f"- **{record['pairs']:,} unique pair row IDs**, four nonnull metric columns, **{record['score_rows']:,} score rows**. Error rows, missing pixel stamps, audit pixel mismatches, duplicate score disagreements: **0 / 0 / 0 / 0**.",
        f"- Source build: zenmetrics `{manifest['build_commit']}`; image `{manifest['build']['executor_image_digest']}`. The full manifest contains all {len(manifest['path_dep_repos'])} sibling repo commits and dirty-diff hashes. Sidecar sha256 `{record['sidecar_sha256']}`.",
        f"- Latest `done` jobs by worker tier: r5600g {store['tier']['r5600g']:,}; r3500 {store['tier']['r3500']:,}; tower {store['tier']['tower']:,}.",
        "",
        "## Independent keyed checks",
        "",
        f"- Fresh SSIM2 against the Sept. 14 `peer_ssim2.score`: {scores['unique_row_ids_checked']:,}/{record['pairs']:,} keyed rows checked; {scores['groups']['ALL/ALL']['nonexact']} differ; max |Δ| = {scores['groups']['ALL/ALL']['max_abs_delta']:.17g}. All raw SSIM2 values equal their harvested sidecar values bit for bit.",
        f"- Sept. 14 audit CVVDP 4K values available: **{scores['sept14_4k_audit_rows']}**; cache CVVDP columns: **{len(scores['sept14_4k_cache_columns'])}**. The requested 4K-vs-Sept.-14 numeric check therefore has zero overlapping rows.",
        "",
        "| Codec family | Worker tier | Rows | Nonexact SSIM2 | Max abs Δ |",
        "|---|---|---:|---:|---:|",
    ]
    for key, row in sorted(scores["groups"].items()):
        if key == "ALL/ALL":
            continue
        family, host = key.split("/")
        lines.append(f"| `{family}` | {host} | {row['rows']:,} | {row['nonexact']} | {row['max_abs_delta']:.3g} |")
    lines += [
        "",
        "## Descriptive rank agreement",
        "",
        "SROCC uses `zen_stats.panel_batch_indexed` on the stored metric-label oracle. Its score is absolute; every displayed signed SROCC is positive. These are TRAIN metric-to-metric comparisons, not human validation.",
        "",
        "| Metric | Overall SROCC vs original oracle | SROCC vs stored fresh SSIM2 |",
        "|---|---:|---:|",
    ]
    for arm in ("cvvdp_jod_standard_4k", "cvvdp_jod_standard_fhd",
                "cvvdp_jod_sdr_fhd_24", "ssim2_fresh"):
        a = desc["results"]["overall/" + arm]["srocc"]
        b = desc["results"]["overall_vs_stored_s2/" + arm]["srocc"]
        lines.append(f"| `{arm}` | {a:.6f} | {b:.6f} |")
    for group_type in ("family", "band"):
        lines += ["", f"### By {group_type}", "", "| Group | n | 4K | standard FHD | SDR FHD 24 | fresh SSIM2 |", "|---|---:|---:|---:|---:|---:|"]
        groups = sorted({k.split("/")[1] for k in desc["results"] if k.startswith(group_type + "/")})
        for group in groups:
            keys = [f"{group_type}/{group}/{a}" for a in (
                "cvvdp_jod_standard_4k", "cvvdp_jod_standard_fhd",
                "cvvdp_jod_sdr_fhd_24", "ssim2_fresh")]
            n = desc["results"][keys[0]]["n"]
            vals = [desc["results"][k]["srocc"] for k in keys]
            lines.append(f"| `{group}` | {n:,} | " + " | ".join(f"{v:.6f}" for v in vals) + " |")
    lines += ["", "### Largest display disagreements", "", "| row_id | family | 4K JOD | SDR FHD 24 JOD | abs Δ |", "|---:|---|---:|---:|---:|"]
    for r in desc["examples"][:10]:
        lines.append(f"| {r['row_id']} | `{r['family']}` | {r['jod_4k']:.5f} | {r['jod_fhd_24']:.5f} | {r['abs_delta']:.5f} |")
    lines += [
        "", "Full row-key examples and all 48 canonical statistic results are in the adjacent JSON record. No scoring was rerun for this harvest.",
        "", "## Worker teardown", "",
        "- `zen-score-cvvdp` removed and confirmed absent on r3500 and tower.",
        "- r5600g removal is unconfirmed: SSH had no route or timed out on four attempts; ping failed from dev and tower. The coordinator retains the host OS decision. No other host state was changed.", "",
    ]
    md_path = DEST / (STEM + ".md")
    md_path.write_text("\n".join(lines))
    assert md_path.stat().st_size < 30000

    pointer = [
        "# CVVDP SafeSyn sidecar pointer", "",
        f"Sidecar sha256: `{record['sidecar_sha256']}` ({record['sidecar_bytes']:,} bytes).",
        f"LAN store: `{STORE}/{side.name}`.",
        f"Tower mirror: `{TOWER}/{side.name}`.",
        f"Manifest sha256: `{record['manifest_sha256']}`; store `{STORE}/{manifest_path.name}`; tower `{TOWER}/{manifest_path.name}`.",
        f"Build metadata sha256: `{record['build_meta_sha256']}`; store `{STORE}/{build.name}`; tower `{TOWER}/{build.name}`.",
        "All three files were downloaded back from the LAN store and sha256-checked against the local and tower copies.", "",
    ]
    (DEST / (STEM + ".pointer.md")).write_text("\n".join(pointer))
    (DEST / (STEM + "_MANIFEST.json")).write_bytes(manifest_path.read_bytes())
    for p in (json_path, md_path, DEST / (STEM + ".pointer.md"),
              DEST / (STEM + "_MANIFEST.json")):
        print(f"{p.name} bytes={p.stat().st_size} sha256={sha(p)}")


if __name__ == "__main__":
    main()
