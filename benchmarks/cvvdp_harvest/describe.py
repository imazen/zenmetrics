#!/usr/bin/env python3
"""TRAIN metric-label rank agreement through the canonical zen_stats owner."""
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, "/home/lilith/work/zen/zensim/scripts/lib")
from zen_stats import panel_batch_indexed

ROOT = "/var/tmp/cvvdp-safesyn"
CACHE = "/var/tmp/zensim-validation-2026-09-14/baseline-recovery/safesyn-train944.parquet"
ARMS = ["cvvdp_jod_standard_4k", "cvvdp_jod_standard_fhd",
        "cvvdp_jod_sdr_fhd_24", "ssim2_fresh"]


def band(v):
    if v >= -10:
        return "oracle_ge_m10"
    if v >= -25:
        return "oracle_m25_to_m10"
    if v >= -50:
        return "oracle_m50_to_m25"
    return "oracle_lt_m50"


def main():
    side = pq.read_table(os.path.join(ROOT, "safesyn_cvvdp_sidecar.parquet"),
                         columns=["row_id", "source_path"] + ARMS).to_pydict()
    cache = pq.read_table(CACHE, columns=["row_id", "original_oracle", "human_score"]).to_pydict()
    ix = {rid: i for i, rid in enumerate(cache["row_id"])}
    ids = side["row_id"]
    assert len(ids) == len(ix) == 196086 and len(set(ids)) == len(ids)
    assert set(ids) == set(ix)
    oracle = [cache["original_oracle"][ix[rid]] for rid in ids]
    stored_s2 = [cache["human_score"][ix[rid]] for rid in ids]
    fam = [p.split("/")[-2] for p in side["source_path"]]
    quality = [band(v) for v in oracle]
    bases = {"oracle": oracle, "stored_s2": stored_s2}
    bases.update({a: side[a] for a in ARMS})
    sets = {"overall": None}
    for v in sorted(set(fam)):
        sets["family/" + v] = [i for i, x in enumerate(fam) if x == v]
    for v in sorted(set(quality)):
        sets["band/" + v] = [i for i, x in enumerate(quality) if x == v]
    jobs = []
    for group, indices in sets.items():
        for arm in ARMS:
            jobs.append((group + "/" + arm, arm, "oracle", indices))
    for arm in ARMS:
        jobs.append(("overall_vs_stored_s2/" + arm, arm, "stored_s2", None))
    stats = panel_batch_indexed(bases, jobs, stats="srocc")
    assert len(stats) == len(jobs)
    results = {}
    for r in stats:
        assert r["n_dropped"] == 0
        results[r["label"]] = r
    large = sorted(range(len(ids)), key=lambda i: abs(
        side["cvvdp_jod_standard_4k"][i] - side["cvvdp_jod_sdr_fhd_24"][i]),
        reverse=True)[:20]
    examples = [{"row_id": ids[i], "family": fam[i],
                 "jod_4k": side["cvvdp_jod_standard_4k"][i],
                 "jod_fhd_24": side["cvvdp_jod_sdr_fhd_24"][i],
                 "abs_delta": abs(side["cvvdp_jod_standard_4k"][i]
                                  - side["cvvdp_jod_sdr_fhd_24"][i])}
                for i in large]
    out = {"rows": len(ids), "stat_owner": "zen_stats.panel_batch_indexed",
           "target_role": "TRAIN metric labels only", "results": results,
           "examples": examples}
    path = os.path.join(ROOT, "descriptive.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"rows={len(ids)} groups={len(sets)} stats={len(stats)} "
          f"all_n_dropped=0")
    for arm in ARMS:
        r = results["overall/" + arm]
        print(f"overall/{arm} n={r['n']} srocc={r['srocc']:.9f}")
    print("report=" + path)


if __name__ == "__main__":
    main()
