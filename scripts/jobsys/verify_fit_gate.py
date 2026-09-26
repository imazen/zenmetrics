#!/usr/bin/env python3
"""Compare two zenfleet pilot replays to dev originals, without score thresholds."""

import argparse
import json
from pathlib import Path
import shutil

from harvest_fit_cells import cell_dir, digest, read_ids, rows_from_ledger, verify_blob, weights_sha


FIELDS = ("prediction", "score", "selected_epoch", "inner_mean_geomean3_by_epoch",
          "bounds", "fit_rows", "test_rows", "test_index", "init_seed", "sample_seed",
          "label_source", "preflight_sha256")


def load_gate(root: Path, manifest: list, ids: list, label: str) -> dict:
    rows = rows_from_ledger(root / "ledger.parquet")
    values = {}
    for (job_id, name), job in zip(ids, manifest):
        row = rows.get(job_id)
        if row is None or not row["output_sha"]:
            raise ValueError(f"{label}: no DONE row for {name}")
        blob = root / "blobs" / row["output_sha"]
        if digest(blob) != row["output_sha"]:
            raise ValueError(f"{label}: blob SHA mismatch for {name}")
        stage = root / "verified" / job_id
        if stage.exists():
            shutil.rmtree(stage)
        verify_blob(blob, stage, name, job["kind"])
        dest = cell_dir(stage, name, job["kind"])
        result = json.loads((dest / "result.json").read_text())
        selected = dest / "refit" / Path(result["selected_bake"]).name
        values[name] = {"result": result, "selected_sha": weights_sha(selected), "selected_full_sha": digest(selected),
                        "result_sha": digest(dest / "result.json")}
    return values


def compare(a: dict, b: dict) -> dict:
    return {field: a["result"][field] == b["result"][field] for field in FIELDS} | {
        "inner_excluding_runtime_log_hash": [
            {key: value for key, value in inner.items() if key != "log_sha256"}
            for inner in a["result"]["inner"]
        ] == [
            {key: value for key, value in inner.items() if key != "log_sha256"}
            for inner in b["result"]["inner"]
        ],
        "selected_checkpoint_bytes": a["selected_sha"] == b["selected_sha"],
        "selected_checkpoint_full_file": a["selected_full_sha"] == b["selected_full_sha"],
        "full_result_bytes": a["result_sha"] == b["result_sha"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--gate-a", type=Path, required=True)
    p.add_argument("--gate-b", type=Path, required=True)
    p.add_argument("--original-root", type=Path, default=Path("/var/tmp/rev4-featpot/fits"))
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    ids = read_ids(args.ids)
    if len(manifest) != len(ids):
        raise ValueError("manifest/ids count mismatch")
    for (_, name), job in zip(ids, manifest):
        if job["cell"]["image_path"] != name or job["kind"]["kind"] != "fit_cell":
            raise ValueError(f"manifest/ids mismatch at {name}")
    a = load_gate(args.gate_a, manifest, ids, "a")
    b = load_gate(args.gate_b, manifest, ids, "b")
    report = {}
    for _, name in ids:
        original_path = args.original_root / name / "result.json"
        original_result = json.loads(original_path.read_text())
        original = {"result": original_result, "result_sha": digest(original_path),
                    "selected_sha": weights_sha(Path(original_result["selected_bake"])),
                    "selected_full_sha": digest(Path(original_result["selected_bake"]))}
        report[name] = {"a_vs_original": compare(a[name], original),
                        "b_vs_original": compare(b[name], original),
                        "a_vs_b": compare(a[name], b[name])}
    required = (*FIELDS, "inner_excluding_runtime_log_hash", "selected_checkpoint_bytes")
    passed = all(all(checks[field] for field in required)
                 for cell in report.values() for checks in cell.values())
    print(json.dumps({"gate_pass": passed, "cells": report}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
