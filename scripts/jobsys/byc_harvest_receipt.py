#!/usr/bin/env python3
"""Inventory verified byc cells and every retained refit file for later evaluation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from harvest_fit_cells import digest, read_ids, rows_from_ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError("preserve the previous harvest receipt")
    manifest = json.loads(args.manifest.read_text())
    ids, done = read_ids(args.ids), rows_from_ledger(args.ledger)
    if not manifest or len(manifest) != len(ids):
        raise ValueError("manifest/ids inventory differs")
    cells = []
    for (job_id, name), job in zip(ids, manifest, strict=True):
        kind = job["kind"]
        if kind["argv"][0] != "byc_cells.py" or name != job["cell"]["image_path"]:
            raise ValueError("receipt requires only manifest-bound byc cells")
        path = args.cells / name
        receipt = json.loads((path / "fleet_receipt.json").read_text())
        if (receipt["cell"] != name or any(receipt[k] != kind[k] for k in ("program_sha", "data_sha", "argv_sha"))
                or not done.get(job_id, {}).get("output_sha")):
            raise ValueError("cell receipt or DONE binding differs")
        listed = receipt["files"]
        actual = {str(f.relative_to(path)): digest(f) for f in path.rglob("*")
                  if f.is_file() and f.name != "fleet_receipt.json"}
        if actual != listed:
            raise ValueError("retained cell files differ from the verified archive")
        refit = {k: v for k, v in listed.items() if k.startswith("refit/")}
        if "refit/last.bin" not in refit:
            raise ValueError("retained candidate bake missing")
        cells.append(dict(cell=name, path=str(path), job_id=job_id,
                          output_blob_sha256=done[job_id]["output_sha"],
                          fleet_receipt_sha256=digest(path / "fleet_receipt.json"),
                          selected_bake_sha256=receipt["selected_bake_sha"], refit_files=refit,
                          dense_files={k: v for k, v in listed.items() if k.startswith("dense/")}))
    report = dict(schema="byc-harvest-retained-bakes-v1", observed_at=datetime.now(timezone.utc).isoformat(),
                  verified=len(cells), expected=len(manifest), manifest_sha256=digest(args.manifest),
                  ledger_sha256=digest(args.ledger), cells=cells)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(verified=len(cells), receipt=str(args.out), sha256=digest(args.out))))


if __name__ == "__main__":
    main()
