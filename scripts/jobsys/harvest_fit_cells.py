#!/usr/bin/env python3
"""Verify zenfleet fit blobs against manifest, ledger and receipts, then harvest.

`zenfleet-ctl ids` supplies job IDs; this script does not mint them or schedule
work. All blobs are verified before any original fit directory is moved.

One grid cell, one installed copy. A cell can be DONE in more than one jobset (a draining jobset's
running passes keep claiming, and a later jobset re-declares the remaining cells under a newer
program), so `--install` applies this rule:

* a target with no `fleet_receipt.json` is a local original: it is moved to `--rescue-root`
  and the fleet copy takes its place (a stale rescue copy blocks the move);
* a target that already carries a fleet receipt is a fleet cell and is NEVER moved or replaced:
  the first installed fleet copy wins. A later DONE copy of the same cell is verified like any
  other, compared with the installed copy on the required fields (`compare_science`), and, if
  they agree, reported as a duplicate and not installed. If they disagree, the whole install
  aborts before anything is touched.

Duplicate DONE rows for one job inside a ledger are reported, not silently collapsed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pyarrow.parquet as pq


POT_ROOT = Path("/var/tmp/rev4-featpot")
# fit_cell_exec.py archives members relative to /var/tmp/rev4-featpot, under <root>/<cell>/ where the
# root depends on the fit program (must match SCRIPTS in fit_cell_exec.py).
PROGRAM_ROOTS = {"mlp_probe.py": "fits", "p2_mlp.py": "p2/mlp", "p2_lodo_mlp.py": "p2/d2_mlp"}


# Program shas whose binaries carry tier parity (AVX-512 and scalar kernels reproduce AVX2 bit for
# bit). Empty: every fit program built so far (v6-v8) predates zensim main 322514dc, so a receipt
# from an AVX-512 host is only acceptable when it records the tier cap. Keep in step with
# fit_cell_exec.TIER_PARITY_PROGRAMS (test_fit_tools.py checks they are equal).
TIER_PARITY_PROGRAMS = frozenset()


def blob_root(kind: dict) -> str:
    return PROGRAM_ROOTS[kind["argv"][0]]


def cell_dir(stage: Path, name: str, kind: dict) -> Path:
    return stage / blob_root(kind) / name


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weights_sha(path: Path) -> str:
    """SHA-256 of a checkpoint's weight bytes, excluding the trailing embedded JSON metadata.

    zensim_mlp_train appends a JSON spec (hostname, timestamp_epoch, cwd, resolved input
    paths) after the weights, so whole-file hashes differ per run by design.
    """
    data = path.read_bytes()
    argv_at = data.find(b'"argv"')
    start = data.rfind(b'{"', 0, argv_at) if argv_at >= 0 else -1
    if start < 0:
        raise ValueError(f"no embedded metadata boundary in {path}")
    # Two little header fields hold the metadata JSON length (offset 68 and the 4 bytes before the JSON);
    # they change with source-path strings, not with the trained weights, so they are masked. On the
    # production pairs compared so far the only differing pre-trailer bytes were offsets 68-69.
    masked = bytearray(data[:start])
    masked[68:72] = b"\0\0\0\0"
    masked[start - 4:start] = b"\0\0\0\0"
    return hashlib.sha256(bytes(masked)).hexdigest()


def _normalized_result(result: dict) -> dict:
    result = json.loads(json.dumps(result))
    for inner in result.get("inner", []):
        inner.pop("log_sha256", None)
    for key in ("selected_bake", "selected_bake_sha256", "table_receipt_sha256"):
        result.pop(key, None)
    return result


def compare_science(a: Path, b: Path) -> dict:
    """Bit-for-bit comparison of two verified cell directories on the fields that carry the science.

    Drops only runtime-bearing and path-bearing fields (inner logs' hashes, the checkpoint's
    absolute path and whole-file hash, the table receipt hash); compares the rest of result.json,
    the selected checkpoint's weight bytes, the prediction files, and every importance family.
    """
    ra = json.loads((a / "result.json").read_text())
    rb = json.loads((b / "result.json").read_text())
    checks = {"result.json (minus runtime/path fields)": _normalized_result(ra) == _normalized_result(rb),
              "selected checkpoint weights": weights_sha(a / "refit" / Path(ra["selected_bake"]).name) ==
                                             weights_sha(b / "refit" / Path(rb["selected_bake"]).name)}
    for filename in ("test_preds.tsv", "eval_preds.tsv"):
        pa, pb = a / "refit" / filename, b / "refit" / filename
        if pa.exists() or pb.exists():
            checks[filename] = pa.is_file() and pb.is_file() and digest(pa) == digest(pb)
    ia, ib = a / "importance.json", b / "importance.json"
    if ia.exists() or ib.exists():
        if ia.is_file() and ib.is_file():
            fa = json.loads(ia.read_text())["family"]
            fb = json.loads(ib.read_text())["family"]
            checks["importance families (all)"] = fa.keys() == fb.keys() and all(
                fa[k]["permuted_srocc"] == fb[k]["permuted_srocc"] for k in fa)
        else:
            checks["importance families (all)"] = False
    return checks


def duplicate_done_rows(path: Path) -> dict:
    """Jobs with more than one DONE row in this ledger: {job_id: [output_sha, ...]}."""
    table = pq.read_table(path, columns=["job_id", "status", "output_sha"])
    seen: dict = {}
    for row in table.to_pylist():
        if row["status"] == "done":
            seen.setdefault(row["job_id"], []).append(row["output_sha"])
    return {job: shas for job, shas in seen.items() if len(shas) > 1}


def rows_from_ledger(path: Path) -> dict:
    table = pq.read_table(path, columns=["job_id", "status", "output_sha", "ts"])
    done = {}
    for row in table.to_pylist():
        if row["status"] != "done":
            continue
        previous = done.get(row["job_id"])
        if previous is None or row["ts"] > previous["ts"]:
            done[row["job_id"]] = row
    return done


def read_ids(path: Path) -> list[tuple[str, str]]:
    ids = []
    for line in path.read_text().splitlines():
        index, job_id, name = line.split("\t", 2)
        if int(index) != len(ids):
            raise ValueError("zenfleet-ctl ids order has a gap")
        ids.append((job_id, name))
    return ids


def fetch_blob(sha: str, prefix: str, cache: Path, endpoint: str) -> Path:
    path = cache / sha
    if path.is_file() and digest(path) == sha:
        return path
    cache.mkdir(parents=True, exist_ok=True)
    temp = cache / f"{sha}.download"
    temp.unlink(missing_ok=True)
    subprocess.run(["s5cmd", "--endpoint-url", endpoint, "cp", f"{prefix.rstrip('/')}/{sha}", str(temp)], check=True)
    if digest(temp) != sha:
        temp.unlink(missing_ok=True)
        raise ValueError(f"blob SHA-256 mismatch: {sha}")
    temp.replace(path)
    return path


def clean_blob(blob: Path, stage: Path) -> Path:
    """Return a readable path for a cell blob.

    The first cell each worker container runs downloads the data archive with an `s5cmd cp` whose
    progress line (`cp s3://... /scratch/...`) reached the executor's stdout, i.e. it prefixes that
    cell's output bytes. Every later cell of the container is clean. The prefix is a single text
    line before the gzip stream; strip exactly that and nothing else (receipts inside the tar are
    still verified below).
    """
    with blob.open("rb") as stream:
        head = stream.read(4096)
    if head[:2] == b"\x1f\x8b":
        return blob
    newline = head.find(b"\n")
    if not head.startswith(b"cp s3://") or newline < 0 or head[newline + 1:newline + 3] != b"\x1f\x8b":
        raise ValueError(f"blob is neither a gzip stream nor a gzip stream after one s5cmd line: {blob}")
    fixed = stage / "blob.tar.gz"
    with blob.open("rb") as src, fixed.open("wb") as dst:
        src.seek(newline + 1)
        shutil.copyfileobj(src, dst, 8 << 20)
    return fixed


def check_tier(receipt: dict, name: str) -> str:
    """Tier recorded by the executor (fit_cell_exec.effective_tier) -> "v3" | "v4" | "unrecorded".

    Receipts written before the executor recorded the tier (every cell of the 2026-09-25/26 era)
    carry none and are accepted as "unrecorded". A recorded AVX-512 (v4) run is rejected for any
    program without tier parity: its numerics differ from the AVX2 fleet's.
    """
    tier = receipt.get("tier")
    if tier is None:
        return "unrecorded"
    if tier.get("effective") == "v3":
        return "v3"
    if receipt.get("program_sha") in TIER_PARITY_PROGRAMS and tier.get("effective") == "v4":
        return "v4"
    raise ValueError(f"receipt tier {tier!r} is not the AVX2 tier and the program has no tier parity: {name}")


def verify_blob(blob: Path, stage: Path, name: str, kind: dict) -> dict:
    """Verify one cell blob into `stage`; returns the receipt (with `tier_status` added in memory)."""
    stage.mkdir(parents=True, exist_ok=True)
    blob = clean_blob(blob, stage)
    with tarfile.open(blob, "r:gz") as tar:
        members = tar.getmembers()
        prefix = f"{blob_root(kind)}/{name}/"
        for member in members:
            path = Path(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts or
                    not member.name.startswith(prefix)):
                raise ValueError(f"unsafe or out-of-cell blob member: {member.name}")
        tar.extractall(stage, filter="data")
    dest = cell_dir(stage, name, kind)
    receipt = json.loads((dest / "fleet_receipt.json").read_text())
    if (receipt.get("schema") != "zenfleet-fit-cell-receipt-v1" or
            receipt.get("cell") != name or
            receipt.get("program_sha") != kind["program_sha"] or
            receipt.get("data_sha") != kind["data_sha"] or
            receipt.get("argv_sha") != kind["argv_sha"]):
        raise ValueError(f"receipt identity mismatch: {name}")
    receipt["tier_status"] = check_tier(receipt, name)
    listed = receipt["files"]
    actual = {str(path.relative_to(dest)): digest(path) for path in dest.rglob("*")
              if path.is_file() and path.name != "fleet_receipt.json"}
    if listed != actual or listed["result.json"] != receipt["result_sha"]:
        raise ValueError(f"receipt file hash mismatch: {name}")
    result = json.loads((dest / "result.json").read_text())
    selected = Path(result["selected_bake"])
    install_dir = POT_ROOT / blob_root(kind) / name
    if selected != install_dir / "refit" / selected.name:
        raise ValueError(f"selected checkpoint destination mismatch: {name}")
    relative = selected.relative_to(install_dir)
    if digest(dest / relative) != result["selected_bake_sha256"] or \
            result["selected_bake_sha256"] != receipt["selected_bake_sha"]:
        raise ValueError(f"selected checkpoint SHA mismatch: {name}")
    if len(result["prediction"]) != result.get("test_rows", result.get("rows")):
        raise ValueError(f"prediction count mismatch: {name}")
    if Path(name).name.startswith("o"):
        # Outer-fold cells must carry the permutation-importance receipt that
        # summarize.py / check_core_gates.py require, bound to this result.json.
        importance_path = dest / "importance.json"
        if not importance_path.is_file():
            raise ValueError(f"missing importance.json: {name}")
        if json.loads(importance_path.read_text())["source_result_sha256"] != listed["result.json"]:
            raise ValueError(f"importance source hash mismatch: {name}")
    return receipt


def install_stages(stages: list, rescue_root: Path) -> dict:
    """Install verified (name, source_dir, target_dir) cells; see the module docstring for the rule."""
    already_installed, duplicates, to_install = set(), {}, []
    for name, source, target in stages:
        if not target.exists():
            to_install.append((name, source, target))
            continue
        current = target / "fleet_receipt.json"
        if not current.is_file():
            # A local original: fine to rescue, unless a rescue copy is already there.
            if (rescue_root / name).exists():
                raise ValueError(f"existing rescue and nonmatching target: {name}")
            to_install.append((name, source, target))
        elif digest(current) == digest(source / "fleet_receipt.json"):
            already_installed.add(name)
        else:
            # A different fleet copy of the same cell (another jobset/program): keep the installed one
            # if the science agrees, otherwise stop before anything is changed.
            checks = compare_science(source, target)
            if not all(checks.values()):
                raise ValueError(f"duplicate fleet copies of {name} disagree: "
                                 f"{sorted(k for k, ok in checks.items() if not ok)}")
            duplicates[name] = "agrees with the installed fleet copy; installed copy kept"
    for name, source, target in to_install:
        target.parent.mkdir(parents=True, exist_ok=True)
        rescue = rescue_root / name
        if target.exists():
            rescue.parent.mkdir(parents=True, exist_ok=True)
            if rescue.exists():
                raise ValueError(f"existing rescue prevents overwrite: {rescue}")
            target.rename(rescue)
        try:
            source.rename(target)
        except Exception:
            if rescue.exists() and not target.exists():
                rescue.rename(target)
            raise
    return {"installed": len(to_install), "already_installed": len(already_installed),
            "duplicates_kept_installed": duplicates, "rescued_originals_under": str(rescue_root)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True, help="zenfleet-ctl ids output")
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--blobs-prefix", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--scratch", type=Path, default=Path("/var/tmp/fleet-fits/harvest"))
    p.add_argument("--rescue-root", type=Path, default=Path("/var/tmp/fleet-fits/original-era"))
    p.add_argument("--install", action="store_true")
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    ids = read_ids(args.ids)
    if len(ids) != len(manifest):
        raise ValueError("manifest and zenfleet-ctl ids counts differ")
    done = rows_from_ledger(args.ledger)
    duplicate_rows = duplicate_done_rows(args.ledger)
    stages, tiers = [], {}
    for (job_id, name), job in zip(ids, manifest):
        if job["cell"]["image_path"] != name or job["kind"]["kind"] != "fit_cell":
            raise ValueError(f"manifest/ids mismatch at {name}")
        row = done.get(job_id)
        if row is None or not row["output_sha"]:
            raise ValueError(f"no DONE output in ledger for {name} ({job_id})")
        blob = fetch_blob(row["output_sha"], args.blobs_prefix, args.scratch / "blobs", args.endpoint)
        stage = args.scratch / "staged" / job_id
        if stage.exists():
            shutil.rmtree(stage)
        receipt = verify_blob(blob, stage, name, job["kind"])
        tiers[receipt["tier_status"]] = tiers.get(receipt["tier_status"], 0) + 1
        stages.append((name, cell_dir(stage, name, job["kind"]), POT_ROOT / blob_root(job["kind"]) / name))
    print(json.dumps({"verified": len(stages), "manifest": len(manifest), "install": args.install,
                      "tier_status": tiers, "duplicate_done_rows": duplicate_rows}))
    if not args.install:
        return
    print(json.dumps(install_stages(stages, args.rescue_root)))


if __name__ == "__main__":
    main()
