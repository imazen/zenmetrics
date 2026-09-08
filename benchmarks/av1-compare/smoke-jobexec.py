#!/usr/bin/env python3
"""Exercise declaration and the real executor contract; no scheduling is duplicated.

Run under run-heavy. Supply a canonical RGB PNG and a new output directory.
This is a protocol/correctness smoke, not a routing calibration experiment.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    binary, source, output = (p.resolve() for p in (args.binary, args.source, args.output))
    output.mkdir()
    scratch = output / "scratch"
    scratch.mkdir()
    blobs = output / "failure-blobs"
    env = dict(os.environ, TMPDIR=str(scratch), ZEN_CORPUS_DIR=str(source.parent),
               ZEN_AV1_FAILURE_STORE=json.dumps({"kind": "local", "path": str(blobs)}))
    reference = "svt-mainline-4.2.0-9292ec8e32bce26f781f277ec8739b53426c4300"
    arms = [{"backend": "zenav1-svt", "speed": -1, "quantizer": 32,
             "bit_depth": depth, "svt_reference": reference,
             "zen_intra_edge_filter": edge}
            for depth in (8, 10) for edge in (False, True)]
    arms.append({"backend": "libaom", "speed": 6, "quantizer": 32})
    item = {"image_path": source.name, "source_sha": hashlib.sha256(source.read_bytes()).hexdigest(),
            "codec": "av1-compare", "q": 0,
            "knob_tuple_json": json.dumps({"max_edges": [128], "arms": arms, "repeats": 3})}

    def declare(value):
        result = subprocess.run([str(binary), "declare"], input=json.dumps(value).encode(),
                                capture_output=True, env=env, check=True)
        jobs = json.loads(result.stdout)
        assert len(jobs) == 1
        return jobs[0]

    def execute(label, job, succeeds):
        print(f"{label}: starting", flush=True)
        (output / f"{label}.job.json").write_text(json.dumps(job, indent=2))
        with (output / f"{label}.tar").open("wb") as out, (output / f"{label}.stderr").open("wb") as err:
            result = subprocess.run([str(binary), "jobexec"], input=json.dumps(job).encode(),
                                    stdout=out, stderr=err, env=env)
        assert (result.returncode == 0) == succeeds, (label, result.returncode)
        print(f"{label}: exit {result.returncode}", flush=True)

    job = declare(item)
    execute("success", job, True)
    with tarfile.open(output / "success.tar") as archive:
        validation = json.load(archive.extractfile("comparison/validation.json"))
        rows = [json.loads(line) for line in archive.extractfile("comparison/rows.jsonl")]
        witnesses = list(archive.extractfile("comparison/reconstruction-verification.jsonl"))
        assert validation["complete"] and validation["verified_svt_cells"] == 4
        assert validation["svt_reconstruction_required"]
        assert len(rows) == 15 and len(witnesses) == 4
        for row in rows:
            payload = archive.extractfile(f"comparison/obu/{row['output_sha256']}.obu").read()
            assert hashlib.sha256(payload).hexdigest() == row["output_sha256"]

    invalid = copy.deepcopy(item)
    bad_arm = dict(arms[1], speed=0)  # Zen continuation requires native -1.
    invalid["knob_tuple_json"] = json.dumps({"max_edges": [128], "arms": [bad_arm], "repeats": 3})
    invalid_job = declare(invalid)
    execute("invalid-setting", invalid_job, False)
    saved = list(blobs.iterdir())
    assert len(saved) == 1
    assert hashlib.sha256(saved[0].read_bytes()).hexdigest() == saved[0].name
    with tarfile.open(saved[0]) as archive:
        failure = json.load(archive.extractfile("comparison-failure/failure.json"))
        assert failure["complete"] is False
        assert "validation.json" not in "\n".join(archive.getnames())
        assert any(name.endswith("-reference.png") for name in archive.getnames())
    assert str(saved[0]) in (output / "invalid-setting.stderr").read_text()

    wrong_build = copy.deepcopy(job)
    settings = json.loads(wrong_build["kind"]["knobs"])
    settings["binary_sha256"] = "0" * 64
    wrong_build["kind"]["knobs"] = json.dumps(settings)
    execute("wrong-build", wrong_build, False)
    assert "protocol/build mismatch" in (output / "wrong-build.stderr").read_text()

    # Exercise the canonical worker and ledger as well as direct jobexec.
    # Serial mode preserves the same timing-resource isolation used by sweeps.
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps([job, invalid_job]))
    worker_blobs = output / "worker-blobs"
    worker_failures = output / "worker-failures"
    worker_env = dict(env, ZEN_CHUNK_WALL_SEC="0", ZEN_AV1_FAILURE_STORE=json.dumps(
        {"kind": "local", "path": str(worker_failures)}))
    print("canonical worker: starting success and failure jobs", flush=True)
    with (output / "worker.log").open("wb") as log:
        subprocess.run([str(binary.with_name("zenfleet-worker")), "--manifest", str(manifest),
                        "--ledger-out", str(output / "ledger.parquet"),
                        "--blobs", str(worker_blobs), "--exec", str(binary)],
                       stdout=log, stderr=subprocess.STDOUT, env=worker_env, check=True)
    ledgers = list(output.glob("ledger*.parquet"))
    assert ledgers, "worker produced no ledger"
    command = [str(binary.with_name("zenfleet-ctl")), "catalog", "--manifest", str(manifest)]
    for ledger in ledgers:
        command.extend(["--ledger", str(ledger)])
    catalog = json.loads(subprocess.check_output(command, env=env))
    assert len(catalog) == 1
    assert {k: catalog[0][k] for k in ("total", "done", "poison", "gap")} == {
        "total": 2, "done": 1, "poison": 0, "gap": 1}
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    for directory, member, complete in (
        (worker_blobs, "comparison/validation.json", True),
        (worker_failures, "comparison-failure/failure.json", False),
    ):
        artifacts = list(directory.iterdir())
        assert len(artifacts) == 1
        payload = artifacts[0].read_bytes()
        assert hashlib.sha256(payload).hexdigest() == artifacts[0].name
        with tarfile.open(artifacts[0]) as archive:
            assert json.load(archive.extractfile(member))["complete"] is complete
    print("canonical worker: one done, one failed, both artifacts retained", flush=True)
    summary = {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
               "source_sha256": item["source_sha"], "measured_rows": len(rows),
               "verified_svt_cells": len(witnesses), "failure_blob_sha256": saved[0].name,
               "worker_catalog": catalog, "complete": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
