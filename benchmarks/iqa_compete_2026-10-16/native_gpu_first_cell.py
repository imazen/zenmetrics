#!/usr/bin/env python3
"""Candidate explicit CUDA scorer cold call; never prime the worker."""
import argparse
import json
import math
import os
from pathlib import Path

from haarpsi_primary import execute


def parse_native(text, case):
    rows = [json.loads(line[8:]) for line in text.splitlines()
            if line.startswith("IQA_ROW ")]
    if len(rows) != 1:
        raise ValueError("exactly one IQA_ROW required")
    row = rows[0]
    expected = dict(metric_name=case["metric"], width=case["width"],
                    height=case["height"], threads=case["threads"],
                    timing_phase="cold", only_mode="score", n=1, status="ok", gpu_runtime="cuda")
    for key, value in expected.items():
        if type(row.get(key)) is not type(value) or row[key] != value:
            raise ValueError(f"native {key} mismatch/refusal: {row.get(key)!r}")
    for key in ("cold_elapsed_ns", "score"):
        value = row.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("invalid native " + key)
    if row["cold_elapsed_ns"] <= 0:
        raise ValueError("nonpositive native cold elapsed")
    return row


def run(binary, case, output, seconds):
    if case.get("gpu_runtime") != "cuda" or not case["metric"].endswith("-gpu"):
        raise ValueError("explicit CUDA GPU cold case required")
    output.mkdir()
    config = dict(width=case["width"], height=case["height"],
                  reference=case["reference"], distorted=case["distorted"],
                  metric_name=case["metric"], threads=case["threads"],
                  only_mode="score", timing_phase="cold", probe_only=False,
                  required_samples=1, locked_process_accounting=True, gpu_runtime="cuda")
    path = output / "config.json"
    path.write_text(json.dumps(config) + "\n")
    rc, command = execute([binary, str(path)], dict(os.environ, TMPDIR=str(output)),
                          output, "worker", seconds)
    (output / "command.json").write_text(json.dumps(command) + "\n")
    (output / "exit.json").write_text(json.dumps({"returncode": rc}) + "\n")
    if rc:
        raise ValueError(f"native worker exit {rc}; retained stderr")
    row = parse_native((output / "worker-stdout.log").read_text(), case)
    (output / "observation.json").write_text(json.dumps(row, indent=2) + "\n")
    print(f"COLD {row['cold_elapsed_ns']} {row['score']:.17g}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary")
    p.add_argument("case", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--seconds", type=int, default=300)
    a = p.parse_args()
    run(a.binary, json.loads(a.case.read_text()), a.output, a.seconds)


if __name__ == "__main__":
    main()
