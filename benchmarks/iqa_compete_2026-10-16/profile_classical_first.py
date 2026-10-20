#!/usr/bin/env python3
"""Profile fresh CPU first-call workers; retain completed captures on adoption.

Caller supplies shared admission. Instrumented wall times are not measurements.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from compare_canonical import sha
from native_classical_first_cell import parse_native


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def worker(root):
    request = json.loads((root / "request.json").read_text())
    env = dict(os.environ, TMPDIR=str(root))
    for key in ["RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        env[key] = str(request["threads"])
    for number in range(request["repeats"]):
        with (root / f"worker-{number:03d}.log").open("x") as stream:
            subprocess.run([request["binary"], str(root / "config.json")],
                           env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("case")
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=99)
    parser.add_argument("--adopt", type=Path, help="completed perf capture with request/config and worker logs")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000 or not 1 <= args.frequency <= 999:
        parser.error("explicit repeat/frequency limits required")
    plan = json.loads(args.plan.read_text())
    case = next(c for c in plan["cases"] if c["id"] == args.case)
    if (plan.get("phase") != "scorer_construct_first_eval"
            or case["metric"].endswith("-gpu")
            or case["first_call_scope"].get("matched") is not True):
        parser.error("matched CPU constructor/first-evaluation case required")
    def verify():
        for path, expected in plan["bindings"].items():
            if sha(path) != expected:
                raise ValueError("bound artifact changed: " + path)
    verify()
    config = dict(width=case["width"], height=case["height"],
                  reference=case["reference"], distorted=case["distorted"],
                  metric_name=case["metric"], threads=case["threads"],
                  only_mode="score", timing_phase="cold", probe_only=False,
                  required_samples=1, locked_process_accounting=True)
    if shutil.disk_usage(args.output.parent).free < 1500 * 1024 * 1024:
        raise ValueError("at least 1500 MiB free disk required before capture/postprocessing")
    args.output.mkdir()
    root = args.output.resolve()
    request = dict(case=case["id"], plan=str(args.plan.resolve()), plan_sha256=sha(args.plan),
                   binary=case["binary"], threads=case["threads"], repeats=args.repeats)
    write(root / "request.json", request)
    write(root / "config.json", config)
    perf = str(Path(shutil.which("perf")).resolve())
    if args.adopt:
        capture = args.adopt.resolve(strict=True)
        original = json.loads((capture / "request.json").read_text())
        for key in ["case", "plan_sha256", "binary", "threads", "repeats"]:
            if original[key] != request[key]:
                raise ValueError("adopted capture identity differs: " + key)
        if json.loads((capture / "config.json").read_text()) != config:
            raise ValueError("adopted capture configuration differs")
        command = None
    else:
        capture = root
        command = [perf, "record", "-F", str(args.frequency), "--max-size", "32M",
                   "-e", "cycles:P", "--call-graph", "dwarf,8192", "-o", str(root / "perf.data"),
                   "--", sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(root)]
        with (root / "record.log").open("x") as stream:
            subprocess.run(command, env=dict(os.environ, TMPDIR=str(root)),
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    rows = [parse_native(path.read_text(), case) for path in sorted(capture.glob("worker-*.log"))]
    if len(rows) != args.repeats:
        raise ValueError("incomplete first-call repetitions; capture retained")
    # Per-task cycles recordings need not contain PERF_SAMPLE_CPU. Do not ask
    # perf script to synthesize a CPU field or infer placement from these data.
    commands = dict(script=[perf, "script", "-F", "comm,pid,tid,time,event,ip,sym,dso"],
                    report=[perf, "report", "--stdio", "--no-children"])
    for name, prefix in commands.items():
        with (root / f"{name}.txt").open("x") as stream, (root / f"{name}-stderr.log").open("x") as errors:
            subprocess.run([*prefix, "-i", str(capture / "perf.data")],
                           stdout=stream, stderr=errors, check=True)
    verify()
    write(root / "receipt.json", dict(case=case["id"], plan_sha256=sha(args.plan),
          controller_sha256=sha(__file__), perf_sha256=sha(perf), command=command,
          adopted_from=str(capture) if args.adopt else None,
          perf_data=str(capture / "perf.data"), perf_data_sha256=sha(capture / "perf.data"),
          config_sha256=sha(capture / "config.json"), repetitions=len(rows),
          worker_log_sha256={p.name: sha(p) for p in capture.glob("worker-*.log")},
          score_bits=sorted({row["score_bits"] for row in rows}),
          scope="Complete fresh cold-worker processes, including startup, input loading and teardown; no warm scorer evaluations. Sampled shares do not isolate the internal cold clock or worker-thread spans. Instrumented wall times are not comparison evidence; no CPU placement is inferred."))
    print("CLASSICAL_FIRST_PROFILE", case["id"], "adopted" if args.adopt else "captured", flush=True)


if __name__ == "__main__":
    main()
