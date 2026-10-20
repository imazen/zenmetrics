#!/usr/bin/env python3
"""Additive CUDA-only constructor/first-evaluation controller candidate.

Copied from the invocation-local metadata verifier; preserves every content-hash
check, numerical gate, deadline and alternating block. New explicit GPU names,
CUDA visibility, native runtime forwarding and helper self bindings only.
Original CPU helpers remain unchanged. Read-only controls required before use.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

from compare_canonical import compare, sha
from haarpsi_primary import execute
from supervision import timeout_command

PHASE = "scorer_construct_first_eval"
HELPERS = ("compare_classical_first.py", "compare_classical_first_verified.py",
           "compare_gpu_first_verified.py", "native_gpu_first_cell.py",
           "native_classical_first_cell.py",
           "compare_canonical.py", "haarpsi_primary.py", "octave_primary.py",
           "supervision.py")


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def parse_cold(text):
    rows = [line.split() for line in text.splitlines() if line.startswith("COLD ")]
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("exactly one COLD ns score required")
    ns, score = map(float, rows[0][1:])
    if not math.isfinite(ns) or ns <= 0 or not math.isfinite(score):
        raise ValueError("nonfinite score or invalid first-call time")
    return dict(ns=ns, score=score)


def scope_reasons(case, route):
    reasons = []
    if route["classification"] != "first_evaluation_candidate":
        reasons.append("review classification: " + route["classification"])
    if (route.get("adapter_pair_evaluations_before_first_timed_observation") != 0
            or route.get("canonical_cold_is_first_pair_evaluation") is not True):
        reasons.append("canonical observation is not an unprimed first evaluation")
    if route.get("canonical_emits_cold") is not True:
        reasons.append("canonical COLD unavailable")
    scope = case.get("first_call_scope", {})
    if scope.get("matched") is not True or scope.get("boundary") != PHASE:
        reasons.append("construction/parameter-I/O scope not explicitly matched")
    for field in ("construction", "parameter_io", "runtime", "teardown", "evidence"):
        if not scope.get(field):
            reasons.append("missing scope declaration: " + field)
    if case.get("matched_scope") is not True:
        reasons.append("source evaluator scope mismatched")
    return reasons


def summarize(blocks, reasons, expected):
    if reasons or len(blocks) != expected or expected < 3:
        return None
    ratios = []
    for number, block in enumerate(blocks):
        order = ["canonical", "native"] if number % 2 == 0 else ["native", "canonical"]
        if (block.get("errors") or block.get("passed") is False
                or block.get("order") != order):
            return None
        arms = block.get("arms", {})
        if set(arms) != {"native", "canonical"}:
            return None
        times = [arms[role]["ns"] for role in ("native", "canonical")]
        if not all(math.isfinite(t) and t > 0 for t in times):
            return None
        ratios.append(times[0] / times[1])
    return statistics.median(ratios)


def verify(plan_path, expected):
    if sha(plan_path) != expected:
        raise ValueError("plan changed")
    plan = json.loads(plan_path.read_text())
    if plan.get("phase") != PHASE:
        raise ValueError("plan phase must equal " + PHASE)
    if type(plan["seconds"]) is not int or not 1 <= plan["seconds"] <= 450:
        raise ValueError("arm deadline must be within 1..450 seconds")
    bindings = plan["bindings"]
    required = [str(Path(__file__).resolve().parent / name) for name in HELPERS]
    required += [plan["review"], sys.executable, "/usr/bin/time", "/usr/bin/timeout",
                 "/usr/bin/flock", str(Path.home() / "work/claudehints/scripts/run-heavy")]
    review = json.loads(Path(plan["review"]).read_text())
    routes = {r["metric"]: r for r in review["routes"]}
    if len({c["id"] for c in plan["cases"]}) != len(plan["cases"]):
        raise ValueError("duplicate case identity")
    # A new cache is created for EVERY verify invocation. Hashes are never
    # cached: the complete original bindings loop below still runs each time.
    sources = {}
    checked_source_bindings = set()
    for case in plan["cases"]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case["id"]) or case["id"] in (".", ".."):
            raise ValueError("unsafe case identity")
        if case["metric"] not in routes or case["metric"] not in {"butteraugli-gpu", "ssim2-gpu", "dssim-gpu", "iwssim-gpu", "cvvdp-gpu"} or case.get("gpu_runtime") != "cuda":
            raise ValueError("only reviewed explicit CUDA GPU names are admitted")
        for key in ("width", "height", "threads"):
            if type(case[key]) is not int or case[key] < 1:
                raise ValueError("invalid " + key)
        source_path = case["source_plan"]
        required.append(source_path)
        if source_path not in sources:
            source = json.loads(Path(source_path).read_text())
            originals = {}
            for original in source["cases"]:
                identity = original["id"]
                if not isinstance(identity, str) or identity in originals:
                    raise ValueError("invalid/duplicate source case identity")
                originals[identity] = original
            sources[source_path] = source, originals
        source, originals = sources[source_path]
        try:
            original = originals[case["source_case_id"]]
        except KeyError:
            raise ValueError("source case missing") from None
        for key, value in original.items():
            if case.get(key) != value:
                raise ValueError("source case changed: " + key)
        if case["binary"] != source["binary"] or case["rust_source"] != source["rust_source"]:
            raise ValueError("source worker identity changed")
        if case["canonical"]["command"] != routes[case["metric"]]["canonical_command"]:
            raise ValueError("canonical adapter differs from reviewed route")
        if source_path not in checked_source_bindings:
            for path, digest in source["bindings"].items():
                if bindings.get(path) != digest:
                    raise ValueError("source binding missing/changed: " + path)
            checked_source_bindings.add(source_path)
        required += [case["binary"], case["reference"], case["distorted"]]
        for bound in ("absolute_bound", "relative_bound"):
            value = case.get(bound)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("invalid numerical bound")
    for path in required:
        if path not in bindings:
            raise ValueError("unbound artifact: " + path)
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError("bound artifact changed: " + path)
    return plan, routes


def run_arm(case, role, directory, seconds):
    directory.mkdir()
    env = dict(os.environ, TMPDIR=str(directory), LC_ALL="C", CUDA_VISIBLE_DEVICES="0",
               HF_HUB_OFFLINE="1", ZENBENCH_NO_SAVE="1", ZENBENCH_CPU_POLICY="unrestricted",
               PYTHONDONTWRITEBYTECODE="1")
    for key in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(case["threads"])
    if role == "native":
        write(directory / "case.json", case)
        command = [sys.executable, str(Path(__file__).with_name("native_gpu_first_cell.py")),
                   case["binary"], str(directory / "case.json"), str(directory / "worker"),
                   "--seconds", str(seconds)]
    else:
        # Some frozen helpers require a positive sample count. The later SAMPLE
        # cannot contaminate the earlier COLD and is retained but not compared.
        fields = dict(case, samples=1, output=str(directory))
        command = [part.format_map(fields) for part in case["canonical"]["command"]]
    write(directory / "requested-command.json", command)
    rc, wrapped = execute(command, env, directory, role, min(1100, seconds + 30))
    write(directory / "execution.json", dict(command=wrapped, returncode=rc))
    if rc:
        raise ValueError(f"{role} exit {rc}; see retained stderr")
    observation = parse_cold((directory / (role + "-stdout.log")).read_text())
    if role == "native":
        observation["worker"] = json.loads((directory / "worker/observation.json").read_text())
    return observation


def block(plan_path, expected, case_id, number, output):
    output.mkdir()
    order = ["canonical", "native"] if number % 2 == 0 else ["native", "canonical"]
    result = dict(order=order, arms={}, errors={}, passed=False, phase=PHASE)
    try:
        plan, routes = verify(plan_path, expected)
        case = next(c for c in plan["cases"] if c["id"] == case_id)
        reasons = scope_reasons(case, routes[case["metric"]])
        result["scope_reasons"] = reasons
        if reasons:
            result["errors"]["scope"] = "; ".join(reasons)
        else:
            for role in order:
                # Admission surrounds hashing, all arm processes and verification.
                verify(plan_path, expected)
                try:
                    result["arms"][role] = run_arm(case, role, output / role, plan["seconds"])
                except Exception as error:
                    result["errors"][role] = str(error)
                verify(plan_path, expected)
            if len(result["arms"]) == 2:
                result["delta"], result["passed"] = compare(
                    result["arms"]["native"]["score"], result["arms"]["canonical"]["score"],
                    case["absolute_bound"], case.get("relative_bound", 0))
        verify(plan_path, expected)
    except Exception as error:
        result["errors"]["admission_or_binding"] = str(error)
    write(output / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--block-case")
    p.add_argument("--block-number", type=int, default=0)
    p.add_argument("--expected-plan-sha")
    a = p.parse_args()
    a.plan = a.plan.resolve(strict=True)
    a.output = a.output.absolute()
    if not a.output.resolve().is_relative_to((Path.home() / "tmp").resolve()):
        p.error("output must be under ~/tmp")
    if a.block_case:
        if not a.expected_plan_sha or a.block_number < 0:
            p.error("internal block requires frozen plan identity and nonnegative index")
        result = block(a.plan, a.expected_plan_sha, a.block_case, a.block_number, a.output)
        return 1 if result["errors"] or result["passed"] is False else 0
    if a.blocks < 3:
        p.error("minimum three alternating fresh-process blocks")
    plan_bytes = a.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if type(plan["seconds"]) is not int or not 1 <= plan["seconds"] <= 450:
        p.error("arm deadline must be within 1..450 seconds")
    a.output.mkdir()
    frozen = a.output / "plan.json"
    frozen.write_bytes(plan_bytes)
    expected = sha(frozen)
    rows = []
    for case in plan["cases"]:
        # Do not allow identifiers to escape output even before admitted validation.
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", case["id"]):
            raise ValueError("unsafe case identity")
        directory = a.output / case["id"]
        directory.mkdir()
        blocks = []
        for number in range(a.blocks):
            destination = directory / f"block-{number:02d}"
            command = [sys.executable, str(Path(__file__).resolve()), str(frozen), str(destination),
                       "--block-case", case["id"], "--block-number", str(number),
                       "--expected-plan-sha", expected]
            admitted = ["flock", str(Path.home() / "tmp/lilith-heavy.lock"),
                        "flock", str(Path.home() / "tmp/lilith-wsl-heavy.lock"),
                        str(Path.home() / "work/claudehints/scripts/run-heavy"),
                        "--mem", "12G", "--jobs", str(case["threads"]), "--",
                        *timeout_command("Linux", 1100), *command]
            # Lock waiting is outside the benchmark clocks. Only admitted work
            # and its arms have deadlines; shared queues can exceed five minutes.
            write(directory / f"block-{number:02d}-command.json", admitted)
            with (directory / f"block-{number:02d}-wrapper.log").open("x") as log:
                rc = subprocess.run(admitted, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, TMPDIR=str(directory),
                                             PYTHONDONTWRITEBYTECODE="1")).returncode
            path = destination / "result.json"
            result = json.loads(path.read_text()) if path.exists() else dict(
                errors={"block": f"no result; wrapper exit {rc}"}, passed=False)
            if rc:
                result.setdefault("errors", {})["wrapper"] = f"exit {rc}"
            blocks.append(result)
        reasons = sorted({s for b in blocks for s in b.get("scope_reasons", [])})
        row = dict(case=case["id"], metric=case["metric"], phase=PHASE, blocks=blocks,
                   first_call_scope=case["first_call_scope"], scope_reasons=reasons,
                   absolute_bound=case["absolute_bound"], relative_bound=case.get("relative_bound", 0),
                   ratio=summarize(blocks, reasons, a.blocks), plan_sha256=expected,
                   rust_source=case["rust_source"],
                   qualification="descriptive evaluator first calls; no CI/regression acceptance; "
                                 "null bound means no assigned numerical qualification")
        write(directory / "summary.json", row)
        rows.append(row)
        print("CASE_DONE", case["id"], "ratio", row["ratio"], flush=True)
    write(a.output / "summary.json", rows)
    return 1 if any(b.get("errors") or b["passed"] is False for r in rows for b in r["blocks"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
