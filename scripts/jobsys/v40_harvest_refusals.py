"""Replay actual local image blobs through V40's installation refusal owner."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--attempt", type=int, required=True)
    a = p.parse_args()
    reports = []
    for arm, smoke_attempt in (("control", 2), ("hb4", 1), ("hc4", 1), ("palette", 1)):
        key = f"{arm}-bounded-{smoke_attempt}"
        scratch = a.bundle / f"harvest-refusal-{a.attempt}-{key}"
        if scratch.exists():
            raise ValueError("fresh refusal scratch required")
        command = [
            sys.executable,
            str(a.bundle / "harvest_driver_v40.py"),
            "--bundle",
            str(a.bundle),
            "offline",
            str(a.bundle / f"smoke-manifest-{key}.json"),
            "--dry-run-blob",
            str(a.bundle / f"smoke-{key}/EXECUTOR_SHORT_OUTPUT.tar.gz"),
            "--scratch",
            str(scratch),
            "--install",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        (a.bundle / f"harvest-refusal-{a.attempt}-{key}.log").write_text(
            result.stdout + result.stderr
        )
        if (
            result.returncode != 1
            or "local smoke cannot install as a registered full-budget cell"
            not in result.stderr
        ):
            raise AssertionError((arm, result.returncode, result.stdout, result.stderr))
        reports.append(
            dict(
                arm=arm,
                returncode=result.returncode,
                command=command,
                error=result.stderr.strip(),
                installed=False,
            )
        )
    with (a.bundle / f"HARVEST_REFUSALS-{a.attempt}.json").open("x") as file:
        file.write(json.dumps(reports, indent=2) + "\n")
    print("PASS: all four actual bounded blobs verified and refused installation")


if __name__ == "__main__":
    main()
