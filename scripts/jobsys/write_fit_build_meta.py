#!/usr/bin/env python3
"""Audit both Cargo locks against fetched main heads and write fit build provenance."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from urllib.parse import urlsplit


ZENFLEET_CHANGED = {"zenfleet-core", "zenfleet-ctl", "zenfleet-worker"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def commit(value: str) -> str:
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"not a full lowercase git commit ID: {value!r}")
    return value


def git_repo(source: str) -> tuple[str, str]:
    address, revision = source.rsplit("#", 1)
    name = Path(urlsplit(address.removeprefix("git+").split("?", 1)[0]).path).name
    return name.removesuffix(".git"), commit(revision)


def packages(lock: Path, workspace: str, base: str, lane: str,
             git_heads: dict[str, str]) -> list[dict]:
    rows = []
    for package in tomllib.loads(lock.read_text())["package"]:
        source = package.get("source", "")
        row = {"name": package["name"], "version": package["version"]}
        if not source:
            row["source"] = "workspace"
            row["repo"] = workspace
            row["commit"] = lane if workspace == "zenmetrics" and row["name"] in ZENFLEET_CHANGED else base
        elif source.startswith("git+"):
            repo, actual = git_repo(source)
            expected = git_heads.get(repo)
            if expected is None or actual != expected:
                raise ValueError(f"{row['name']}: {repo} at {actual}, expected current main {expected}")
            row.update(source="git", repo=repo, commit=actual)
        elif source.startswith("registry+"):
            row.update(source="registry", registry=source,
                       checksum=package.get("checksum"))
        else:
            raise ValueError(f"{row['name']}: unrecognized Cargo source {source}")
        rows.append(row)
    return sorted(rows, key=lambda row: (row["name"], row["version"], row["source"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zenmetrics-lock", type=Path, required=True)
    p.add_argument("--zensim-lock", type=Path, required=True)
    p.add_argument("--zenmetrics-main", required=True)
    p.add_argument("--zenmetrics-lane", required=True)
    p.add_argument("--zensim-main", required=True)
    p.add_argument("--scripts-commit", required=True)
    p.add_argument("--git-main", action="append", required=True,
                   help="repeat repo=40-character fetched main/master commit")
    p.add_argument("--image-recipe", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    heads = {}
    for item in args.git_main:
        repo, value = item.split("=", 1)
        if repo in heads:
            raise ValueError(f"duplicate git main for {repo}")
        heads[repo] = commit(value)
    zenmetrics_main = commit(args.zenmetrics_main)
    zenmetrics_lane = commit(args.zenmetrics_lane)
    zensim_main = commit(args.zensim_main)
    script_commit = commit(args.scripts_commit)
    crates = {
        "zenmetrics": packages(args.zenmetrics_lock, "zenmetrics", zenmetrics_main,
                               zenmetrics_lane, heads),
        "zensim": packages(args.zensim_lock, "zensim", zensim_main,
                           zensim_main, heads),
    }
    metadata = {
        "schema": "fleet-fits-build-v1",
        "zenmetrics_main": zenmetrics_main,
        "zenmetrics_lane": zenmetrics_lane,
        "zensim_main": zensim_main,
        "p0_scripts_commit": script_commit,
        "git_main_heads": dict(sorted(heads.items())),
        "cargo_locks": {"zenmetrics": sha(args.zenmetrics_lock),
                        "zensim": sha(args.zensim_lock)},
        "crates": crates,
        "image_recipe_sha256": sha(args.image_recipe),
        "python_build_version": sys.version.split()[0],
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "cargo": subprocess.check_output(["cargo", "--version"], text=True).strip(),
        "rustflags": os.environ.get("RUSTFLAGS", ""),
        "zensim_profile": "dev",
        "zensim_target": "x86_64-unknown-linux-gnu",
        "fit_userland": "ubuntu:26.04 (glibc 2.43), same as dev box and LAN workers",
        "zenfleet_profile": "release",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"out": str(args.out),
                      "crate_records": {key: len(value) for key, value in crates.items()},
                      "git_heads": len(heads), "sha256": sha(args.out)}))


if __name__ == "__main__":
    main()
