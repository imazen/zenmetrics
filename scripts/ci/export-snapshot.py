#!/usr/bin/env python3
"""Export a jj tree without ignored files, repository metadata or Cargo output.

An empty revision snapshots the working copy first; explicit revisions never
snapshot it. The archive supplies file bytes, executable bits and symlinks from
one immutable tree, including jj-tracked additions and deletions.

File mtimes: git archive stamps every file with the commit time. Cargo decides
freshness by comparing source mtimes with the previous build's dep-info, so an
older revision exported after a newer build would look up to date and cargo
would hand back the newer revision's binary. A manifest beside the tree
(`<tree>.mtimes.json`: path -> [sha256, mode, mtime_ns]) keeps the previous
mtime for byte-identical files, so incremental builds stay incremental, and
stamps every new or changed file later than anything stamped before.
"""

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath


def output(repo, *args):
    return subprocess.check_output(args, cwd=repo)


def load_manifest(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def restamp(destination, extracted, manifest_path):
    """Give unchanged files their previous mtime and changed ones a newer one."""
    old = load_manifest(manifest_path)
    newest_old = max((entry[2] for entry in old.values()), default=0)
    stamp = max(time.time_ns(), newest_old + 1)
    new = {}
    for name, mode in extracted:
        path = destination / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = old.get(name)
        if previous and previous[0] == digest and previous[1] == mode:
            mtime = previous[2]
        else:
            mtime = stamp
        os.utime(path, ns=(mtime, mtime))
        new[name] = [digest, mode, mtime]
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(new, f)
    os.replace(tmp, manifest_path)


def export(repo, destination, revision):
    git_dir = (
        output(repo, "jj", "git", "root", "--ignore-working-copy").decode().strip()
    )
    command = ["jj", "log", "--no-graph", "-r", revision or "@", "-T", "commit_id"]
    if revision:
        command.append("--ignore-working-copy")
    commit = output(repo, *command).decode().strip()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("snapshot revision must resolve to exactly one Git commit")
    git = ["git", f"--git-dir={git_dir}"]
    names = output(repo, *git, "ls-tree", "-rz", "--name-only", commit)
    paths = [
        PurePosixPath(name.decode("utf-8", "surrogateescape"))
        for name in names.split(b"\0")
        if name
    ]
    blocked = set()
    for path in paths:
        # Markers may be tracked accidentally, or ignored in a working tree
        # that has tracked members underneath a custom Cargo target directory.
        if path.name in {"CACHEDIR.TAG", ".rustc_info.json"}:
            blocked.update(
                parent for parent in path.parents if parent.name.startswith("target")
            )
        if not revision:
            for parent in path.parents:
                if parent.name.startswith("target") and any(
                    (repo / parent / marker).is_file()
                    for marker in ("CACHEDIR.TAG", ".rustc_info.json")
                ):
                    blocked.add(parent)

    extracted = []
    with subprocess.Popen(
        [*git, "archive", "--format=tar", commit], stdout=subprocess.PIPE, cwd=repo
    ) as archive:
        with tarfile.open(fileobj=archive.stdout, mode="r|") as members:
            for member in members:
                path = PurePosixPath(member.name)
                if any(part in {".git", ".jj", ".workongoing"} for part in path.parts):
                    continue
                if any(path == parent or parent in path.parents for parent in blocked):
                    continue
                # Git archive contains relative paths; data filtering also
                # rejects path traversal and symlinks escaping the snapshot.
                members.extract(member, destination, filter="data")
                if member.isfile():
                    extracted.append((member.name, member.mode & 0o777))
        if archive.wait() != 0:
            raise RuntimeError("git archive failed")
    restamp(destination, extracted, destination.with_name(destination.name + ".mtimes.json"))
    print(f"lock.sh: exported tracked tree {commit[:12]} (Cargo output excluded)")


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
