#!/usr/bin/env python3
"""Export a jj tree without ignored files, repository metadata or Cargo output.

An empty revision snapshots the working copy first; explicit revisions never
snapshot it. The archive supplies file bytes, executable bits and symlinks from
one immutable tree, including jj-tracked additions and deletions.
"""

import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath


def output(repo, *args):
    return subprocess.check_output(args, cwd=repo)


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
        if archive.wait() != 0:
            raise RuntimeError("git archive failed")
    print(f"lock.sh: exported tracked tree {commit[:12]} (Cargo output excluded)")


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
