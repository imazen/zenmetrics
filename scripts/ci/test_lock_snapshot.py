#!/usr/bin/env python3
"""Exercise the real lock driver with synthetic siblings and a Cargo tripwire."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class LockSnapshotTest(unittest.TestCase):
    def run_command(self, *args, **kwargs):
        result = subprocess.run(
            args, cwd=self.repo, env=self.env, capture_output=True, text=True, **kwargs
        )
        if result.returncode:
            self.fail(
                f"{args}: exit {result.returncode}\n{result.stdout}\n{result.stderr}"
            )
        return result

    def setUp(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(
            prefix="lock-snapshot-test-", dir=scratch
        )
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        self.snap = Path(self.temp.name) / "snapshot"
        self.env = dict(os.environ, LOCK_SNAP_DIR=str(self.snap), TMPDIR=str(scratch))
        self.run_command("git", "init", "--quiet")
        self.run_command("jj", "git", "init", "--colocate")
        self.write(".gitignore", "/ignored/\n/target*/\n**/target*/\n.workongoing\n")
        for name in ("lock.sh", "export-snapshot.py"):
            dest = self.repo / "scripts/ci" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = Path(__file__).parent / name
            if name == "lock.sh" and os.environ.get("LOCK_SNAPSHOT_TEST_DRIVER"):
                source = Path(os.environ["LOCK_SNAPSHOT_TEST_DRIVER"])
            shutil.copy2(source, dest)
        self.write("Cargo.lock", "locked\n")
        self.write("ci/sibling-pins.tsv", "synthetic pin\n")
        self.write("source.txt", "original\n")
        self.write("deleted.txt", "original\n")
        self.write("target-source/source.txt", "legitimate source\n")
        self.run_command(
            "jj", "file", "track", "--include-ignored", "target-source/source.txt"
        )
        self.write(
            "scripts/ci/clone-siblings.sh", "#!/bin/sh\nexit 0\n", executable=True
        )
        self.run_command("jj", "describe", "-m", "test: initial snapshot fixture")
        self.base = self.run_command(
            "jj", "log", "--no-graph", "-r", "@", "-T", "commit_id"
        ).stdout
        self.run_command("jj", "new", "-m", "test: working tree source edits")
        self.write("source.txt", "edited\n")
        self.write("new file\nwith newline.txt", "new\n", executable=True)
        (self.repo / "deleted.txt").unlink()
        (self.repo / "source-link").symlink_to("source.txt")
        self.write("ignored/payload", "must not copy\n")
        self.write("target-gpu-rev2/CACHEDIR.TAG", "cache\n")
        self.write("target-gpu-rev2/payload", "must not copy\n")
        self.write("nested/target-bookworm/.rustc_info.json", "{}\n")
        self.write("nested/target-bookworm/payload", "must not copy\n")
        # An accidentally tracked cache must also be excluded, in both modes.
        self.write("nested/target-tracked/CACHEDIR.TAG", "cache\n")
        self.write("nested/target-tracked/payload", "must not copy\n")
        self.run_command(
            "jj",
            "file",
            "track",
            "--include-ignored",
            "nested/target-tracked/CACHEDIR.TAG",
            "nested/target-tracked/payload",
        )
        self.write(".workongoing", "synthetic lock\n")
        tools = Path(self.temp.name) / "tools"
        tools.mkdir()
        cargo = tools / "cargo"
        cargo.write_text("""#!/bin/sh
set -eu
[ "$1" = fetch ] || [ "$1" = metadata ]
[ "$1" = metadata ] || [ "$2" = --locked ]
test "$(cat Cargo.lock)" = locked
test "$(cat ci/sibling-pins.tsv)" = 'synthetic pin'
test ! -e .jj
test ! -e .git
test ! -e .workongoing
test ! -e ignored
test ! -e target-gpu-rev2
test ! -e nested/target-bookworm
test ! -e nested/target-tracked
""")
        cargo.chmod(0o755)
        self.env["PATH"] = str(tools) + os.pathsep + self.env["PATH"]

    def write(self, name, contents, executable=False):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        if executable:
            path.chmod(0o755)

    def test_working_tree_and_revision_check(self):
        self.run_command("bash", "scripts/ci/lock.sh", "--check")
        tree = self.snap / "work/zenmetrics"
        self.assertEqual((tree / "source.txt").read_text(), "edited\n")
        self.assertFalse((tree / "deleted.txt").exists())
        self.assertTrue((tree / "new file\nwith newline.txt").stat().st_mode & 0o111)
        self.assertEqual(os.readlink(tree / "source-link"), "source.txt")
        self.assertEqual(
            (tree / "target-source/source.txt").read_text(), "legitimate source\n"
        )
        working = self.run_command(
            "jj", "log", "--no-graph", "-r", "@", "-T", "commit_id"
        ).stdout
        self.run_command("bash", "scripts/ci/lock.sh", "--check", "--rev", working)
        self.assertEqual((tree / "source.txt").read_text(), "edited\n")
        self.run_command("bash", "scripts/ci/lock.sh", "--check", "--rev", self.base)
        self.assertEqual((tree / "source.txt").read_text(), "original\n")
        self.assertTrue((tree / "deleted.txt").exists())
        self.assertFalse((tree / "source-link").exists())

    def test_regen_keeps_matching_lock(self):
        self.run_command("bash", "scripts/ci/lock.sh", "--regen")
        self.assertEqual((self.repo / "Cargo.lock").read_text(), "locked\n")

    def test_check_refuses_mismatched_lock_without_rewriting(self):
        self.write("Cargo.lock", "mismatched\n")
        result = subprocess.run(
            ["bash", "scripts/ci/lock.sh", "--check"],
            cwd=self.repo,
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual((self.repo / "Cargo.lock").read_text(), "mismatched\n")

    def test_bad_revision_is_setup_failure(self):
        result = subprocess.run(
            ["bash", "scripts/ci/lock.sh", "--check", "--rev", "does-not-exist"],
            cwd=self.repo,
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == "__main__":
    unittest.main()
