#!/usr/bin/env python3
"""Tests for the fit-cell tools: harvest verification (negative controls), install policy for cells
DONE in more than one jobset, the tier guard, the grid-coverage gate, and the golden FitCell job id.

    python3 -m unittest scripts/jobsys/test_fit_tools.py        (needs pyarrow)

Synthetic cells only: no network, no LAN store, no fit binaries.
"""

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_cell_exec as exe  # noqa: E402
import fit_grid_coverage as cov  # noqa: E402
import harvest_fit_cells as h  # noqa: E402
import pack_fit_data_v2 as pack  # noqa: E402

PROGRAM = "a" * 64
DATA = "b" * 64
ARGV = ["mlp_probe.py", "--set", "s", "--outer", "0"]
ARGV_SHA = hashlib.sha256(json.dumps(ARGV, separators=(",", ":")).encode()).hexdigest()
KIND = {"kind": "fit_cell", "program_sha": PROGRAM, "data_sha": DATA, "argv_sha": ARGV_SHA, "argv": ARGV}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checkpoint(weights: bytes, host: str) -> bytes:
    """Header (72 B) + weights + embedded JSON metadata trailer, like zensim_mlp_train's output."""
    return b"\x01" * 72 + weights + json.dumps({"host": host, "argv": ["x"]}).encode()


class Cell:
    """A synthetic cell blob; every knob a negative control needs is an argument."""

    def __init__(self, tmp: Path, name="POT_s_r0_mlp32/o0_r0", weights=b"W" * 64, host="h1", preds=3,
                 importance="ok", kind=None, tier=None, extra_receipt=None):
        self.name, self.kind = name, kind or KIND
        pot = h.POT_ROOT
        install = pot / "fits" / name
        ckpt = checkpoint(weights, host)
        result = {"score": {"srocc": 0.9}, "selected_epoch": 5, "prediction": [0.1] * preds, "test_rows": 3,
                  "inner": [{"epoch": 5, "log_sha256": sha(host.encode())}],
                  "selected_bake": str(install / "refit" / "best.bin"), "selected_bake_sha256": sha(ckpt)}
        result_bytes = json.dumps(result, sort_keys=True).encode()
        files = {"result.json": result_bytes, "refit/best.bin": ckpt, "refit/test_preds.tsv": b"0.1\n0.1\n0.1\n"}
        if importance != "none":
            source = sha(result_bytes) if importance == "ok" else "0" * 64
            files["importance.json"] = json.dumps(
                {"source_result_sha256": source, "family": {"basic": {"permuted_srocc": 0.5}}}).encode()
        self.files = files
        listed = {k: sha(v) for k, v in files.items()}
        self.receipt = {"schema": "zenfleet-fit-cell-receipt-v1", "program_sha": self.kind["program_sha"],
                        "data_sha": self.kind["data_sha"], "argv_sha": self.kind["argv_sha"], "cell": name,
                        "result_sha": sha(result_bytes), "selected_bake_sha": sha(ckpt), "files": listed}
        if tier is not None:
            self.receipt["tier"] = tier
        self.receipt.update(extra_receipt or {})

    def blob(self, tmp: Path, prefix_lines=0, tamper=None, extra=None, drop=None, rename_dir=None, links=None):
        files = dict(self.files)
        if tamper:
            files[tamper] = files[tamper] + b"x"
        if drop:
            files.pop(drop)
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as z, tarfile.open(fileobj=z, mode="w") as tar:
            base = rename_dir or f"fits/{self.name}"
            members = {f"{base}/{k}": v for k, v in files.items()}
            members[f"{base}/fleet_receipt.json"] = json.dumps(self.receipt).encode()
            members.update({k: v for k, v in (extra or {}).items()})
            for arc, data in members.items():
                info = tarfile.TarInfo(arc)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            for arc, target in (links or {}).items():
                info = tarfile.TarInfo(arc)
                info.type, info.linkname = tarfile.SYMTYPE, target
                tar.addfile(info)
        raw = buf.getvalue()
        path = tmp / f"blob-{sha(raw)[:8]}-{prefix_lines}"
        path.write_bytes(b"cp s3://x /y\n" * prefix_lines + raw)
        return path


class HarvestVerify(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)
        self._old = h.POT_ROOT
        h.POT_ROOT = self.tmp / "pot"
        self.addCleanup(lambda: setattr(h, "POT_ROOT", self._old))
        self.addCleanup(self._dir.cleanup)
        self.n = 0

    def verify(self, cell: Cell, **blob_args):
        self.n += 1
        return h.verify_blob(cell.blob(self.tmp, **blob_args), self.tmp / f"stage{self.n}", cell.name, cell.kind)

    def rejects(self, cell, **blob_args):
        with self.assertRaises(Exception):
            self.verify(cell, **blob_args)

    def test_accepts_valid_cells(self):
        self.assertEqual(self.verify(Cell(self.tmp))["tier_status"], "unrecorded")
        # A full cell ("full_r0") carries no importance receipt.
        self.verify(Cell(self.tmp, name="POT_s_r0_mlp32/full_r0", importance="none"))
        # The first cell of a container has one s5cmd progress line before the gzip stream.
        self.verify(Cell(self.tmp), prefix_lines=1)

    def test_rejects_two_prefix_lines(self):
        self.rejects(Cell(self.tmp), prefix_lines=2)

    def test_rejects_tampering(self):
        c = Cell(self.tmp)
        self.rejects(c, tamper="refit/test_preds.tsv")
        self.rejects(c, extra={"fits/POT_s_r0_mlp32/o0_r0/extra.txt": b"x"})
        self.rejects(c, drop="refit/test_preds.tsv")
        self.rejects(c, tamper="refit/best.bin")

    def test_rejects_identity_mismatches(self):
        for field in ("program_sha", "data_sha", "argv_sha", "cell", "schema"):
            self.rejects(Cell(self.tmp, extra_receipt={field: "wrong"}))

    def test_rejects_unsafe_members(self):
        c = Cell(self.tmp)
        self.rejects(c, extra={"fits/POT_other/o0_r0/x": b"x"})
        self.rejects(c, extra={"fits/POT_s_r0_mlp32/o0_r0/../../evil": b"x"})
        self.rejects(c, links={"fits/POT_s_r0_mlp32/o0_r0/link": "/etc/passwd"})

    def test_rejects_result_problems(self):
        self.rejects(Cell(self.tmp, preds=2))  # prediction count != test_rows
        self.rejects(Cell(self.tmp, importance="wrong"))  # importance bound to another result
        self.rejects(Cell(self.tmp, importance="none"))  # outer cell without importance

    def test_tier_recorded_in_receipt(self):
        v3 = {"cpu_avx512f": True, "max_tier_env": "v3", "effective": "v3"}
        v4 = {"cpu_avx512f": True, "max_tier_env": "", "effective": "v4"}
        self.assertEqual(self.verify(Cell(self.tmp, tier=v3))["tier_status"], "v3")
        self.rejects(Cell(self.tmp, tier=v4))  # program without tier parity, AVX-512 numerics
        # A program listed as tier-parity may record v4.
        h.TIER_PARITY_PROGRAMS = frozenset({PROGRAM})
        self.addCleanup(lambda: setattr(h, "TIER_PARITY_PROGRAMS", frozenset()))
        self.assertEqual(self.verify(Cell(self.tmp, tier=v4))["tier_status"], "v4")

    def test_weights_sha_ignores_metadata_trailer(self):
        a, b, c = (self.tmp / n for n in "abc")
        a.write_bytes(checkpoint(b"W" * 64, "host-1"))
        b.write_bytes(checkpoint(b"W" * 64, "another-host-with-a-longer-name"))
        c.write_bytes(checkpoint(b"V" * 64, "host-1"))
        self.assertEqual(h.weights_sha(a), h.weights_sha(b))
        self.assertNotEqual(h.weights_sha(a), h.weights_sha(c))


class InstallPolicy(unittest.TestCase):
    """One grid cell, one installed copy: never move a fleet cell; agreeing duplicates are kept out."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.tmp = Path(self._dir.name)
        self._old = h.POT_ROOT
        h.POT_ROOT = self.tmp / "pot"
        self.addCleanup(lambda: setattr(h, "POT_ROOT", self._old))
        self.rescue = self.tmp / "rescue"
        self.n = 0

    def stage(self, cell: Cell) -> tuple:
        self.n += 1
        stage = self.tmp / f"stage{self.n}"
        h.verify_blob(cell.blob(self.tmp), stage, cell.name, cell.kind)
        return cell.name, h.cell_dir(stage, cell.name, cell.kind), h.POT_ROOT / "fits" / cell.name

    def test_local_original_is_rescued_then_fleet_copy_installed(self):
        cell = Cell(self.tmp)
        name, source, target = self.stage(cell)
        target.mkdir(parents=True)
        (target / "result.json").write_text("dev original")
        out = h.install_stages([(name, source, target)], self.rescue)
        self.assertEqual(out["installed"], 1)
        self.assertTrue((target / "fleet_receipt.json").is_file())
        self.assertEqual((self.rescue / name / "result.json").read_text(), "dev original")

    def test_same_receipt_is_already_installed(self):
        cell = Cell(self.tmp)
        name, source, target = self.stage(cell)
        h.install_stages([(name, source, target)], self.rescue)
        again = self.stage(cell)
        out = h.install_stages([again], self.rescue)
        self.assertEqual((out["installed"], out["already_installed"]), (0, 1))

    def test_agreeing_duplicate_from_another_program_is_kept_out(self):
        first = Cell(self.tmp, host="h1")
        name, source, target = self.stage(first)
        h.install_stages([(name, source, target)], self.rescue)
        older = dict(KIND, program_sha="c" * 64)
        second = Cell(self.tmp, kind=older, host="other-host")  # different program + runtime fields, same science
        dup = self.stage(second)
        out = h.install_stages([dup], self.rescue)
        self.assertEqual((out["installed"], list(out["duplicates_kept_installed"])), (0, [name]))
        self.assertEqual(json.loads((target / "fleet_receipt.json").read_text())["program_sha"], PROGRAM)
        self.assertFalse((self.rescue / name).exists(), "a fleet cell must never be moved to the original-era root")

    def test_disagreeing_duplicate_aborts_before_anything_moves(self):
        first = Cell(self.tmp)
        name, source, target = self.stage(first)
        h.install_stages([(name, source, target)], self.rescue)
        other_cell = Cell(self.tmp, name="POT_s_r0_mlp32/o1_r0")
        fresh = self.stage(other_cell)
        bad = self.stage(Cell(self.tmp, kind=dict(KIND, program_sha="c" * 64), weights=b"V" * 64))
        with self.assertRaises(ValueError):
            h.install_stages([fresh, bad], self.rescue)
        self.assertFalse(fresh[2].exists(), "nothing may be installed when any duplicate disagrees")

    def test_local_original_with_existing_rescue_is_refused(self):
        cell = Cell(self.tmp)
        name, source, target = self.stage(cell)
        target.mkdir(parents=True)
        (self.rescue / name).mkdir(parents=True)
        with self.assertRaises(ValueError):
            h.install_stages([(name, source, target)], self.rescue)


class TierGuard(unittest.TestCase):
    def test_uncapped_avx512_is_refused_without_parity(self):
        with self.assertRaises(ValueError):
            exe.effective_tier(PROGRAM, environ={}, avx512f=True)
        with self.assertRaises(ValueError):
            exe.effective_tier(PROGRAM, environ={"ZENSIM_MAX_TIER": "v4"}, avx512f=True)

    def test_recorded_tiers(self):
        capped = exe.effective_tier(PROGRAM, environ={"ZENSIM_MAX_TIER": "v3"}, avx512f=True)
        self.assertEqual((capped["effective"], capped["cpu_avx512f"]), ("v3", True))
        avx2 = exe.effective_tier(PROGRAM, environ={}, avx512f=False)
        self.assertEqual(avx2["effective"], "v3")

    def test_parity_lists_agree(self):
        self.assertEqual(exe.TIER_PARITY_PROGRAMS, h.TIER_PARITY_PROGRAMS)

    def test_program_roots_agree(self):
        # The harvester must look for each fit program's cells where the executor archives them.
        self.assertEqual({name: root for name, (root, _) in exe.SCRIPTS.items()}, h.PROGRAM_ROOTS)


class FetchData(unittest.TestCase):
    """The data archive is extracted once per scratch dir, verified member by member, then dropped."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.saved_root, exe.ROOT = exe.ROOT, self.tmp / "fit-cell"
        self.saved_env = os.environ.get("ZEN_FIT_DATA_LOCAL")

    def tearDown(self):
        exe.ROOT = self.saved_root
        if self.saved_env is None:
            os.environ.pop("ZEN_FIT_DATA_LOCAL", None)
        else:
            os.environ["ZEN_FIT_DATA_LOCAL"] = self.saved_env

    def archive(self, files, inventory=None):
        inv = {"schema": "zenfleet-fit-data-v1", "files": inventory or {k: sha(v) for k, v in files.items()}}
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as z, tarfile.open(fileobj=z, mode="w") as t:
            for name, data in {"input_inventory.json": json.dumps(inv).encode(), **files}.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
        path = self.tmp / "data.tar.gz"
        path.write_bytes(buf.getvalue())
        os.environ["ZEN_FIT_DATA_LOCAL"] = str(path)
        return path, sha(buf.getvalue())

    def test_extracts_once_then_drops_and_never_needs_the_archive(self):
        path, digest = self.archive({"rev4-featpot/v2/wide/keep_lists.json": b"{}"})
        out = exe.fetch_data(digest)
        self.assertEqual((out / "rev4-featpot/v2/wide/keep_lists.json").read_bytes(), b"{}")
        self.assertFalse((exe.ROOT / f"{digest}.tar.gz").exists())
        path.unlink()  # a second cell must not fetch again
        self.assertEqual(exe.fetch_data(digest), out)

    def test_marker_for_another_archive_forces_a_fresh_extraction(self):
        path, digest = self.archive({"a.txt": b"a"})
        out = exe.fetch_data(digest)
        (out / ".verified").write_text("0" * 64 + "\n")
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            exe.fetch_data(digest)

    def test_changed_member_is_refused_and_leaves_no_marked_extraction(self):
        _, digest = self.archive({"a.txt": b"a"}, inventory={"a.txt": sha(b"b")})
        with self.assertRaises(ValueError):
            exe.fetch_data(digest)
        self.assertFalse((exe.ROOT / digest / ".verified").exists())


class EnsureLink(unittest.TestCase):
    """Binary links in the shared extraction are created by every container on a host at once."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.target = self.tmp / "bin"
        self.target.write_text("x")

    def test_link_created_by_a_concurrent_container_is_accepted(self):
        link = self.tmp / "link"
        real_symlink = Path.symlink_to

        def racing(path, target, target_is_directory=False):
            real_symlink(path, target)  # the other container wins between our check and our create
            raise FileExistsError(path)

        Path.symlink_to = racing
        try:
            exe.ensure_link(link, self.target, "binary")
        finally:
            Path.symlink_to = real_symlink
        self.assertEqual(link.resolve(), self.target.resolve())

    def test_wrong_target_and_plain_file_are_refused(self):
        other = self.tmp / "other"
        other.write_text("y")
        link = self.tmp / "link"
        link.symlink_to(other)
        with self.assertRaises(ValueError):
            exe.ensure_link(link, self.target, "binary")
        plain = self.tmp / "plain"
        plain.write_text("z")
        with self.assertRaises(ValueError):
            exe.ensure_link(plain, self.target, "binary")


class Coverage(unittest.TestCase):
    def js(self, name, declared, done=(), claims=None, drained=False):
        return {"name": name, "declared": set(declared), "done": set(done), "claims": claims or {}, "drained": drained}

    def test_complete_grid_across_jobsets(self):
        r = cov.coverage(["a/1", "a/2", "a/3"], [self.js("old", ["a/1", "a/2"], ["a/1", "a/2"]),
                                                   self.js("new", ["a/3"], ["a/3"])], {"new"}, 1000)
        self.assertTrue(r["complete"])
        self.assertEqual(r["done"], 3)

    def test_stranded_and_pending_cells_are_told_apart(self):
        old = self.js("old", ["a/1", "a/2"], ["a/1"], claims={"a/2": ("w-gone", 30000)}, drained=True)
        new = self.js("new", ["a/3"], [], claims={"a/3": ("w-live", 60)})
        r = cov.coverage(["a/1", "a/2", "a/3"], [old, new], {"new"}, 1000, live={"w-live"})
        self.assertFalse(r["complete"])
        by = {m["cell"]: m for m in r["missing"]}
        self.assertEqual(by["a/2"]["status"], "STRANDED")
        self.assertEqual(by["a/2"]["claims"]["old"], {"worker": "w-gone", "age_s": 30000, "finished": False, "live": False})
        self.assertEqual(by["a/3"]["status"], "IN_FLIGHT")
        self.assertEqual(r["stranded"], ["a/2"])

    def test_cell_declared_nowhere_is_stranded_and_outside_cells_are_reported(self):
        r = cov.coverage(["a/1", "a/2"], [self.js("s", ["a/1", "z/9"], ["a/1", "z/9"])], {"s"}, 1)
        self.assertEqual(r["stranded"], ["a/2"])
        self.assertEqual(r["done_outside_grid"], ["z/9"])
        self.assertFalse(r["complete"])

    def test_duplicates_across_jobsets_are_reported(self):
        r = cov.coverage(["a/1"], [self.js("x", ["a/1"], ["a/1"]), self.js("y", ["a/1"], ["a/1"])], {"y"}, 1)
        self.assertEqual(r["done_in_several_jobsets"], {"a/1": ["x", "y"]})
        self.assertTrue(r["complete"])

    def test_live_workers_decide_in_flight_and_stranded(self):
        old = self.js("old", ["a/1", "a/2", "a/3"], [], claims={"a/1": ("w-a", 100), "a/2": ("w-gone", 900),
                                                              "a/3": ("w-a", 5000)}, drained=True)
        new = self.js("new", ["a/2"], [], claims={})
        r = cov.coverage(["a/1", "a/2", "a/3"], [old, new], {"new"}, 1000, live={"w-a"})
        by = {m["cell"]: m for m in r["missing"]}
        # w-a holds one cell: its newest claim (a/1, age 100); the older one (a/3) is a dead incarnation's.
        self.assertEqual(by["a/1"]["status"], "IN_FLIGHT")
        self.assertEqual(by["a/3"]["status"], "STRANDED")
        self.assertFalse(by["a/3"]["claims"]["old"]["live"])
        # a/2: dead owner, but the served jobset `new` declares it, so it can be re-claimed.
        self.assertEqual(by["a/2"]["status"], "pending")
        self.assertEqual(r["in_flight"], ["a/1"])
        self.assertEqual(r["stranded"], ["a/3"])
        # Without a live list every claim counts as live.
        r = cov.coverage(["a/1", "a/2", "a/3"], [old, new], {"new"}, 1000)
        self.assertEqual({m["cell"]: m["status"] for m in r["missing"]}["a/2"], "IN_FLIGHT")

    def test_failed_cells_are_failed_until_redeclared(self):
        v8 = self.js("v8", ["a/1", "a/2", "a/3"], ["a/3"], claims={"a/1": ("w-a", 500, True), "a/2": ("w-a", 900, True)})
        v8["failed"] = {"a/1": "disk_full", "a/2": "upload_fail"}
        r = cov.coverage(["a/1", "a/2", "a/3"], [v8], {"v8"}, 1000, live={"w-a"})
        by = {m["cell"]: m for m in r["missing"]}
        # A failed cell's claim reads done/total (finished chunk): not a running fit, so both are FAILED.
        self.assertEqual(r["failed"], ["a/1", "a/2"])
        self.assertTrue(by["a/1"]["claims"]["v8"]["finished"])
        self.assertEqual(by["a/2"]["status"], "FAILED")
        self.assertEqual(by["a/2"]["failed_in"], {"v8": "upload_fail"})
        # Re-declared in a fresh jobset with no failed row: pending, then IN_FLIGHT once a live worker claims it.
        retry = self.js("retry", ["a/2"], [])
        r = cov.coverage(["a/1", "a/2", "a/3"], [v8, retry], {"v8", "retry"}, 1000, live={"w-a"})
        self.assertEqual({m["cell"]: m["status"] for m in r["missing"]}["a/2"], "pending")
        retry["claims"] = {"a/2": ("w-b", 30)}
        r = cov.coverage(["a/1", "a/2", "a/3"], [v8, retry], {"v8", "retry"}, 1000, live={"w-a", "w-b"})
        self.assertEqual({m["cell"]: m["status"] for m in r["missing"]}["a/2"], "IN_FLIGHT")
        # A failed cell in a jobset nobody serves is FAILED, not STRANDED.
        r = cov.coverage(["a/1", "a/2", "a/3"], [v8], set(), 1000, live={"w-a"})
        self.assertEqual({m["cell"]: m["status"] for m in r["missing"]}["a/2"], "FAILED")

    def test_finished_claim_without_a_done_row_is_a_lost_result(self):
        v8 = self.js("v8", ["a/1", "a/2"], ["a/2"], claims={"a/1": ("w-a", 800, True)})
        r = cov.coverage(["a/1", "a/2"], [v8], {"v8"}, 1000, live={"w-a"})
        m = r["missing"][0]
        self.assertEqual((m["cell"], m["status"]), ("a/1", "FAILED"))
        self.assertEqual(m["failed_in"], {"v8": "finished_claim_no_done_row"})
        # A running claim (not finished) is IN_FLIGHT, and a re-declaration in a fresh jobset makes it pending.
        v8["claims"] = {"a/1": ("w-a", 800, False)}
        self.assertEqual(cov.coverage(["a/1", "a/2"], [v8], {"v8"}, 1000, live={"w-a"})["missing"][0]["status"], "IN_FLIGHT")
        v8["claims"] = {"a/1": ("w-a", 800, True)}
        retry = self.js("retry", ["a/1"], [])
        r = cov.coverage(["a/1", "a/2"], [v8, retry], {"v8", "retry"}, 1000, live={"w-a"})
        self.assertEqual(r["missing"][0]["status"], "pending")

    def test_claim_parsing(self):
        self.assertEqual(cov.parse_claim("900 host-1 0/1", 1000), ("host-1", 100, False))
        self.assertEqual(cov.parse_claim("900 host-1 1/1", 1000), ("host-1", 100, True))
        self.assertEqual(cov.parse_claim("900 host-1", 1000), ("host-1", 100, False))
        self.assertEqual(cov.parse_claim("garbage", 1000), ("?", None, False))


class GoldenJobId(unittest.TestCase):
    def test_python_job_id_matches_the_rust_golden(self):
        """Same cell and id as `fit_cell_serialization_and_job_id_are_golden_stable` in zenfleet-core."""
        argv = ["mlp_probe.py", "--set", "kadid_train", "--arm", "r0", "--hidden", "32", "--outer", "0",
                "--rep", "0"]
        program = "a115996c1c535bb1cfff526cfe9ac08b0eaa89ff5842b09fae03293cd8c4eb5b"
        data = "5db2419e2bb24e6f75b000d03c99e137e17fd6a710f89c39208b793e86184af4"
        argv_sha = sha(json.dumps(argv, separators=(",", ":")).encode())
        self.assertEqual(argv_sha, "5b8e7e7f01b63a1b2d37371a8138471172e2d2ee2d74182bca7e7702be62bc23")
        job = {"kind": {"kind": "fit_cell", "program_sha": program, "data_sha": data, "argv_sha": argv_sha,
                        "argv": argv}, "inputs": [program, data, argv_sha]}
        self.assertEqual(cov.job_id(job), "64b6925760fc41fd15aca593ada2c85ba8420490ced978e98e5967e4d9d448a5")


class CellRoots(unittest.TestCase):
    """Executor and harvest agree on where a cell lives, including `--root` (v2-canon) and confirm fits."""

    CASES = [
        (["v2_lodo_mlp.py", "--spec", "r0", "--head", "N"], "v2/cells"),
        (["v2_lodo_mlp.py", "--spec", "r0", "--root", "/var/tmp/rev4-featpot/v2c"], "v2c/cells"),
        (["v2_confirm_fit.py", "--spec", "r0", "--head", "F"], "v2/confirm/cells"),
        (["v2_confirm_fit.py", "--root", "/var/tmp/rev4-featpot/v2c", "--spec", "r0"], "v2c/confirm/cells"),
        (["p2_lodo_mlp.py", "--arm", "a1", "--root", "/var/tmp/rev4-featpot/v2c"], "p2/d2_mlp"),
        (["e5_epochs.py", "cell", "--spec", "r0@h8", "--head", "N"], "e5/v2/cells"),
    ]

    def test_roots_agree(self):
        self.assertEqual(set(exe.ROOTED), set(h.ROOTED))
        for argv, want in self.CASES:
            self.assertEqual(exe.cell_root(argv), want, argv)
            self.assertEqual(h.blob_root({"argv": argv}), want, argv)

    def test_unsafe_roots_are_refused(self):
        for bad in ("/var/tmp/rev4-featpot", "/var/tmp/other/v2c", "/var/tmp/rev4-featpot/v2c/deeper",
                    "/var/tmp/rev4-featpot/..", "/var/tmp/rev4-featpot/.hidden"):
            argv = ["v2_lodo_mlp.py", "--root", bad]
            with self.assertRaises(ValueError, msg=bad):
                exe.cell_root(argv)
            with self.assertRaises(ValueError, msg=bad):
                h.blob_root({"argv": argv})


class PackDataSelect(unittest.TestCase):
    """`pack_fit_data_v2 --select` packs only the chosen variant directories, each still receipt-checked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wide = Path(self.tmp.name) / "wide"
        for family in pack.FAMILIES:
            for variant in pack.VARIANTS:
                vdir = self.wide / family / variant
                vdir.mkdir(parents=True)
                table = vdir / "kadid.parquet"
                table.write_bytes(f"{family}-{variant}".encode())
                Path(f"{table}.manifest.json").write_text("{}")
                keys = vdir / "kadid.keys.parquet"
                keys.write_bytes(b"keys")
                (vdir / "receipt.json").write_text(json.dumps({
                    "schema": "rev4-featpot-v2-wide-v2", "family": family, "variant": variant,
                    "legs": {"kadid": {"full": {"path": str(table), "sha256": pack.digest(table),
                                                "manifest_sha256": pack.digest(Path(f"{table}.manifest.json"))},
                                       "keys_sha256": pack.digest(keys)}}}))
        (self.wide / "keep_lists.json").write_text("{}")
        self.old = pack.WIDE
        pack.WIDE = self.wide

    def tearDown(self):
        pack.WIDE = self.old
        self.tmp.cleanup()

    def run_pack(self, *extra):
        out = Path(self.tmp.name) / "out.tar.gz"
        old_argv = sys.argv
        sys.argv = ["pack_fit_data_v2.py", "--out", str(out), *extra]
        try:
            pack.main()
        finally:
            sys.argv = old_argv
        with tarfile.open(out, "r:gz") as tar:
            names = tar.getnames()
            inventory = json.loads(tar.extractfile("input_inventory.json").read())
        return names, inventory

    def test_default_packs_every_variant_without_a_selection_record(self):
        names, inventory = self.run_pack()
        self.assertEqual(len([n for n in names if n.endswith("/receipt.json")]), 8)
        self.assertNotIn("variant_dirs", inventory)

    def test_select_packs_only_the_chosen_directories(self):
        names, inventory = self.run_pack("--select", "aux/p1", "--select", "main/real")
        dirs = sorted({"/".join(n.split("/")[3:5]) for n in names if n.startswith("rev4-featpot/v2/wide/") and
                       n.count("/") >= 5})
        self.assertEqual(dirs, ["aux/p1", "main/real"])
        self.assertEqual(inventory["variant_dirs"], ["main/real", "aux/p1"])
        self.assertIn("rev4-featpot/v2/wide/keep_lists.json", names)

    def test_unknown_or_repeated_selection_is_refused(self):
        with self.assertRaises(SystemExit):
            self.run_pack("--select", "aux/p9")
        with self.assertRaises(SystemExit):
            self.run_pack("--select", "aux/p1", "--select", "aux/p1")

    def test_changed_table_is_still_refused(self):
        (self.wide / "aux" / "p1" / "kadid.parquet").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.run_pack("--select", "aux/p1")


if __name__ == "__main__":
    unittest.main()
