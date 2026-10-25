"""Fail-closed by-construction transport controls; no training or label reads."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
import sys
import os
import subprocess
from unittest.mock import Mock, patch

import byc_fit_contract as c
import fit_cell_exec as executor
import harvest_fit_cells as harvest
import build_fit_spec as spec


ARGV = ["byc_cells.py", "run", "--arm", "mono:thr:mp32", "--label", "Mw100",
        "--folds", "kadid", "--seeds", "0", "--jobs", "1"]


class BycTransport(unittest.TestCase):
    def test_program_caches_refuse_equal_size_fixed_mtime_stale_code(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(executor, "ROOT", Path(tmp)):
            source = Path(tmp) / "cache_probe.py"
            code = f"import sys; sys.path.insert(0, {tmp!r}); import cache_probe; print(cache_probe.value)"
            def run(program):
                env = dict(os.environ, PYTHONPYCACHEPREFIX=str(executor.bytecode_cache(program)))
                env.pop("PYTHONDONTWRITEBYTECODE", None)
                return subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip()
            source.write_text("value = 1\n")
            os.utime(source, (0, 0))
            self.assertEqual(run("a" * 64), "1")
            source.write_text("value = 2\n")
            os.utime(source, (0, 0))
            self.assertEqual(run("a" * 64), "1", "negative control must expose the timestamp cache collision")
            self.assertEqual(run("b" * 64), "2", "a new program must execute its own source bytes")

    def test_declaration_uses_canonical_cell_and_refuses_started_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ["build_fit_spec.py", "--grid", "byc", "--program-sha", "a" * 64,
                    "--data-sha", "b" * 64, "--byc-label", "Mw100", "--byc-folds", "kadid,pairs",
                    "--byc-seeds", "0,1,2", "--existing-root", str(root / "cells"),
                    "--out", str(root / "spec.json")]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                spec.main()
            cells = json.loads((root / "spec.json").read_text())["cells"]
            self.assertEqual(len(cells), 6)
            self.assertEqual(cells[0], dict(name="Mw100/without_kadid_s0", argv=ARGV))
            (root / "cells/Mw100/without_pairs_s1").mkdir(parents=True)
            with patch.object(sys, "argv", args), self.assertRaisesRegex(ValueError, "already started"):
                spec.main()

    def test_bound_root_alias_and_thread_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            member = "rev4-featpot/byc/inputs/fit"
            (data / member).mkdir(parents=True)
            root = str(data / "root-alias")
            # Only the two approved runtime namespaces may be bound.
            binding = dict(schema="byc-fit-input-v1", label="Mw100", root=root,
                           score_root=root, bindings={root: member})
            (data / "input_inventory.json").write_text(json.dumps(dict(byc=binding)))
            link = Mock()
            with self.assertRaisesRegex(ValueError, "unsafe"):
                c.environment(ARGV, data, data / "program", data / "fit", link)
            link.assert_not_called()
            root = "/var/tmp/byc-test/root"
            binding.update(root=root, score_root=root, bindings={root: member})
            (data / "input_inventory.json").write_text(json.dumps(dict(byc=binding)))
            with patch.object(Path, "mkdir"):
                env = c.environment(ARGV, data, data / "program", data / "fit", link)
            self.assertEqual(env["BYC_ROOT"], root)
            self.assertEqual(env["RAYON_NUM_THREADS"], "1")
            self.assertEqual(env["ZENSIM_MAX_TIER"], "v3")
            link.assert_called_once()

    def test_one_cell_only(self):
        self.assertEqual(c.identity(ARGV), ("Mw100", "kadid", 0))
        for flag, value in (("--folds", "kadid,tid2013"), ("--seeds", "0,1"),
                            ("--jobs", "2"), ("--label", "../Mw100"), ("--arm", "mono")):
            bad = ARGV.copy()
            bad[bad.index(flag) + 1] = value
            with self.assertRaises(ValueError):
                c.identity(bad)
        with self.assertRaises(ValueError):
            c.identity(ARGV + ["--seeds", "1"])

    def test_shared_archive_layout(self):
        kind = {"argv": ARGV}
        self.assertEqual(executor.cell_root(ARGV), "byc/cells")
        self.assertEqual(harvest.blob_root(kind), "byc/cells")

    def test_output_prediction_and_epoch_tampering(self):
        def sha(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "Mw100/without_kadid_s0"
            (dest / "refit").mkdir(parents=True)
            (dest / "refit/last.bin").write_bytes(b"synthetic checkpoint")
            (dest / "pred_kadid.tsv").write_text("row_idx\tpred\n0\t1\n")
            record = dict(fold="kadid", seed=0, tokens="mono:thr:mp32",
                          last_bin_sha256=sha(dest / "refit/last.bin"),
                          pred_sha256=sha(dest / "pred_kadid.tsv"))
            result = dict(epochs=120, pairs_per_epoch=50000, seed_index=0,
                          selection=dict(selected_epoch=119))
            for name, value in (("byc_record.json", record), ("result.json", result),
                                ("LABEL_READS.json", {}), ("DATASET_GROUP_SELECTION.json", {})):
                (dest / name).write_text(json.dumps(value))
            c.verify_output(dest, ARGV, sha)
            (dest / "pred_kadid.tsv").write_text("row_idx\tpred\n0\t2\n")
            with self.assertRaisesRegex(ValueError, "prediction"):
                c.verify_output(dest, ARGV, sha)
            result["selection"]["selected_epoch"] = 118
            (dest / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "scientific"):
                c.verify_output(dest, ARGV, sha)

    def test_archive_harvest_verifies_existing_control_without_replacing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "Mw100/without_kadid_s0"
            pot = root / "pot"
            dest = pot / "byc/cells" / name
            (dest / "refit").mkdir(parents=True)
            bake = dest / "refit/last.bin"
            bake.write_bytes(b"synthetic checkpoint")
            pred = dest / "pred_kadid.tsv"
            pred.write_text("row_idx\tpred\n0\t1\n")
            record = dict(fold="kadid", seed=0, tokens="mono:thr:mp32",
                          last_bin_sha256=harvest.digest(bake), pred_sha256=harvest.digest(pred))
            result = dict(epochs=120, pairs_per_epoch=50000, seed_index=0,
                          selection=dict(selected_epoch=119), selected_bake=str(bake),
                          selected_bake_sha256=harvest.digest(bake))
            for filename, value in (("byc_record.json", record), ("result.json", result),
                                    ("LABEL_READS.json", {}), ("DATASET_GROUP_SELECTION.json", {})):
                (dest / filename).write_text(json.dumps(value))
            kind = dict(kind="fit_cell", argv=ARGV, program_sha="a" * 64,
                        data_sha="b" * 64, argv_sha="c" * 64)
            files = {str(p.relative_to(dest)): harvest.digest(p) for p in dest.rglob("*") if p.is_file()}
            receipt = dict(schema="zenfleet-fit-cell-receipt-v1", cell=name,
                           program_sha=kind["program_sha"], data_sha=kind["data_sha"],
                           argv_sha=kind["argv_sha"], result_sha=files["result.json"],
                           selected_bake_sha=harvest.digest(bake), files=files,
                           tier=dict(effective="v3"))
            blob = root / "blob.tar.gz"
            with patch.object(executor, "FIT_ROOT", pot):
                blob.write_bytes(executor.output_bytes(dest, receipt))
            cells = root / "originals"
            original = cells / name
            original.mkdir(parents=True)
            marker = original / "keep.txt"
            marker.write_text("local original")
            manifest, ids = root / "manifest.json", root / "ids.tsv"
            manifest.write_text(json.dumps([dict(cell=dict(image_path=name), kind=kind)]))
            ids.write_text(f"0\tjob1\t{name}\n")
            args = ["harvest_fit_cells.py", "--manifest", str(manifest), "--ids", str(ids),
                    "--ledger", str(root / "ledger"), "--blobs-prefix", "unused", "--endpoint", "unused",
                    "--scratch", str(root / "stage"), "--byc-install-root", str(cells)]
            with patch.object(harvest, "POT_ROOT", pot), patch.object(sys, "argv", args), \
                    patch.object(harvest, "rows_from_ledger", return_value={"job1": dict(output_sha="d" * 64)}), \
                    patch.object(harvest, "duplicate_done_rows", return_value={}), \
                    patch.object(harvest, "fetch_blob", return_value=blob), contextlib.redirect_stdout(io.StringIO()):
                harvest.main()
                self.assertEqual(marker.read_text(), "local original")
                args.append("--install")
                with self.assertRaisesRegex(ValueError, "refusing replacement"):
                    harvest.main()


if __name__ == "__main__":
    unittest.main()
