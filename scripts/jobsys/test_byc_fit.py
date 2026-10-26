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
import byc_harvest_receipt as summary
from byc_linked_receipt import check_linked_cell


ARGV = ["byc_cells.py", "run", "--arm", "mono:thr:mp32", "--label", "Mw100",
        "--folds", "kadid", "--seeds", "0", "--jobs", "1"]


class BycTransport(unittest.TestCase):
    def test_linked_harvest_rejects_retained_family_and_wrong_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';dest=Path(tmp)/'cell'
            (root/'datasets').mkdir(parents=True);(root/'wide').mkdir();dest.mkdir()
            def write(path,value):path.write_text(json.dumps(value))
            authority=root/'datasets/linked-subset-omissions.json'
            write(authority,dict(labels_read=False,folds=dict(mos7=dict(heldout_collections=['mdid'],excluded_subset_collections=['cognitive']))))
            pin=harvest.digest(authority)
            packet=root/'datasets/receipt.json'
            write(packet,dict(groups=[dict(collection=c) for c in ['pipal','mdid','cognitive','bapps']],linked_subset_omissions=dict(path='datasets/linked-subset-omissions.json',sha256=pin)))
            packet_sha=harvest.digest(packet)
            write(root/'wide/frozen.json',dict(dataset_legs_sha256=packet_sha,auxiliary_files={'datasets/linked-subset-omissions.json':pin}))
            frozen=harvest.digest(root/'wide/frozen.json')
            record=dict(fold='mos7',linked_subset_omission_sha256=pin,dataset_exclusions=['mdid','cognitive'])
            selected=dict(root_sha256=frozen,receipt_sha256=packet_sha,requested_exclusions=['mdid','cognitive'],omitted_collections=['mdid','cognitive'],retained_collections=['pipal','bapps'])
            write(dest/'byc_record.json',record);write(dest/'result.json',dict(frozen_sha256=frozen));write(dest/'DATASET_GROUP_SELECTION.json',selected)
            self.assertEqual(check_linked_cell(dest,root,frozen,harvest.digest)['retained_groups'],2)
            selected['retained_collections'].append('cognitive');write(dest/'DATASET_GROUP_SELECTION.json',selected)
            with self.assertRaisesRegex(ValueError,'linked exclusions'):check_linked_cell(dest,root,frozen,harvest.digest)
            selected['retained_collections'].pop();write(dest/'DATASET_GROUP_SELECTION.json',selected)
            record['dataset_exclusions']=[];write(dest/'byc_record.json',record)
            with self.assertRaisesRegex(ValueError,'linked exclusions'):check_linked_cell(dest,root,frozen,harvest.digest)
            with self.assertRaisesRegex(ValueError,'registered pin'):check_linked_cell(dest,root,'f'*64,harvest.digest)

    def test_partial_blob_verification_and_installation_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ["Mw100/without_kadid_s0", "Mw100/without_tid2013_s0"]
            manifest, ids = root / "manifest.json", root / "ids.tsv"
            manifest.write_text(json.dumps([dict(cell=dict(image_path=n), kind=dict(kind="fit_cell")) for n in names]))
            ids.write_text("".join(f"{i}\tjob{i}\t{n}\n" for i, n in enumerate(names)))
            args = ["harvest_fit_cells.py", "--manifest", str(manifest), "--ids", str(ids),
                    "--ledger", str(root / "ledger"), "--blobs-prefix", "unused", "--endpoint", "unused",
                    "--scratch", str(root / "stage")]
            rows = {"job0": dict(output_sha="d" * 64)}
            with patch.object(sys, "argv", args), patch.object(harvest, "rows_from_ledger", return_value=rows), \
                    patch.object(harvest, "duplicate_done_rows", return_value={}), \
                    patch.object(harvest, "fetch_blob", return_value=root / "blob") as fetch, \
                    patch.object(harvest, "verify_blob", return_value=dict(tier_status="v3")) as verify, \
                    patch.object(harvest, "cell_dir", return_value=root / "cell"), \
                    patch.object(harvest, "blob_root", return_value="byc/cells"), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaisesRegex(ValueError, "no DONE output"):
                    harvest.main()
                fetch.assert_not_called()
                args.append("--done-only")
                harvest.main()
                self.assertEqual((json.loads(output.getvalue())["verified"], len(names)), (1, 2))
                self.assertEqual(verify.call_args.args[2], names[0])
                rows.clear()
                with self.assertRaisesRegex(ValueError, "no DONE outputs to verify"):
                    harvest.main()
                args.append("--install")
                rows["job0"] = dict(output_sha="d" * 64)
                with patch.object(harvest, "install_stages", return_value={"installed": 1}) as install:
                    harvest.main()
                    self.assertEqual(len(install.call_args.args[0]), 1)
                    self.assertEqual(install.call_args.args[0][0][0], names[0])

    def test_input_staging_reuses_matching_objects_and_refuses_corruption(self):
        script = Path(__file__).with_name("stage_fit_inputs.sh")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "input"
            artifact.write_bytes(b"frozen input\n")
            stub = root / "aws"
            stub.write_text("""#!/usr/bin/env python3
import os, sys
from pathlib import Path
root = Path(os.environ['STUB_ROOT'])
mode = os.environ['STUB_MODE']
if 'head-object' in sys.argv:
    if mode == 'missing' and not (root/'uploaded').exists():
        print('An error occurred (404): Not Found', file=sys.stderr)
        sys.exit(1)
    if mode == 'denied':
        print('An error occurred (403): Forbidden', file=sys.stderr)
        sys.exit(1)
    pin = '0'*64 if mode == 'corrupt' else os.environ['STUB_PIN']
    size = '1' if mode == 'wrong_size' else os.environ['STUB_BYTES']
    print(pin + '\\t' + size)
elif 'cp' in sys.argv:
    (root/'uploaded').write_text('uploaded')
else:
    sys.exit(2)
""")
            stub.chmod(0o755)
            env = dict(os.environ, PATH=f"{tmp}:{os.environ['PATH']}", EP="stub",
                       STUB_ROOT=tmp, STUB_PIN=harvest.digest(artifact),
                       STUB_BYTES=str(artifact.stat().st_size))
            for mode in ("existing", "missing", "corrupt", "wrong_size", "denied"):
                (root / "uploaded").unlink(missing_ok=True)
                result = subprocess.run(["bash", str(script), "run", str(artifact)],
                                        env=dict(env, STUB_MODE=mode), capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, mode in ("existing", "missing"),
                                 (mode, result.stdout, result.stderr))
                self.assertEqual((root / "uploaded").exists(), mode == "missing")

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
        plus = ARGV.copy()
        plus[plus.index("--label") + 1] = "Mlic+sub"
        self.assertEqual(c.identity(plus), ("Mlic+sub", "kadid", 0))
        for flag, value in (("--folds", "kadid,tid2013"), ("--seeds", "0,1"),
                            ("--jobs", "2"), ("--label", "../Mw100"), ("--arm", "mono"),
                            ("--label", "Mlic+sub/../../other"), ("--label", "Mlic;false")):
            bad = ARGV.copy()
            bad[bad.index(flag) + 1] = value
            with self.assertRaises(ValueError):
                c.identity(bad)
        with self.assertRaises(ValueError):
            c.identity(ARGV + ["--seeds", "1"])

    def test_mp64_declaration_keeps_explicit_recipe_and_input_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ["build_fit_spec.py", "--grid", "byc", "--program-sha", "a" * 64,
                    "--data-sha", "b" * 64, "--byc-label", "Mlic64", "--byc-arm", "mono:thr:mp64",
                    "--byc-folds", "pipal,mos7,pairs", "--byc-seeds", "0,1,2",
                    "--existing-root", str(root / "cells"), "--out", str(root / "spec.json")]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                spec.main()
            cells = json.loads((root / "spec.json").read_text())["cells"]
            self.assertEqual(len(cells), 9)
            for cell in cells:
                argv = cell["argv"]
                self.assertEqual(argv[argv.index("--arm") + 1], "mono:thr:mp64")
                self.assertEqual(c.identity(argv)[0], "Mlic64")
            args[args.index("--byc-label") + 1] = "Mlic+sub"
            args[args.index("--byc-arm") + 1] = "mono:thr:mp32"
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                spec.main()
            cells = json.loads((root / "spec.json").read_text())["cells"]
            self.assertEqual(len(cells), 9)
            self.assertEqual(cells[0]["name"], "Mlic+sub/without_pipal_s0")
            for cell in cells:
                self.assertEqual(c.identity(cell["argv"])[0], "Mlic+sub")

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
            result = dict(epochs=120, pairs_per_epoch=50000, seed_index=0, training_only=True,
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
                self.assertEqual(bake.read_bytes(), b"synthetic checkpoint")
                self.assertTrue((dest / "fleet_receipt.json").is_file())
                with self.assertRaisesRegex(FileExistsError, "preserve the earlier candidate"):
                    executor.run_fit_locked({}, kind, "e" * 64, {}, ARGV, "c" * 64,
                                            Path(name), None, dest)
                self.assertEqual(bake.read_bytes(), b"synthetic checkpoint")
            cells = root / "originals"
            original = cells / name
            original.mkdir(parents=True)
            marker = original / "keep.txt"
            marker.write_text("local original")
            manifest, ids = root / "manifest.json", root / "ids.tsv"
            manifest.write_text(json.dumps([dict(cell=dict(image_path=name), kind=kind)]))
            ids.write_text(f"0\tjob1\t{name}\n")
            (root / "ledger").write_bytes(b"synthetic ledger")
            args = ["harvest_fit_cells.py", "--manifest", str(manifest), "--ids", str(ids),
                    "--ledger", str(root / "ledger"), "--blobs-prefix", "unused", "--endpoint", "unused",
                    "--scratch", str(root / "stage"), "--byc-install-root", str(cells)]
            with patch.object(harvest, "POT_ROOT", pot), patch.object(sys, "argv", args), \
                    patch.object(harvest, "rows_from_ledger", return_value={"job1": dict(output_sha="d" * 64)}), \
                    patch.object(harvest, "duplicate_done_rows", return_value={}), \
                    patch.object(harvest, "fetch_blob", return_value=blob), contextlib.redirect_stdout(io.StringIO()):
                harvest.main()
                self.assertEqual(marker.read_text(), "local original")
                summary_args = ["byc_harvest_receipt.py", "--cells", str(root / "stage/staged/job1/byc/cells"),
                                "--manifest", str(manifest), "--ids", str(ids), "--ledger", str(root / "ledger"),
                                "--out", str(root / "receipt.json")]
                with patch.object(sys, "argv", summary_args), \
                        patch.object(summary, "rows_from_ledger", return_value={"job1": dict(output_sha="d" * 64)}):
                    summary.main()
                    retained = json.loads((root / "receipt.json").read_text())
                    self.assertEqual(retained["verified"], 1)
                    self.assertEqual(retained["cells"][0]["refit_files"], {"refit/last.bin": receipt["selected_bake_sha"]})
                    with self.assertRaisesRegex(ValueError, "preserve the previous harvest receipt"):
                        summary.main()
                    summary_args[-1] = str(root / "bad-receipt.json")
                    (root / "stage/staged/job1/byc/cells" / name / "refit/last.bin").write_bytes(b"changed")
                    with self.assertRaisesRegex(ValueError, "retained cell files differ"):
                        summary.main()
                    self.assertFalse((root / "bad-receipt.json").exists())
                with self.assertRaisesRegex(ValueError, "preserve prior byc verification"):
                    harvest.main()
                self.assertTrue((root / "stage/staged/job1/byc/cells" / name / "refit/last.bin").is_file())
                args[args.index("--scratch") + 1] = str(root / "stage-install")
                args.append("--install")
                with self.assertRaisesRegex(ValueError, "refusing replacement"):
                    harvest.main()
                # A separate study retains the registered study's canonical path.
                args[args.index("--scratch") + 1] = str(root / "stage-separate")
                args[args.index("--byc-install-root") + 1] = str(root / "finalists/cells")
                args.append("--no-runtime-alias")
                original_bake = b"synthetic checkpoint"
                self.assertEqual(bake.read_bytes(), original_bake)
                harvest.main()
                installed = root / "finalists/cells" / name
                self.assertEqual((installed / "refit/last.bin").read_bytes(), original_bake)
                self.assertEqual(bake.read_bytes(), original_bake)
                self.assertEqual(marker.read_text(), "local original")


if __name__ == "__main__":
    unittest.main()
