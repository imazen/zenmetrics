"""Manifest-bound recipe selection and budgets use synthetic archive inputs."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from qualified_fit_contract import V40_CONTRACT, trusted_contract, verify_training, sha
from unittest.mock import patch
from types import SimpleNamespace


class Contract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inspector = self.root / "inspector"
        self.inspector.write_bytes(b"synthetic inspector identity")
        self.contract = dict(
            schema="v40-research-fit-contract-v1",
            name="palette",
            launchable=True,
            root="/var/tmp/rev4-featpot/v2e32",
            spec="registered",
            data_sha="b" * 64,
            head="N",
            routes={"kadid": []},
            research_hdr=False,
            seed_index=0,
            epochs=120,
            pairs_per_epoch=50000,
            selected_epoch=119,
            fold_order=["kadid", "tid2013", "konfig", "cid22_a25"],
            init_seeds=list(range(10)),
            sample_seeds=list(range(101, 111)),
            columns=[13, 14],
        )
        self.argv = [
            "v2_lodo_mlp.py",
            "--root",
            self.contract["root"],
            "--spec",
            "registered",
            "--head",
            "N",
            "--strict-admission",
            "--train-only",
            "--heldout",
            "kadid",
            "--seed-index",
            "0",
            "--columns",
            "13,14",
        ]

    def prepare(self, argv=None, contract=None, corrupt_inventory=False):
        raw = json.dumps(
            dict(
                schema="v40-research-fit-package-v1",
                variants=[contract or self.contract],
            )
        ).encode()
        meta = json.dumps(
            {
                "files": {
                    V40_CONTRACT: "0" * 64
                    if corrupt_inventory
                    else hashlib.sha256(raw).hexdigest(),
                    "bin/inspect_qualified_checkpoint": sha(self.inspector),
                }
            }
        ).encode()
        archive = self.root / "program.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name, value in [(V40_CONTRACT, raw), ("build_meta.json", meta)]:
                item = tarfile.TarInfo(name)
                item.size = len(value)
                tar.addfile(item, io.BytesIO(value))
        argv = argv or self.argv
        kind = dict(
            program_sha=sha(archive),
            data_sha="b" * 64,
            argv=argv,
            argv_sha=hashlib.sha256(
                json.dumps(argv, separators=(",", ":"), ensure_ascii=False).encode()
            ).hexdigest(),
        )
        return kind, archive

    def test_registered_and_local_smoke_are_distinct(self):
        kind, archive = self.prepare()
        result = trusted_contract(kind, archive, self.inspector)
        self.assertEqual(result["selected_epoch"], 119)
        self.assertEqual(result["execution_contract"], "registered-fit")
        kind, archive = self.prepare(self.argv + ["--local-smoke-budget", "2:128"])
        result = trusted_contract(kind, archive, self.inspector)
        self.assertEqual(result["selected_epoch"], 1)
        self.assertEqual(result["execution_contract"], "local-smoke")

    def test_foreign_root_ids_grid_or_budget_refuse(self):
        for flag, value in [
            ("--root", "/var/tmp/rev4-featpot/other"),
            ("--columns", "13,15"),
            ("--seed-index", "10"),
            ("--heldout", "aic3"),
        ]:
            with self.subTest(flag=flag):
                argv = self.argv[:]
                argv[argv.index(flag) + 1] = value
                kind, archive = self.prepare(argv)
                with self.assertRaises(ValueError):
                    trusted_contract(kind, archive, self.inspector)
        for budget in ["120:128", "2:50000", "0:1"]:
            kind, archive = self.prepare(self.argv + ["--local-smoke-budget", budget])
            with self.assertRaises(ValueError):
                trusted_contract(kind, archive, self.inspector)

    def test_upiq_requires_exact_fit_and_owner_decision_argv(self):
        contract = {**self.contract, "research_upiq": True}
        argv = self.argv + ["--upiq380-fit", "/var/tmp/rev4-featpot/upiq380-fit/upiq380_fit.parquet",
            "--upiq-label-disposition", "/var/tmp/rev4-featpot/upiq380-fit/owner_disposition.json"]
        kind, archive = self.prepare(argv, contract)
        result = trusted_contract(kind, archive, self.inspector)
        self.assertIsNone(result["feature_set_id"])
        for flag in ("--upiq380-fit", "--upiq-label-disposition"):
            bad = argv[:]
            bad[bad.index(flag) + 1] = "/var/tmp/rev4-featpot/upiq380-fit/development.parquet"
            kind, archive = self.prepare(bad, contract)
            with self.assertRaisesRegex(ValueError, "binding differs"):
                trusted_contract(kind, archive, self.inspector)

    def test_native_fit_disposition_and_checkpoint_are_verified(self):
        native = dict(name="upiq380", table_sha256="a" * 64,
            manifest_sha256="b" * 64, keys_sha256="c" * 64,
            label_source={"original": "registered"}, label_disposition={"state": "approved"},
            label_disposition_sha256="d" * 64)
        expected = dict(schema="v40-research-fit-contract-v1", research_upiq=True,
            feature_set_id=None, columns=[13, 14], route="kadid", epochs=2,
            pairs_per_epoch=128, selected_epoch=1, seed_index=0,
            wide_receipt_sha256="wide", frozen_sha256="frozen", data_role_decision_sha256="D1",
            execution_contract="local-smoke", init_seed=1101, sample_seed=101,
            upiq_admission=native, table_admission=[native],
            input_contracts={"kadid": {"upiq380": dict(loss_mode="Rank", n_features=1853,
                rows=330, train_w=4.34410740924913, val_w=0.0, within_ref=False)}},
            train_weights={"kadid": {"hdr": 4.34410740924913}})
        result = {k: expected[k] for k in ("epochs", "pairs_per_epoch", "seed_index",
            "wide_receipt_sha256", "frozen_sha256", "data_role_decision_sha256", "execution_contract")}
        result.update(schema="e31-native-hdr-research-training-cell-v1", training_only=True,
            train_weights=expected["train_weights"]["kadid"], selection=dict(epoch_rule="last",
                selected_epoch=1, strict_table_admission=[native], upiq380_fit_admission=native))
        inp = {**expected["input_contracts"]["kadid"]["upiq380"], "name":"upiq380",
            "path":"/data/fit.parquet", "sha256": native["table_sha256"]}
        receipt = {**native, "path":inp["path"], "selected_ids":[13,14],
            "native_width":1825, "logical_width":1853, "input_contract":"upiq-exr-bt709-nits-v1"}
        repro = dict(checkpoint_epoch="001", epochs=2, requested_epochs=2, pairs_per_epoch=128,
            pair_sampling="uniform", init_seed=1101, sample_seed=101, inputs=[inp],
            effective_minibatch=1, target_column="human_score", target_scale=1.0,
            table_admission=dict(qualified_provenance=False, formula_revision=5,
                tables=[], upiq380=receipt))
        decoded = dict(formula_revision=5, qualified_provenance=False, feature_set_id=None, repro=repro)
        kind = dict(argv=["--strict-admission", "--train-only"], data_sha="e"*64)
        with patch('qualified_fit_contract.subprocess.run', return_value=SimpleNamespace(stdout=json.dumps(decoded))):
            self.assertEqual(verify_training(result, self.root/'checkpoint', kind, expected, self.inspector), "local-smoke")
        for key, value in (("label_disposition_sha256", "0"*64), ("keys_sha256", "0"*64),
                ("selected_ids", [13]), ("native_width", 1853), ("path", "/data/development.parquet")):
            bad = copy.deepcopy(decoded)
            bad["repro"]["table_admission"]["upiq380"][key] = value
            with self.subTest(key=key), patch('qualified_fit_contract.subprocess.run', return_value=SimpleNamespace(stdout=json.dumps(bad))):
                with self.assertRaises(ValueError):
                    verify_training(result, self.root/'checkpoint', kind, expected, self.inspector)

    def test_owner_block_and_contract_inventory_cannot_be_relabelled(self):
        blocked = copy.deepcopy(self.contract)
        blocked["launchable"] = False
        kind, archive = self.prepare(contract=blocked)
        with self.assertRaisesRegex(ValueError, "owner-blocked"):
            trusted_contract(kind, archive, self.inspector)
        kind, archive = self.prepare(corrupt_inventory=True)
        with self.assertRaisesRegex(ValueError, "inventory"):
            trusted_contract(kind, archive, self.inspector)


if __name__ == "__main__":
    unittest.main()
