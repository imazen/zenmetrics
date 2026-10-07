"""Manifest-bound recipe selection and budgets use synthetic archive inputs."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from qualified_fit_contract import V40_CONTRACT, trusted_contract, sha


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
