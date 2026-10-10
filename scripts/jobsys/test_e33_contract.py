"""E33 variant packages: route-aware selection (A and full-data A share a spec) and one package per program."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from qualified_fit_contract import E33_CONTRACT, V40_CONTRACT, sha, trusted_contract

BASE = dict(schema="e33-research-fit-contract-v1", launchable=True, root="/var/tmp/rev4-featpot/v2e29",
            spec="sel:a", data_sha="b" * 64, head="N", research_hdr=False, epochs=120, pairs_per_epoch=50000,
            selected_epoch=119, fold_order=["kadid", "tid2013", "konfig", "cid22_a25"],
            init_seeds=list(range(10)), sample_seeds=list(range(101, 111)), columns=[13, 14])


def argv(script, route_flag):
    return [script, "--root", BASE["root"], "--spec", "sel:a", "--head", "N", "--strict-admission",
            "--train-only", *route_flag, "--seed-index", "0", "--columns", "13,14"]


class E33Contract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inspector = self.root / "inspector"
        self.inspector.write_bytes(b"synthetic inspector identity")
        lodo = {**BASE, "name": "a", "routes": {"kadid": []}, "requested_ids": [13, 14]}
        full = {**BASE, "name": "full-a", "routes": {"production": []}, "requested_ids": [13, 14]}
        self.variants = [lodo, full]

    def prepare(self, args, packages=(E33_CONTRACT,), schema="e33-research-fit-package-v1"):
        raw = json.dumps(dict(schema=schema, variants=self.variants)).encode()
        files = {name: hashlib.sha256(raw).hexdigest() for name in packages}
        files["bin/inspect_qualified_checkpoint"] = sha(self.inspector)
        meta = json.dumps({"files": files}).encode()
        archive = self.root / "program.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name, value in [*((p, raw) for p in packages), ("build_meta.json", meta)]:
                item = tarfile.TarInfo(name)
                item.size = len(value)
                tar.addfile(item, io.BytesIO(value))
        kind = dict(program_sha=sha(archive), data_sha="b" * 64, argv=args,
                    argv_sha=hashlib.sha256(json.dumps(args, separators=(",", ":"),
                                                       ensure_ascii=False).encode()).hexdigest())
        return kind, archive

    def test_route_selects_between_variants_sharing_a_spec(self):
        kind, archive = self.prepare(argv("v2_lodo_mlp.py", ["--heldout", "kadid"]))
        self.assertEqual(trusted_contract(kind, archive, self.inspector)["name"], "a")
        kind, archive = self.prepare(argv("v2_confirm_fit.py", ["--pack-production"]))
        contract = trusted_contract(kind, archive, self.inspector)
        self.assertEqual((contract["name"], contract["route"]), ("full-a", "production"))

    def test_two_packages_or_wrong_schema_refuse(self):
        kind, archive = self.prepare(argv("v2_lodo_mlp.py", ["--heldout", "kadid"]),
                                     packages=(V40_CONTRACT, E33_CONTRACT))
        with self.assertRaisesRegex(ValueError, "more than one"):
            trusted_contract(kind, archive, self.inspector)
        kind, archive = self.prepare(argv("v2_lodo_mlp.py", ["--heldout", "kadid"]),
                                     schema="v40-research-fit-package-v1")
        with self.assertRaisesRegex(ValueError, "unregistered variant package"):
            trusted_contract(kind, archive, self.inspector)

    def test_unknown_route_refuses(self):
        kind, archive = self.prepare(argv("v2_lodo_mlp.py", ["--heldout", "tid2013"]))
        with self.assertRaisesRegex(ValueError, "unregistered or owner-blocked"):
            trusted_contract(kind, archive, self.inspector)


if __name__ == "__main__":
    unittest.main()
