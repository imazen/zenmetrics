import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock
import pack_fit_program as pack


class RegisteredProgram(unittest.TestCase):
    def test_opaque_binding_and_public_source_inventory_only(self):
        with tempfile.TemporaryDirectory() as scratch:
            root=Path(scratch);source=root/"public-source";source.mkdir()
            paths=["scripts/rev4_featpot/"+name for name in ("subjective_cell.py","subjective_loso.py",
                "subjective_training_leg.py","v2_lodo_mlp.py","v2_common.py","v2_teacher.py","v2c_wide.py",
                "v2_human_role.py","v2_production_pack.py","rev5_bank.py","e33_fx1_declaration.json")]
            paths.extend(("scripts/lib/zen_stats.py","scripts/lib/assessment_identity.py"))
            files={}
            for name in paths:
                path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# invented public source\n")
                files[name]=pack.digest(path)
            pins=root/"pins.json";pins.write_text(json.dumps(dict(schema="registered-public-program-v1",files=files)))
            self.assertEqual(pack.registered_sources(pins,pack.digest(pins)),files)
            for omitted in ("scripts/lib/zen_stats.py","scripts/lib/assessment_identity.py",
                            "scripts/rev4_featpot/rev5_bank.py","scripts/rev4_featpot/e33_fx1_declaration.json"):
                incomplete=root/"incomplete.json"
                incomplete.write_text(json.dumps(dict(schema="registered-public-program-v1",
                    files={k:v for k,v in files.items() if k!=omitted})))
                with self.assertRaises(ValueError):pack.registered_sources(incomplete,pack.digest(incomplete))
            with self.assertRaises(ValueError):pack.registered_sources(pins,"a"*64)
            executor=root/"fit_cell_exec.py";executor.write_text("# invented executor\n")
            for name in ("fit_paths.py","registered_fit_contract.py"):(root/name).write_text("# invented public helper\n")
            binaries=root/"bin";binaries.mkdir()
            for name in (*pack.BINARIES,"inspect_qualified_checkpoint"):(binaries/name).write_bytes(b"invented binary fixture")
            metadata=root/"build-meta.json";metadata.write_text(json.dumps(dict(schema="fleet-fits-build-v1",
                registered_fit=dict(data_sha256="a"*64,contract_sha256="b"*64))))
            output=root/"program.tar.gz"
            argv=["pack_fit_program.py","--source",str(source),"--executor",str(executor),"--bin-dir",str(binaries),
                  "--build-meta",str(metadata),"--profile","registered","--source-pins",str(pins),
                  "--source-pins-sha256",pack.digest(pins),"--out",str(output)]
            with mock.patch("sys.argv",argv),mock.patch("sys.stdout",new_callable=io.StringIO):pack.main()
            with tarfile.open(output) as archive:
                self.assertNotIn("private-contract.json",archive.getnames())
                meta=json.load(archive.extractfile("build_meta.json"))
                self.assertEqual(meta["registered_fit"],dict(data_sha256="a"*64,contract_sha256="b"*64))
                self.assertIn("registered_fit_contract.py",meta["files"])


if __name__=="__main__":unittest.main()
