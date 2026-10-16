import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import fit_cell_exec as executor
import fit_paths
import registered_fit_contract as contract
import build_fit_spec


class RegisteredTransport(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.path=self.root/"private-contract.json"
        qualification=dict(schema="registered-fit-contract-v1",init_seed=1101,sample_seed=101,seed_index=0,
                           selected_epoch=119,epochs=120,pairs_per_epoch=50000,execution_contract="registered-fit")
        self.cell=dict(route="loso",tier="v3",formula_revision=5,init_seed=1101,sample_seed=101,seed_index=0,
                       selected_epoch=119,epochs=120,pairs_per_epoch=50000,wall_cap_sec=9,qualification=qualification)
        self.packet=dict(schema="registered-training-cells-v1",cells={"A-h__N/loso_00_s0":self.cell})

    def save(self):
        self.path.write_text(json.dumps(self.packet))
        pin=hashlib.sha256(self.path.read_bytes()).hexdigest()
        argv=["subjective_cell.py","--contract","/unmounted/private-contract.json","--contract-sha256",pin,
              "--cell-id","A-h__N/loso_00_s0","--strict-admission","--train-only",
              "--root","/var/tmp/rev4-featpot/inputs","--dest","/var/tmp/rev4-featpot/outputs/cells/A-h__N/loso_00_s0"]
        meta=dict(registered_fit=dict(data_sha256="a"*64,contract_sha256=pin))
        return argv,meta

    def test_external_private_contract_and_finite_watchdog(self):
        argv,meta=self.save()
        with mock.patch.dict(os.environ,ZEN_REGISTERED_FIT_CONTRACT=str(self.path)):
            expected=contract.load(argv,"a"*64,meta)
            self.assertEqual(expected["wall_cap_sec"],9)
            self.assertEqual(fit_paths.explicit_root(argv,Path("/var/tmp/rev4-featpot")),"outputs/cells")
            self.assertEqual(executor.cell_root(argv),"outputs/cells")
            with self.assertRaises(ValueError): contract.load(argv,"b"*64,meta)
        self.assertNotIn(str(self.path),json.dumps(meta),"image metadata must carry only opaque pins")

    def test_no_missing_cap_or_budget_shrink(self):
        for field,value in (("wall_cap_sec",None),("epochs",1),("pairs_per_epoch",10),("tier","v4")):
            old=self.cell[field];self.cell[field]=value
            argv,meta=self.save()
            with mock.patch.dict(os.environ,ZEN_REGISTERED_FIT_CONTRACT=str(self.path)):
                with self.assertRaises(ValueError): contract.load(argv,"a"*64,meta)
            self.cell[field]=old

    def test_grid_pin_uniqueness_and_late_cell_refuse(self):
        cells=[dict(name=f"fixture/cell-{i}",argv=["subjective_cell.py","--strict-admission","--train-only",
                     "--cell-id",f"fixture/cell-{i}"]) for i in range(736)]
        path=self.root/"grid.json";path.write_text(json.dumps(dict(cells=cells)))
        pin=hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(len(build_fit_spec.grid_registered(path,pin)),736)
        cells[0]=cells[1];path.write_text(json.dumps(dict(cells=cells)))
        with self.assertRaises(ValueError): build_fit_spec.grid_registered(path,hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaises(ValueError): build_fit_spec.grid_registered(path,pin)

    def test_registered_result_uses_existing_decoded_checkpoint_gate(self):
        import qualified_fit_contract as q
        from test_strict_fit_harvest import StrictHarvest
        fixture=StrictHarvest()
        cell=fixture.cell(self.root)
        result=json.loads(cell.files["result.json"])
        result["schema"]="registered-training-cell-v1"
        result["train_weights"]={"human":32.}
        expected=fixture.contract|dict(schema="registered-fit-contract-v1",init_seed=1101,sample_seed=101,
            seed_index=0,route="kadid",execution_contract="registered-fit",
            table_admission=fixture.contract["routes"]["kadid"],train_weights={"kadid":{"human":32.}})
        inputs=dict(loss_mode="Rank",n_features=1825,rows=2,train_w=32.,val_w=0.,within_ref=True)
        expected["input_contracts"]={"kadid":{"human":inputs}}
        fixture.decoded["repro"]["inputs"][0].update(inputs)
        fixture.decoded["repro"].update(effective_minibatch=1,target_column="human_score",target_scale=1.)
        with mock.patch.object(q.subprocess,"run",side_effect=lambda *a,**k:SimpleNamespace(stdout=json.dumps(fixture.decoded))):
            self.assertEqual(q.verify_training(result,self.root/"checkpoint",cell.kind,expected,fixture.inspector),"registered-fit")
            fixture.decoded["repro"]["inputs"][0]["within_ref"]=False
            with self.assertRaisesRegex(ValueError,"roles/weights/modes"):
                q.verify_training(result,self.root/"checkpoint",cell.kind,expected,fixture.inspector)


if __name__=="__main__":unittest.main()
