"""Strict training-only output verification has explicit admission/epoch gates."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_fit_tools as fixtures
import harvest_fit_cells as h


class StrictHarvest(unittest.TestCase):
    def cell(self, root, **changes):
        name='spec__N/without_kadid_s0'
        argv=['v2_lodo_mlp.py','--root','/var/tmp/rev4-featpot/v2d1','--strict-admission','--train-only',
              '--dest',f'/var/tmp/rev4-featpot/d1-results/cells/{name}']
        kind={**fixtures.KIND,'argv':argv,'argv_sha':hashlib.sha256(json.dumps(argv,separators=(',',':')).encode()).hexdigest()}
        cell=fixtures.Cell(root,name=name,kind=kind,importance='none')
        r=json.loads(cell.files['result.json'])
        r.update(schema='rev5-qualified-training-cell-v1',training_only=True,epochs=120,
                 selection={'epoch_rule':'last','selected_epoch':119,'strict_table_admission':[{'table_sha256':'d'*64}]},
                 data_role_decision_sha256='1'*64,wide_receipt_sha256='2'*64,frozen_sha256='3'*64,
                 selected_bake=f'/var/tmp/rev4-featpot/d1-results/cells/{name}/refit/best.bin')
        r.pop('prediction');r.pop('test_rows');r.update(changes)
        cell.files['result.json']=json.dumps(r).encode()
        cell.receipt['files']['result.json']=fixtures.sha(cell.files['result.json'])
        cell.receipt['result_sha']=cell.receipt['files']['result.json']
        return cell

    def verify(self, root, cell):
        blob=cell.blob(root,rename_dir='d1-results/cells/'+cell.name)
        return h.verify_blob(blob,root/'stage',cell.name,cell.kind)

    def test_strict_training_only_accepts_complete_registered_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.verify(root,self.cell(root))
            self.assertEqual(r['schema'],'zenfleet-fit-cell-receipt-v1')

    def test_missing_admission_wrong_epoch_and_legacy_route_refuse(self):
        for changes in ({'data_role_decision_sha256':None}, {'frozen_sha256':None},
                        {'selection':{'epoch_rule':'last','selected_epoch':118,'strict_table_admission':[{}]}},
                        {'selection':{'epoch_rule':'last','selected_epoch':119,'strict_table_admission':[]}}):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)
                with self.assertRaisesRegex(ValueError,'unqualified/incomplete'):
                    self.verify(root,self.cell(root,**changes))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cell=self.cell(root);cell.kind['argv'].remove('--strict-admission')
            with self.assertRaisesRegex(ValueError,'strict training-only'):
                self.verify(root,cell)


if __name__=='__main__':
    unittest.main()
