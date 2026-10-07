"""Trusted manifest/program budget and actual decoded checkpoint must agree."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fit_tools as fixtures
import harvest_fit_cells as h
import qualified_fit_contract as q


class StrictHarvest(unittest.TestCase):
    def cell(self, root, **changes):
        name = 'spec__N/without_kadid_s0'
        argv = ['v2_lodo_mlp.py','--root','/var/tmp/rev4-featpot/v2d1','--strict-admission','--train-only',
                '--spec','spec','--head','N','--seed-index','0','--heldout','kadid','--columns','13,14',
                '--dest',f'/var/tmp/rev4-featpot/d1-results/cells/{name}']
        kind = {**fixtures.KIND,'argv':argv,
                'argv_sha':hashlib.sha256(json.dumps(argv,separators=(',',':')).encode()).hexdigest()}
        cell = fixtures.Cell(root,name=name,kind=kind,importance='none')
        table = {'name':'human','table_sha256':'d'*64,'manifest_sha256':'e'*64,
                 'keys_sha256':'f'*64,'row_keys_sha256':'a'*64,'row_selection_sha256':'b'*64,
                 'data_role_decision_sha256':'1'*64}
        declaration = {'feature_set_id':'test-rev5','formula_revision':5,'decoder_era':'test-decoder'}
        self.contract = {'schema':'shippath-qualified-fit-contract-v1','data_sha':kind['data_sha'],
            'epochs':120,'pairs_per_epoch':50000,'selected_epoch':119,'spec':'spec','head':'N',
            'columns':[13,14], 'init_seeds':[1101]*10,'sample_seeds':[101]*10,'fold_order':['kadid'],
            'wide_receipt_sha256':'2'*64,'frozen_sha256':'3'*64,'data_role_decision_sha256':'1'*64,
            'feature_set_id':'test-rev5','table_declaration':declaration,'routes':{'kadid':[table]}}
        r = json.loads(cell.files['result.json'])
        r.update(schema='rev5-qualified-training-cell-v1',training_only=True,execution_contract='registered-fit',
                 epochs=120,pairs_per_epoch=50000,seed_index=0,
                 selection={'epoch_rule':'last','selected_epoch':119,'strict_table_admission':[table]},
                 data_role_decision_sha256='1'*64,wide_receipt_sha256='2'*64,frozen_sha256='3'*64,
                 selected_bake=f'/var/tmp/rev4-featpot/d1-results/cells/{name}/refit/best.bin')
        r.pop('prediction'); r.pop('test_rows'); r.update(changes)
        self.decoded = {'formula_revision':5,'qualified_provenance':True,'feature_set_id':'test-rev5',
            'repro':{'checkpoint_epoch':'119','epochs':120,'requested_epochs':120,'pairs_per_epoch':50000,
                'pair_sampling':'uniform','init_seed':1101,'sample_seed':101,
                'inputs':[{'name':'human','path':'/input/human','sha256':'d'*64}],
                'table_admission':{'qualified_provenance':True,'historical_replay':None,'formula_revision':5,
                    'tables':[{'path':'/input/human','inferred':False,'feature_set_id':'test-rev5',
                               'stored_declarations':declaration,'requested_ids':[13,14]}]}}}
        self.inspector = root/'inspector'; self.inspector.write_bytes(b'synthetic decoder test double')
        self.program = root/'program.tar.gz'
        with tarfile.open(self.program,'w:gz') as tar:
            for name, value in [(q.CONTRACT,self.contract),('build_meta.json',{'files':{
                    'bin/inspect_qualified_checkpoint':q.sha(self.inspector)}})]:
                raw=json.dumps(value).encode(); member=tarfile.TarInfo(name);member.size=len(raw)
                tar.addfile(member,io.BytesIO(raw))
        cell.kind['program_sha']=q.sha(self.program)
        cell.receipt['program_sha']=cell.kind['program_sha']
        cell.files['result.json']=json.dumps(r).encode()
        cell.receipt['files']['result.json']=fixtures.sha(cell.files['result.json'])
        cell.receipt['result_sha']=cell.receipt['files']['result.json']
        return cell

    def verify(self, root, cell, **kw):
        blob=cell.blob(root,rename_dir='d1-results/cells/'+cell.name)
        with patch.object(q.subprocess,'run',return_value=type('Decoded',(),{'stdout':json.dumps(self.decoded)})()):
            return h.verify_blob(blob,root/'stage',cell.name,cell.kind,program_archive=self.program,
                                 checkpoint_inspector=self.inspector,**kw)

    def test_accepts_complete_registered_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.verify(root,self.cell(root));self.assertEqual(r['execution_contract'],'registered-fit')

    def test_missing_admission_wrong_epoch_and_legacy_refuse(self):
        for changes in ({'data_role_decision_sha256':None},{'frozen_sha256':None},
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

    def test_short_smoke_under_full_job_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(ValueError,'epochs differs'):
                self.verify(root,self.cell(root,epochs=2,pairs_per_epoch=128,
                    selection={'epoch_rule':'last','selected_epoch':1,'strict_table_admission':[]}))

    def test_epoch119_claim_with_epoch1_checkpoint_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cell=self.cell(root)
            self.decoded['repro'].update(checkpoint_epoch='001',epochs=2,requested_epochs=2,pairs_per_epoch=128)
            with self.assertRaisesRegex(ValueError,'decoded checkpoint budget'):
                self.verify(root,cell)

    def test_zeroed_admission_hashes_refuse(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(ValueError,'differs from manifest-bound'):
                self.verify(root,self.cell(root,data_role_decision_sha256='0'*64,
                                          frozen_sha256='0'*64,wide_receipt_sha256='0'*64))

    def test_wrong_checkpoint_input_or_decoder_refuses(self):
        for bad in ('input','decoder'):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);cell=self.cell(root)
                if bad=='input':self.decoded['repro']['inputs'][0]['sha256']='0'*64
                else:self.decoded['repro']['table_admission']['tables'][0]['stored_declarations']['decoder_era']='other'
                with self.assertRaisesRegex(ValueError,'decoded'):
                    self.verify(root,cell)

    def test_executor_lexical_admission_and_hash_bound_physical_input(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cell=self.cell(root)
            self.decoded['repro']['inputs'][0]['path']='/scratch/fit-cell/'+cell.kind['data_sha']+'/rev4-featpot/v2d1/human.parquet'
            self.decoded['repro']['table_admission']['tables'][0]['path']='/var/tmp/rev4-featpot/v2d1/human.parquet'
            self.verify(root,cell)
            self.decoded['repro']['inputs'][0]['path']='/scratch/fit-cell/'+('0'*64)+'/rev4-featpot/v2d1/human.parquet'
            with self.assertRaisesRegex(ValueError,'decoded admissions'):
                self.verify(root,cell)

    def test_missing_or_wrong_trusted_program_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cell=self.cell(root)
            with self.assertRaisesRegex(ValueError,'trusted program archive'):
                h.verify_blob(cell.blob(root,rename_dir='d1-results/cells/'+cell.name),root/'stage',cell.name,cell.kind)
            self.program.write_bytes(b'wrong archive')
            with self.assertRaisesRegex(ValueError,'trusted program SHA'):
                self.verify(root,cell)

    def test_explicit_smoke_is_verify_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cell=self.cell(root,epochs=2,pairs_per_epoch=128,execution_contract='local-smoke')
            cell.kind['argv'] += ['--local-smoke-budget','2:128']
            cell.kind['argv_sha']=hashlib.sha256(json.dumps(cell.kind['argv'],separators=(',',':')).encode()).hexdigest()
            cell.receipt['argv_sha']=cell.kind['argv_sha']
            result=json.loads(cell.files['result.json']);result['selection']['selected_epoch']=1
            cell.files['result.json']=json.dumps(result).encode()
            cell.receipt['result_sha']=cell.receipt['files']['result.json']=fixtures.sha(cell.files['result.json'])
            self.decoded['repro'].update(checkpoint_epoch='001',epochs=2,requested_epochs=2,pairs_per_epoch=128)
            with self.assertRaisesRegex(ValueError,'local smoke cannot install'):
                self.verify(root,cell)
            r=self.verify(root,cell,allow_local_smoke=True)
            self.assertEqual(r['execution_contract'],'local-smoke')
            src=h.cell_dir(root/'stage',cell.name,cell.kind)
            with self.assertRaisesRegex(ValueError,'local smoke cannot install'):
                h.install_stages([(cell.name,src,root/'installed')],root/'rescue')


if __name__=='__main__': unittest.main()
