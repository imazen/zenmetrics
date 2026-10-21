import json
from pathlib import Path
import tempfile
import unittest
from compare_canonical import sha
from publish_nn_first_evaluation import collect as original_collect
from publish_nn_first_evaluation_recovered import collect


class RecoveredFirstPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(dir=Path.home()/'tmp');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);binary=self.root/'fixture-worker';binary.write_text('fixture only')
        self.case=dict(id='case',metric='fixture',width=64,height=64,threads=1,absolute_bound=.01,matched_scope=True,canonical=dict(label='fixture'),native=dict(command=[str(binary)],environment={'ZENMETRICS_NN_ENGINE':'rten'}))
        self.plan=dict(phase='first',bindings={str(binary):sha(binary)},cases=[self.case]);self.path=self.root/'plan.json';self.path.write_text(json.dumps(self.plan));self.receipt=dict(binary_sha256=sha(binary),source_revision='fixture')
        self.directory=self.root/'case';self.directory.mkdir();self.blocks=[]
        for number in range(3):
            result=dict(case='case',phase='first',arms=dict(native=dict(ns=20,score=1.0),canonical=dict(ns=10,score=1.0)),errors={},delta=0,passed=True,ratio=2)
            output=self.directory/f'block-{number:02d}';output.mkdir();(output/'result.json').write_text(json.dumps(result));Path(str(output)+'.log').write_text('run-heavy: done rc=0 fixture resource receipt\n');self.blocks.append(result)
        self.summary=dict(metric='fixture',case='case',width=64,height=64,threads=1,phase='first',plan_sha256=sha(self.path),passed=True,ratio=2,blocks=self.blocks)
        self.write_summary()

    def write_summary(self):
        (self.directory/'summary.json').write_text(json.dumps(self.summary))

    def test_original_rows_are_identical(self):
        self.assertEqual(collect(self.root,self.receipt),original_collect(self.root,self.receipt))

    def test_recovered_result_and_its_wrapper_are_explicit_without_replacing_partial_log(self):
        old=self.directory/'block-00';new=self.directory/'block-00-recovery-1';old.rename(new);newlog=Path(str(new)+'.log');newlog.write_text('run-heavy: done rc=0 recovered fixture receipt\n')
        partial=self.directory/'block-00.log';partial.write_text('retained interrupted output');before=partial.read_bytes()
        rows,expected=collect(self.root,self.receipt);self.assertEqual(expected,1);self.assertEqual(rows[0]['ratio'],2)
        self.assertEqual(rows[0]['recovery_evidence']['completed_blocks'][0]['path'],str(new/'result.json'))
        self.assertEqual(rows[0]['recovery_evidence']['interrupted_logs'][0]['sha256'],sha(partial));self.assertEqual(partial.read_bytes(),before)
        self.assertIn('run-heavy: done rc=0 recovered fixture receipt',rows[0]['wrapper_receipts'])

    def test_duplicate_completed_attempt_is_rejected(self):
        duplicate=self.directory/'block-00-recovery-1';duplicate.mkdir();(duplicate/'result.json').write_text(json.dumps(self.blocks[0]))
        with self.assertRaisesRegex(ValueError,'multiple completed'):collect(self.root,self.receipt)

    def test_numerical_failure_keeps_original_gate_and_withholds_ratio(self):
        block=self.blocks[0];block['arms']['native']['score']=1.2;block['delta']=abs(1.2-1);block['passed']=False;self.summary['passed']=False
        (self.directory/'block-00/result.json').write_text(json.dumps(block));self.write_summary()
        rows,_=collect(self.root,self.receipt);self.assertFalse(rows[0]['passed']);self.assertIsNone(rows[0]['ratio'])

    def test_projection_cli_records_both_publisher_dependencies(self):
        import contextlib
        import io
        from unittest.mock import patch
        import sys
        import publish_nn_first_evaluation_recovered as publisher
        receipt=self.root/'receipt.json';receipt.write_text(json.dumps(self.receipt));output=self.root/'projection'
        with patch.object(sys,'argv',['publisher',str(output),str(receipt),str(self.root)]),contextlib.redirect_stdout(io.StringIO()):
            publisher.main()
        index=json.loads((output/'index.json').read_text());self.assertEqual(index['observed'],1)
        self.assertEqual(index['publisher_dependencies']['publish_nn_first_evaluation.py'],sha(publisher.ORIGINAL_PUBLISHER_SOURCE))
        self.assertEqual(index['publisher_dependencies']['resume_canonical_campaign.py'],sha(publisher.RECOVERY_SOURCE))
