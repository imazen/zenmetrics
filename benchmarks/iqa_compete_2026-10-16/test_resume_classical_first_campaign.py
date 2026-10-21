import json
from pathlib import Path
import tempfile
import unittest
import resume_classical_first_campaign as recovery

HELPER='''import json
PHASE='scorer_construct_first_eval'
def summarize(blocks,reasons,expected):
    if reasons or any(block.get('errors') or block['passed'] is False for block in blocks):return None
    return 2
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream)
'''


class ColdRecoveryTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(dir=Path.home()/'tmp');self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.output=self.root/'output';self.helper=self.root/'original.py';self.helper.write_text(HELPER)
        self.case=dict(id='case',metric='metric',threads=1,absolute_bound=.01,first_call_scope=dict(matched=True),rust_source='original-source')
        self.plan=self.root/'plan.json';self.spec=dict(phase='scorer_construct_first_eval',bindings={str(self.helper):recovery.sha(self.helper)},cases=[self.case]);self.plan.write_text(json.dumps(self.spec));self.calls=[]

    def block(self,number,failed=False):
        native=2 if failed else 1
        return dict(order=['canonical','native'] if number%2==0 else ['native','canonical'],phase='scorer_construct_first_eval',arms=dict(native=dict(ns=2,score=native),canonical=dict(ns=1,score=1)),errors={},passed=not failed,delta=abs(native-1),scope_reasons=[])

    def execute(self,plan,case,number,output,helper,digest):
        self.calls.append((number,output));output.mkdir();(output/'result.json').write_text(json.dumps(self.block(number)))

    def run_recovery(self):
        return recovery.resume(self.plan,self.output,self.helper,self.spec['bindings'][str(self.helper)],executor=self.execute)

    def test_preserves_failure_and_interruption_then_reuses_all_completed_blocks(self):
        directory=self.output/'case';retained=directory/'block-00';retained.mkdir(parents=True)
        path=retained/'result.json';path.write_text(json.dumps(self.block(0,True)));before=path.read_bytes()
        partial=directory/'block-01';partial.mkdir();(partial/'native.log').write_text('partial')
        result=self.run_recovery();self.assertEqual(result['failed'],1);self.assertEqual([n for n,_ in self.calls],[1,2]);self.assertEqual(self.calls[0][1].name,'block-01-recovery-1')
        self.assertEqual(path.read_bytes(),before);self.assertEqual((partial/'native.log').read_text(),'partial')
        summary=json.loads((directory/'summary.json').read_text());self.assertEqual(summary['blocks'][0]['errors']['wrapper'],'exit 1');self.assertIsNone(summary['ratio'])
        self.calls.clear();self.run_recovery();self.assertEqual(self.calls,[])

    def test_changed_helper_or_frozen_plan_stops_without_acquisition(self):
        self.output.mkdir();(self.output/'plan.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'frozen plan differs'):self.run_recovery()
        self.assertEqual(self.calls,[])
        self.helper.write_text(HELPER+'# changed')
        with self.assertRaisesRegex(ValueError,'binding differs'):self.run_recovery()
        self.assertEqual(self.calls,[])

    def test_confirmation_and_completed_summary_changes_are_rejected(self):
        self.run_recovery();path=self.output/'case'/'summary.json';old=json.loads(path.read_text());old['ratio']=99;path.write_text(json.dumps(old));before=path.read_bytes();self.calls.clear()
        with self.assertRaisesRegex(ValueError,'completed summary differs'):self.run_recovery()
        self.assertEqual(path.read_bytes(),before);self.assertEqual(self.calls,[])
        (self.output/'case'/'block-19').mkdir()
        with self.assertRaisesRegex(ValueError,'confirmation stage'):self.run_recovery()
        self.assertEqual(self.calls,[])
