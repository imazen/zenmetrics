import copy
import json
from pathlib import Path
import tempfile
import unittest
from compare_canonical import sha
import resume_canonical_campaign as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.home()/'tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = self.root/'original.json'
        self.output = self.root/'retained'
        self.case = dict(id='case',metric='metric',width=64,height=64,threads=1,
                         absolute_bound=.01,matched_scope=True,scope='warm',canonical=dict(label='original'))
        self.spec = dict(cases=[self.case],rust_source='original-source',phase='first',bindings={})
        self.plan.write_text(json.dumps(self.spec))
        self.calls = []

    def block(self,number,mode='warm',failed=False):
        native = 2 if failed else 1
        if mode == 'warm':
            return dict(order=['canonical','rust'] if number%2==0 else ['rust','canonical'],arms=dict(canonical=dict(score=1,extra=dict(sample_scores=[1]*5)),rust=dict(score=native)),errors={},delta=abs(native-1),passed=not failed,rust_ns=2,canonical_ns=1)
        return dict(case='case',phase='first',arms=dict(canonical=dict(score=1,ns=1),native=dict(score=native,ns=2)),errors={},delta=abs(native-1),passed=not failed,ratio=2)

    def execute(self,plan,case,number,output,mode,samples,helper):
        self.calls.append((number,output))
        output.mkdir()
        (output/'result.json').write_text(json.dumps(self.block(number,mode)))

    def retain(self,number,failed=False):
        path=self.output/'case'/f'block-{number:02d}'
        path.mkdir(parents=True)
        (path/'result.json').write_text(json.dumps(self.block(number,failed=failed)))
        return path/'result.json'

    def test_reuses_complete_failure_preserves_interrupted_attempt_and_idempotent_summary(self):
        retained=self.retain(0,failed=True)
        old=retained.read_bytes()
        interrupted=self.output/'case'/'block-01'
        interrupted.mkdir()
        (interrupted/'stdout.log').write_text('retained partial output')
        result=recovery.resume(self.plan,self.output,executor=self.execute,emitter=lambda _:None)
        self.assertEqual(result['failed'],1)
        self.assertEqual([n for n,_ in self.calls],[1,2])
        self.assertEqual(self.calls[0][1].name,'block-01-recovery-1')
        self.assertEqual(retained.read_bytes(),old)
        self.assertEqual((interrupted/'stdout.log').read_text(),'retained partial output')
        original=(self.output/'summary.tsv').read_bytes()
        self.calls.clear()
        recovery.resume(self.plan,self.output,executor=self.execute,emitter=lambda _:None)
        self.assertEqual(self.calls,[])
        self.assertEqual((self.output/'summary.tsv').read_bytes(),original)

    def test_changed_plan_and_duplicate_completed_attempt_rejected(self):
        self.retain(0)
        duplicate=self.output/'case'/'block-00-recovery-1'
        duplicate.mkdir()
        (duplicate/'result.json').write_text(json.dumps(self.block(0)))
        with self.assertRaisesRegex(ValueError,'multiple completed'):
            recovery.resume(self.plan,self.output,executor=self.execute)
        self.assertEqual(self.calls,[])
        self.plan.write_text(json.dumps(dict(self.spec,rust_source='changed')))
        with self.assertRaisesRegex(ValueError,'original plan differs'):
            recovery.resume(self.plan,self.output,executor=self.execute)

    def test_changed_retained_score_classification_rejected_without_acquisition(self):
        path=self.retain(0)
        block=json.loads(path.read_text())
        block['arms']['canonical']['extra']['sample_scores'][-1]=3
        path.write_text(json.dumps(block))
        with self.assertRaisesRegex(ValueError,'numerical classification'):
            recovery.resume(self.plan,self.output,executor=self.execute)
        self.assertEqual(self.calls,[])

    def test_first_summary_has_original_schema_and_second_pass_does_not_run(self):
        recovery.resume(self.plan,self.output,'first',executor=self.execute,emitter=lambda _:None)
        summary=json.loads((self.output/'case'/'summary.json').read_text())
        self.assertEqual(summary['plan_sha256'],sha(self.plan))
        self.assertEqual(summary['ratio'],2)
        self.assertTrue(summary['passed'])
        self.assertEqual(len(summary['blocks']),3)
        self.calls.clear()
        recovery.resume(self.plan,self.output,'first',executor=self.execute,emitter=lambda _:None)
        self.assertEqual(self.calls,[])

    def test_changed_summary_never_overwritten(self):
        recovery.resume(self.plan,self.output,executor=self.execute,emitter=lambda _:None)
        path=self.output/'case'/'summary.json'
        value=json.loads(path.read_text());value['ratio']=999
        path.write_text(json.dumps(value));old=path.read_bytes()
        with self.assertRaisesRegex(ValueError,'completed case summary differs'):
            recovery.resume(self.plan,self.output,executor=self.execute)
        self.assertEqual(path.read_bytes(),old)

    def test_non_original_sample_budgets_rejected(self):
        with self.assertRaises(ValueError):
            recovery.resume(self.plan,self.output,samples=3,executor=self.execute)
        self.assertFalse(self.output.exists())

    def test_explicit_helper_is_hashed_and_cannot_override_original_binding(self):
        helper=self.root/'compare_canonical_budgeted.py'
        helper.write_text('# frozen helper')
        digest=sha(helper)
        self.assertEqual(recovery._helper(self.spec,'warm',helper,digest),helper)
        with self.assertRaisesRegex(ValueError,'required together'):
            recovery._helper(self.spec,'warm',helper)
        original=copy.deepcopy(self.spec);original['bindings'][str(helper)]='0'*64
        with self.assertRaisesRegex(ValueError,'contradicts'):
            recovery._helper(original,'warm',helper,digest)
        helper.write_text('# changed helper')
        with self.assertRaisesRegex(ValueError,'identity differs'):
            recovery._helper(self.spec,'warm',helper,digest)

    def test_fixed_confirmation_directories_are_not_treated_as_three_block_campaigns(self):
        path=self.output/'case'/'block-19'
        path.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'expanded campaign'):
            recovery.resume(self.plan,self.output,executor=self.execute)
        self.assertEqual(self.calls,[])

    def test_original_metric_subset_does_not_acquire_excluded_cases(self):
        other=dict(self.case,id='excluded',metric='excluded')
        self.spec['cases'].append(other);self.plan.write_text(json.dumps(self.spec))
        result=recovery.resume(self.plan,self.output,executor=self.execute,emitter=lambda _:None,metrics='metric')
        self.assertEqual(result['expected'],1)
        self.assertEqual(len(self.calls),3)
        self.assertFalse((self.output/'excluded').exists())
        with self.assertRaisesRegex(ValueError,'absent from frozen'):
            recovery.resume(self.plan,self.output,executor=self.execute,metrics='unknown')

    def test_queue_preserves_original_subset_and_rejects_confirmation_budget(self):
        from resume_canonical_queue import options
        selected=options(dict(options=['--metrics','metric','--blocks','3','--samples','5']))
        self.assertEqual(selected.metrics,'metric')
        with self.assertRaisesRegex(ValueError,'nonstandard budget'):
            options(dict(options=['--blocks','20']))
