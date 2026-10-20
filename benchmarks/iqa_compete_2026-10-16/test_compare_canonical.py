import math
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compare_canonical import compare, parse, summarize


class CanonicalComparison(unittest.TestCase):
    def test_engine_cohorts_with_same_basename_keep_both_artifacts(self):
        from compare_canonical import sha
        scratch=Path.home()/'tmp';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            base=Path(temporary);roots=[]
            for source in ('rten','ort'):
                root=base/source/'same-run';root.mkdir(parents=True);roots.append(root)
                case=dict(id='example-t1',metric='example',width=512,height=512,threads=1,
                          absolute_bound=.01,matched_scope=True,scope='resident input')
                plan=dict(binary=sys.executable,bindings={},rust_source=source,cases=[case])
                (root/'plan.json').write_text(json.dumps(plan))
                directory=root/case['id'];directory.mkdir()
                for number in range(3):
                    block=directory/f'block-{number:02d}';block.mkdir()
                    record=dict(passed=True,delta=0.,rust_ns=2.,canonical_ns=4.,arms={
                        'rust':dict(score=.5,times_ns=[2.,2.,2.]),
                        'canonical':dict(score=.5,times_ns=[4.,4.,4.])})
                    (block/'result.json').write_text(json.dumps(record))
                row=dict(case='example-t1',canonical='original',passed=True,max_abs_delta=0.,ratio=.5,
                         rust_source=source,plan_sha256=sha(root/'plan.json'),**{k:v for k,v in case.items() if k!='id'})
                (directory/'summary.json').write_text(json.dumps(row))
            output=base/'published'
            subprocess.run([sys.executable,str(Path(__file__).with_name('publish_canonical_comparison.py')),
                            str(output),*map(str,roots)],check=True,capture_output=True)
            index=json.loads((output/'cohorts.json').read_text())
            self.assertEqual(len(index),2)
            files=[row['artifacts'][0]['file'] for row in index.values()]
            self.assertEqual(len(set(files)),2)
            for row in index.values():
                artifact=row['artifacts'][0]
                self.assertEqual(sha(output/artifact['file']),artifact['sha256'])

    def test_original_indexed_and_unindexed_samples(self):
        self.assertEqual(parse('SAMPLE 1 1234 .25\nSAMPLE 2 2345 .25', 'samples')[:2],
                         (.25, [1234., 2345.]))
        self.assertEqual(parse('SAMPLE 1234 .25', 'samples')[:2], (.25, [1234.]))
        with self.assertRaises(ValueError):
            parse('SAMPLE 1 1234 .25\nSAMPLE 2 2345 .26', 'samples')

    def test_no_ratio_on_failed_or_mismatched_work(self):
        blocks = [dict(passed=True, rust_ns=2., canonical_ns=4.)]*3
        self.assertEqual(summarize(blocks, True), .5)
        self.assertIsNone(summarize(blocks, False))
        self.assertIsNone(summarize([dict(passed=False)], True))
        self.assertIsNone(summarize([dict(passed=True, errors={'rust': 'failed'})], True))

    def test_unassigned_bound_is_not_a_pass(self):
        self.assertEqual(compare(.5, .6, None)[1], None)
        self.assertFalse(compare(.5, .6, .01)[1])
        with self.assertRaises(ValueError):
            compare(math.nan, .5, .01)

    def test_projection_retains_missing_cases_and_unassigned_bounds(self):
        from publish_canonical_comparison import project
        row=dict(metric='example',excluded_contract='',expected_cases=6,plan_sha256='plan',
                 passed=True,max_abs_delta=0.,ratio=.5,width=512,height=512,threads=1,
                 rust_source='source',canonical='primary')
        result=project([row])['example']
        self.assertEqual(result['check'],'1/6 pass')
        self.assertEqual(result['observed_cases'],1)
        result=project([dict(row,passed=None,max_abs_delta=.001)])['example']
        self.assertEqual(result['check'],'Δ 0.001')
        self.assertEqual(result['passed_cases'],0)

    def test_golden_pass_does_not_hide_a_comparative_finding(self):
        from publish_canonical_comparison import attach_goldens
        value={'example':dict(check='1 findings',failed_cases=1)}
        goldens={'metrics':{'example':dict(checks_passed=170,checks_expected=170)}}
        attach_goldens(value,goldens,'receipt')
        self.assertEqual(value['example']['check'],'1 findings')
        self.assertEqual(value['example']['golden_checks']['checks_passed'],170)

    def test_runtime_error_remains_visible_beside_completed_ratio(self):
        from publish_canonical_comparison import project
        row=dict(metric='example',excluded_contract='',expected_cases=1,plan_sha256='good',
                 passed=True,max_abs_delta=0.,ratio=.5,width=512,height=512,threads=1,
                 rust_source='source',canonical='primary')
        error=dict(row,plan_sha256='failed',passed=False,ratio=None,process_errors=1,
                   max_abs_delta=None)
        result=project([row,error])['example']
        self.assertEqual(result['check'],'1 process errors')
        self.assertEqual(result['ratio'],.5)
        self.assertEqual(result['process_error_cases'],1)

    def test_revised_source_selection_keeps_historical_cases(self):
        from publish_canonical_comparison import project
        row=dict(metric='example',excluded_contract='',expected_cases=1,plan_sha256='old-plan',
                 passed=True,max_abs_delta=0.,ratio=.5,width=512,height=512,threads=1,
                 rust_source='old-source',canonical='primary')
        updated=dict(row,plan_sha256='new-plan',rust_source='new-source',ratio=.4,engine='ort-cpu')
        result=project([row,updated],{'example':'new-source'})['example']
        self.assertEqual(result['ratio'],.4)
        self.assertEqual(result['ratio_source'],'new-source')
        self.assertEqual(result['ratio_engine'],'ort-cpu')
        self.assertEqual(result['observed_cases'],2)
        self.assertEqual(result['rust_sources'],['new-source','old-source'])

    def test_publication_rechecks_every_canonical_score(self):
        from publish_canonical_comparison import validate_numerics
        case=dict(absolute_bound=.01)
        block=dict(delta=0.,passed=True,rust_ns=2.,canonical_ns=4.,arms={
            'rust':dict(score=.5,times_ns=[2.,2.,2.]),
            'canonical':dict(score=.5,times_ns=[4.,4.,4.],extra=dict(sample_scores=[.5,.6]))})
        row=dict(passed=True,max_abs_delta=0.)
        with self.assertRaisesRegex(ValueError,'acquired arms'):
            validate_numerics([block],case,row)
        block['delta']=abs(.5-.6);block['passed']=False
        validate_numerics([block],case,dict(passed=False,max_abs_delta=abs(.5-.6)))


if __name__ == '__main__':
    unittest.main()
