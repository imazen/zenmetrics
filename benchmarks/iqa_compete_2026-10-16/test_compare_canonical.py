import math
import unittest

from compare_canonical import compare, parse, summarize


class CanonicalComparison(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
