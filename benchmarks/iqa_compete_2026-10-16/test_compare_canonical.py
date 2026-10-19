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


if __name__ == '__main__':
    unittest.main()
