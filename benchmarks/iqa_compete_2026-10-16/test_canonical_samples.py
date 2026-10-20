import unittest

from compare_canonical_samples import compare_samples, parse


class AllWarmScores(unittest.TestCase):
    def test_every_value_is_retained_and_checked(self):
        score, times, extra = parse('SAMPLE 1000 .25\nSAMPLE 2000 .26', 'samples')
        self.assertEqual((score,times,extra['sample_scores']),(.25,[1000.,2000.],[.25,.26]))
        delta, passed = compare_samples(.25, extra['sample_scores'], .001)
        self.assertAlmostEqual(delta,.01)
        self.assertFalse(passed)

    def test_small_reduction_variation_uses_existing_bound(self):
        self.assertTrue(compare_samples(.25,[.25,.25000001],1e-6)[1])
        self.assertIsNone(compare_samples(.25,[.25,.25000001],None)[1])
        with self.assertRaises(ValueError):
            parse('SAMPLE 1000 nan', 'samples')
