import unittest

from canonical_intervals import interval


class BlockIntervals(unittest.TestCase):
    def test_constant_complete_blocks_have_no_sampling_width(self):
        result = interval([2.]*20, draws=1000)
        self.assertEqual((result['ratio'], result['lower_95'], result['upper_95']), (2., 2., 2.))
        self.assertTrue(result['precision_within_three_percent'])

    def test_short_and_nonfinite_cohorts_are_refused(self):
        for ratios in ([1.]*19, [float('nan')]*20, [0.]*20):
            with self.assertRaises(ValueError):
                interval(ratios)
