import copy
import unittest

from confirm_scored_first_evaluation import analyze, validate_bound


class BoundedFirstConfirmation(unittest.TestCase):
    def setUp(self):
        self.case = dict(id='synthetic', absolute_bound=0.01)
        self.block = dict(case='synthetic', passed=True, errors={}, arms={
            'native': dict(ns=110, score=0.5),
            'canonical': dict(ns=100, score=0.5)})

    def test_unassigned_bound_cannot_qualify_unequal_finite_scores(self):
        self.case['absolute_bound'] = None
        self.block['arms']['native']['score'] = 0.8
        with self.assertRaises(ValueError):
            analyze([self.block] * 20, self.case)

    def test_invalid_and_missing_bounds_refuse_before_acquisition(self):
        for bound in [None, -0.1, float('nan'), float('inf'), True, '0.1']:
            with self.assertRaises(ValueError):
                validate_bound(dict(absolute_bound=bound))
        with self.assertRaises(ValueError):
            validate_bound({})
        for bound in [None, -0.1, float('nan'), float('inf'), True, '0.1']:
            with self.assertRaises(ValueError):
                validate_bound(dict(absolute_bound=0.01, relative_bound=bound))

    def test_false_or_unknown_block_gate_refuses_even_equal_scores(self):
        for passed in [False, None]:
            block = dict(self.block, passed=passed)
            with self.assertRaises(ValueError):
                analyze([block] * 20, self.case)

    def test_gate_is_recomputed_from_scores(self):
        blocks = [copy.deepcopy(self.block) for _ in range(20)]
        blocks[17]['arms']['native']['score'] = 0.7
        with self.assertRaises(ValueError):
            analyze(blocks, self.case)

    def test_valid_fixed_stage_retains_its_slow_ratio(self):
        value = analyze([self.block] * 20, self.case)
        self.assertEqual(value['ratio'], 1.1)
        self.assertTrue(value['precision_within_three_percent'])

    def test_zero_is_an_explicit_exact_bound(self):
        self.case['absolute_bound'] = 0
        self.assertEqual(analyze([self.block] * 20, self.case)['ratio'], 1.1)


if __name__ == '__main__':
    unittest.main()
