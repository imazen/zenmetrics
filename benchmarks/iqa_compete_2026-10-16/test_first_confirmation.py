import copy
import unittest

from confirm_first_evaluation import analyze


class FirstConfirmation(unittest.TestCase):
    def setUp(self):
        self.case = dict(id='synthetic', absolute_bound=0.01)
        self.block = dict(case='synthetic', errors={}, arms={
            'native': dict(ns=110, score=0.5),
            'canonical': dict(ns=100, score=0.5)})

    def test_fixed_blocks_preserve_a_slow_ratio(self):
        value = analyze([copy.deepcopy(self.block) for _ in range(20)], self.case)
        self.assertEqual(value['ratio'], 1.1)
        self.assertGreater(value['lower_95'], 1.08)

    def test_a_single_numerical_failure_refuses_the_interval(self):
        blocks = [copy.deepcopy(self.block) for _ in range(20)]
        blocks[13]['arms']['native']['score'] = 0.6
        with self.assertRaises(ValueError):
            analyze(blocks, self.case)

    def test_nonfinite_or_nonpositive_observations_refuse(self):
        for key, value in [('ns', 0), ('ns', float('nan')), ('score', float('inf'))]:
            block = copy.deepcopy(self.block)
            block['arms']['native'][key] = value
            with self.assertRaises(ValueError):
                analyze([block] * 20, self.case)

    def test_missing_arm_and_failed_execution_refuse(self):
        for block in [dict(self.block, arms={}), dict(self.block, errors={'native': 'exit 1'})]:
            with self.assertRaises(ValueError):
                analyze([block] * 20, self.case)

    def test_insufficient_blocks_refuse(self):
        with self.assertRaises(ValueError):
            analyze([self.block] * 19, self.case)


if __name__ == '__main__':
    unittest.main()
