import unittest

from publish_nn_first_evaluation import observation


class FirstPublicationTests(unittest.TestCase):
    def block(self, bound, native=1.0, canonical=1.0):
        delta = abs(native-canonical)
        return ({'id': 'case', 'absolute_bound': bound},
                dict(case='case', errors={}, arms={
                    'native': dict(ns=20, score=native),
                    'canonical': dict(ns=10, score=canonical)},
                    delta=delta, passed=None if bound is None else delta <= bound,
                    ratio=2.0))

    def test_no_assigned_bound_remains_unqualified(self):
        case, block = self.block(None, 1.2)
        row = observation(block, case)
        self.assertIsNone(row['passed'])
        self.assertEqual(row['ratio'], 2)

    def test_numerical_failure_withholds_ratio(self):
        case, block = self.block(.01, 1.2)
        self.assertIsNone(observation(block, case)['ratio'])
        block['passed'] = True
        with self.assertRaisesRegex(ValueError, 'decision differs'):
            observation(block, case)

    def test_nonfinite_or_invalid_time_is_not_published(self):
        for field, value in [('score', float('nan')), ('ns', 0)]:
            case, block = self.block(.01)
            block['arms']['native'][field] = value
            with self.assertRaisesRegex(ValueError, 'invalid first'):
                observation(block, case)


if __name__ == '__main__':
    unittest.main()
