#!/usr/bin/env python3
"""Guard identity joins, refusal handling and inclusive existing bounds."""
import math
import unittest

from canonical_compare import compare, unique


class ComparisonTests(unittest.TestCase):
    def test_join_by_identity_instead_of_position(self):
        result, worst = compare('metric', 'oracle', {'b': 2., 'a': 1., 'native-only': 9.},
                                {'a': 1., 'reference-only': 8., 'b': 2.}, 0.)
        self.assertEqual((result['n'], result['exact_eq'], result['match']), (2, 2, 'MATCH'))
        self.assertEqual({c['distorted'] for c in worst}, {'a', 'b'})

    def test_duplicate_identity_refused(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            unique([{'distorted': 'a'}, {'distorted': 'a'}])

    def test_bound_is_inclusive_and_cannot_hide_a_tail(self):
        result, _ = compare('metric', 'oracle', {'a': .25, 'b': .5}, {'a': 0., 'b': 0.}, .5)
        self.assertEqual(result['match'], 'MATCH')
        result, worst = compare('metric', 'oracle', {'a': 0., 'b': 1.}, {'a': 0., 'b': 0.}, .5)
        self.assertEqual(result['match'], 'MISS')
        self.assertEqual(worst[0]['distorted'], 'b')
        self.assertTrue(math.isinf(result['max_rel']))

    def test_nonfinite_and_missing_values_are_not_numeric_matches(self):
        result, _ = compare('metric', 'oracle', {'a': math.nan, 'b': '', 'c': 1.}, {'a': 0., 'b': 0.}, 1.)
        self.assertEqual(result['n'], 0)
        self.assertEqual(result['match'], 'MISS')

    def test_no_bound_stays_unqualified(self):
        result, _ = compare('metric', 'oracle', {'a': 10.}, {'a': 1.}, None)
        self.assertEqual(result['match'], 'no-gate')
        self.assertEqual(result['gate'], 'none')


if __name__ == '__main__':
    unittest.main()
