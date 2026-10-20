#!/usr/bin/env python3
"""Current40 preparation contract controls; caller owns heavy admission."""
import copy
import unittest

from first_boundary_hdrvdp_current40 import BINARY_SHA, FEATURES, SOURCE, bounds, check_build, original_grid


class Current40Controls(unittest.TestCase):
    def setUp(self):
        self.receipt = dict(source_revision=SOURCE, binary_sha256=BINARY_SHA,
            adapters_applied=False, features=FEATURES, lto='thin',
            native_cpu_flags='RUSTFLAGS,CARGO_ENCODED_RUSTFLAGS,CARGO_BUILD_RUSTFLAGS unset')

    def test_exact_build_contract_and_identity(self):
        check_build(self.receipt, BINARY_SHA)
        for overrides in (dict(source_revision='older'), dict(binary_sha256='0' * 64),
                          dict(adapters_applied=True), dict(features=['compete']),
                          dict(lto='fat'), dict(native_cpu_flags='-C target-cpu=native')):
            with self.assertRaises(ValueError):
                check_build(dict(self.receipt, **overrides), BINARY_SHA)
        with self.assertRaises(ValueError):
            check_build(self.receipt, '0' * 64)

    def test_v2_retains_existing_official_limit(self):
        self.assertEqual(bounds('hdrvdp')['absolute_bound'], 5e-3)
        self.assertEqual(bounds('hdrvdp')['relative_bound'], 0)

    def test_v3_has_no_invented_official_limit(self):
        bound = bounds('hdrvdp3')
        self.assertIsNone(bound['absolute_bound'])
        self.assertIsNone(bound['bound_source'])
        self.assertFalse(bound['native_parity_qualified'])
        self.assertEqual(bound['retained_other_gate']['absolute_bound'], 1e-9)
        self.assertIn('NumPy', bound['retained_other_gate']['scope'])
        with self.assertRaises(ValueError):
            bounds('hdrvdp3-aic2026')

    def grid(self):
        return dict(cases=[dict(metric='hdrvdp-aic2026', width=64, height=64,
            reference=f'reference-{pair}', distorted=f'distorted-{pair}', threads=threads)
            for pair in range(12) for threads in (1, 28)])

    def test_exact_original_grid_and_both_budgets(self):
        self.assertEqual(len(original_grid(self.grid())), 24)

    def test_grid_omission_and_duplicate_refuse(self):
        plan = self.grid()
        del plan['cases'][-1]
        with self.assertRaises(ValueError):
            original_grid(plan)
        plan = self.grid()
        plan['cases'][-1] = copy.deepcopy(plan['cases'][-2])
        with self.assertRaises(ValueError):
            original_grid(plan)

    def test_eight_threads_do_not_replace_original_budget(self):
        plan = self.grid()
        for case in plan['cases']:
            if case['threads'] == 28:
                case['threads'] = 8
        with self.assertRaises(ValueError):
            original_grid(plan)


if __name__ == '__main__':
    unittest.main()
