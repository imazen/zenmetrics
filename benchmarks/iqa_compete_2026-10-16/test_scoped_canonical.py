import json
import unittest

from compare_canonical import sha, summarize
from prepare_scoped_canonical import scoped_prepare
import test_ordered_canonical as fixtures


class ScopedContracts(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OrderedContracts()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def manifest(self, caveats):
        path = self.fixture.manifest
        value = json.loads(path.read_text())
        value['scope_dispositions'] = caveats
        path.write_text(json.dumps(value))

    def test_caveat_withholds_ratio_without_changing_numerical_gate(self):
        self.manifest({'a': 'Reference parameter preparation occurs before its clock'})
        f = self.fixture
        result = scoped_prepare(f.manifest, f.output, [1, 8])
        self.assertEqual((result['metrics'], result['cases'], result['scope_caveated_cases']), (2, 4, 2))
        plan = json.loads(f.output.read_text())
        blocks = [dict(passed=True, rust_ns=2, canonical_ns=1)]
        for case in plan['cases']:
            self.assertEqual((case['absolute_bound'], case['relative_bound']), (0.004, 0.0001))
            self.assertEqual(case['matched_scope'], case['metric'] != 'a')
            self.assertEqual(summarize(blocks, case['matched_scope']),
                             None if case['metric'] == 'a' else 2)
        self.assertEqual(plan['bindings'][str(f.manifest)], sha(f.manifest))

    def test_unknown_or_empty_caveat_is_refused_before_output(self):
        f = self.fixture
        for caveats in [{'absent': 'scope differs'}, {'a': ' '}, {'a': None}, []]:
            self.manifest(caveats)
            with self.subTest(caveats=caveats), self.assertRaises(ValueError):
                scoped_prepare(f.manifest, f.output, [1])
            self.assertFalse(f.output.exists())

    def test_existing_output_is_never_reclassified(self):
        f = self.fixture
        self.manifest({'a': 'new qualification'})
        f.output.write_text('retained acquisition')
        with self.assertRaises(FileExistsError):
            scoped_prepare(f.manifest, f.output, [1])
        self.assertEqual(f.output.read_text(), 'retained acquisition')


if __name__ == '__main__':
    unittest.main()
