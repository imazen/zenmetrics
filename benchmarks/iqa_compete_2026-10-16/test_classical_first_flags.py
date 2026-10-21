import copy
import json
from pathlib import Path
import tempfile
import unittest

import capture_classical_first_flags as flags


class AggregateSummaryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='synthetic-first-flags-', dir=Path.home()/'tmp'))
        self.log = self.root/'REGRESSIONS.md'
        self.case = dict(id='synthetic-clock-fixture', metric='fixture-only', width=64, height=64, threads=1)
        self.plan = dict(phase=flags.PHASE, cases=[self.case], synthetic_fixture=True)
        self.row = dict(phase=flags.PHASE, case=self.case['id'], ratio=1.2,
            absolute_bound=None, relative_bound=0, scope_reasons=[], blocks=[dict(
                errors={}, passed=None, arms=dict(native=dict(ns=120), canonical=dict(ns=100)))])
        self.write(self.root/'plan.json', self.plan)
        (self.root/self.case['id']).mkdir()
        self.write(self.root/self.case['id']/'summary.json', self.row)
        self.write(self.root/'summary.json', [self.row])

    def write(self, path, value):
        path.write_text(json.dumps(value)+'\n')

    def test_completed_aggregate_counts_case_once_and_retains_null_bound(self):
        result = flags.capture([self.root], self.log)
        self.assertEqual(len(result['completed_summaries']), 1)
        self.assertEqual(len(result['flags']), 1)
        self.assertIsNone(result['flags'][0]['absolute_bound'])
        self.assertIs(result['flags'][0]['regression_qualified'], False)

    def test_repeated_capture_does_not_duplicate_regression_line(self):
        flags.capture([self.root], self.log)
        first = self.log.read_text()
        flags.capture([self.root], self.log)
        self.assertEqual(self.log.read_text(), first)

    def test_aggregate_changed_result_is_rejected(self):
        other = copy.deepcopy(self.row)
        other['ratio'] = 9
        self.write(self.root/'summary.json', [other])
        with self.assertRaisesRegex(ValueError, 'aggregate'):
            flags.capture([self.root], self.log)

    def test_unexpected_aggregate_case_is_rejected(self):
        other = copy.deepcopy(self.row)
        other['case'] = 'other-fixture'
        self.write(self.root/'summary.json', [other])
        with self.assertRaisesRegex(ValueError, 'aggregate'):
            flags.capture([self.root], self.log)

    def test_aggregate_without_plan_is_rejected(self):
        nested = self.root/'unplanned'
        nested.mkdir()
        self.write(nested/'summary.json', [self.row])
        with self.assertRaisesRegex(ValueError, 'aggregate'):
            flags.capture([nested], self.log)


if __name__ == '__main__':
    unittest.main()
