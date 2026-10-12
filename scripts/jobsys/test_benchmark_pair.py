"""Pair compatibility and qualified-output controls, without timing runs."""
import json
from pathlib import Path
import shutil
import unittest

import build_benchmark_pair as pair
import build_benchmark_study as study
import check_benchmark_pair_output as checker
import test_benchmark_analysis as fixtures


class PairTests(unittest.TestCase):
    def setUp(self):
        fixtures.AnalysisTests.setUp(self)
        for sha,row in self.cells.items():
            if row['status']!='ready': continue
            spec=study.load(self.generated/sha/'bundle-spec.json')
            for name,value in spec['files'].items():
                shutil.copyfile(value['source'],self.packaged/sha/name)

    def test_compatible_pair_retains_correctness_and_launch_budget(self):
        output=self.root/'pair'
        record=pair.build(self.generated,self.packaged,self.sha,self.generated,self.packaged,self.sha,output,4,300)
        plan=study.load(output/'pair.json')
        self.assertEqual(record['invocations'],12)
        self.assertTrue(plan['aa']);self.assertFalse(record['executed'])
        self.assertEqual(plan['metric']['extract'],dict(kind='prefix',prefix='METRIC_NS='))
        self.assertEqual(plan['baseline']['env']['ZENBENCH_RESULT_FILE'],'${output}/driver.json')
        self.assertEqual(plan['check']['kind'],'command')
        self.assertTrue((output/'baseline'/'bundle.json').is_file())

    def test_unlike_workload_and_bad_budget_refused(self):
        other=next(sha for sha,row in self.cells.items() if row['status']=='ready' and row['case']['threads']!=self.row['case']['threads'])
        with self.assertRaisesRegex(ValueError,'differ in'):
            pair.build(self.generated,self.packaged,self.sha,self.generated,self.packaged,other,self.root/'pair',4,300)
        self.assertFalse((self.root/'pair').exists())
        with self.assertRaises(ValueError):
            pair.build(self.generated,self.packaged,self.sha,self.generated,self.packaged,self.sha,self.root/'pair',True,300)

    def test_each_arm_must_match_qualification_even_when_both_are_wrong(self):
        expected=self.root/'expected.json'
        expected.write_text(json.dumps(dict(operations=2,correctness=self.row['expected_correctness'])))
        roots=[self.root/'before',self.root/'after']
        for root in roots:
            root.mkdir();(root/'driver.json').write_text(json.dumps(dict(schema=1,operations=2,correctness=self.row['expected_correctness'],memory=[])))
        checker.check(*roots,expected)
        for root in roots:
            (root/'driver.json').write_text(json.dumps(dict(schema=1,operations=2,correctness={'score_bits':[999,999]},memory=[])))
        with self.assertRaisesRegex(ValueError,'qualified'): checker.check(*roots,expected)
