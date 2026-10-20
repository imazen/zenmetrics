from pathlib import Path
import json
import tempfile
import unittest

from audit_canonical_rounds import inspect


class NativeWarmContract(unittest.TestCase):
    def setUp(self):
        (Path.home()/'tmp').mkdir(parents=True, exist_ok=True)

    def test_short_and_inconsistent_native_arms_are_findings(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as name:
            root=Path(name);(root/'plan.json').write_text('{}')
            for index, counts in enumerate([(10,10,10),(9,9,9),(10,9,10)]):
                path=root/'case'/f'block-{index:02d}';path.mkdir(parents=True)
                n,times,means=counts
                (path/'result.json').write_text(json.dumps(dict(arms=dict(rust=dict(
                    extra=dict(n=n,round_means_ns=[1]*means),times_ns=[1]*times)))))
            result,_=inspect([root])
            self.assertEqual(result['successful_native_arms'],3)
            self.assertEqual(result['findings'],2)

    def test_missing_arm_is_reported_without_becoming_a_success(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as name:
            root=Path(name);(root/'plan.json').write_text('{}')
            path=root/'case/block-00';path.mkdir(parents=True)
            (path/'result.json').write_text('{"arms":{},"errors":{"rust":"refused"}}')
            result,arms=inspect([root])
            self.assertEqual(result['observed_blocks'],1)
            self.assertEqual(result['missing_native_arms'],1)
            self.assertEqual(result['successful_native_arms'],0)
            self.assertEqual(arms,[])
