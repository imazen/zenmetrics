import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compare_canonical import sha, summarize
import prepare_canonical_grid_scope as scoped


class GridScopeContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pixels = self.root/'pixels'
        self.pixels.write_bytes(b'rgb')
        self.worker = self.root/'worker'
        self.worker.write_text('fixture worker is never executed')
        cases = [dict(id=name, metric=name, width=1, height=1, threads=1,
                      reference=str(self.pixels), distorted=str(self.pixels),
                      scope='resident RGB', matched_scope=True,
                      absolute_bound=0.004, relative_bound=0.0001,
                      canonical=dict(label='fixture', command=['original', '{threads}']))
                 for name in ('a', 'b')]
        self.original = dict(binary=str(self.worker), rust_source='frozen', cases=cases,
                             bindings={str(path): sha(path) for path in (self.pixels, self.worker)})
        self.reviewed = copy.deepcopy(self.original)
        self.reviewed.update(scope_dispositions={'a': 'different parameter preparation'},
                             scope_qualification='source review only')
        self.reviewed['cases'][0].update(matched_scope=False,
            scope_disposition='different parameter preparation', scope='reviewed caveat')
        self.review_asset = self.root/'review-only-source'
        self.review_asset.write_text('retained review input')
        self.reviewed['bindings'][str(self.review_asset)] = sha(self.review_asset)
        self.source = self.root/'source.json'
        self.review = self.root/'reviewed.json'
        self.pairs = self.root/'pairs.json'
        self.pairs.write_text(json.dumps(dict(pairs=[dict(pair='pairA', content='fixture',
            width=1, height=1, reference='pixels', distorted='pixels',
            reference_sha256=sha(self.pixels), distorted_sha256=sha(self.pixels))])))
        self.output = self.root/'output.json'
        self.save()

    def save(self):
        self.source.write_text(json.dumps(self.original))
        self.review.write_text(json.dumps(self.reviewed))

    def prepare(self, metrics=('a', 'b')):
        return scoped.prepare_scoped_grid(self.source, self.pairs, self.review,
                                          self.output, list(metrics), [1, 8])

    def test_numerical_gates_and_review_only_bindings_survive(self):
        result = self.prepare()
        self.assertEqual((result['cases'], result['scope_caveated_cases']), (4, 2))
        value = json.loads(self.output.read_text())
        self.assertEqual(value['bindings'][str(self.review_asset)], sha(self.review_asset))
        for row in value['cases']:
            self.assertEqual((row['absolute_bound'], row['relative_bound']), (0.004, 0.0001))
            self.assertEqual(summarize([dict(passed=True, rust_ns=2, canonical_ns=1)],
                                      row['matched_scope']), None if row['metric'] == 'a' else 2)
        candidate = Path(result['preparation_directory'])/'qualified-grid.plan.json'
        self.assertEqual(candidate.stat().st_ino, self.output.stat().st_ino)

    def test_changed_numerical_contract_is_refused(self):
        self.reviewed['cases'][0]['absolute_bound'] = 0.4
        self.save()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_inconsistent_scope_flags_are_refused(self):
        before = copy.deepcopy(self.reviewed)
        for mode in ('missing_reason', 'missing_flag', 'unknown_route', 'empty_reason'):
            self.reviewed = copy.deepcopy(before)
            if mode == 'missing_reason': self.reviewed['scope_dispositions'] = {}
            elif mode == 'missing_flag': self.reviewed['cases'][0]['matched_scope'] = True
            elif mode == 'unknown_route': self.reviewed['scope_dispositions']['absent'] = 'caveat'
            else: self.reviewed['scope_dispositions']['a'] = ' '
            self.save()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.output.exists())

    def test_review_asset_drift_is_refused(self):
        self.review_asset.write_text('changed after review')
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_interrupted_qualification_never_publishes_unqualified_plan(self):
        with patch.object(scoped, 'qualify', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError):
                self.prepare()
        self.assertFalse(self.output.exists())
        retained = list(self.root.glob('output.json.prepare-*/source-grid.plan.json'))
        self.assertEqual(len(retained), 1)
        self.assertTrue(all(row['matched_scope'] for row in json.loads(retained[0].read_text())['cases']))

    def test_existing_output_is_preserved(self):
        self.output.write_text('retained observation')
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(self.output.read_text(), 'retained observation')

    def test_original_unmatched_contract_cannot_be_overridden(self):
        self.original['cases'][0]['matched_scope'] = False
        self.save()
        with self.assertRaises(ValueError):
            self.prepare(('a',))
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
