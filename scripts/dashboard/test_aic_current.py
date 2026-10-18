"""Regressions for historical missing labels and omitted qualified measurements."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import aic_reproduction
import aic_runtime
from catalog import ROOT


class CurrentEvidence(unittest.TestCase):
    def test_current_rows_do_not_read_the_superseded_status_table(self):
        original=aic_reproduction.md_tables
        def current_only(path):
            self.assertNotIn('COVERAGE_AUDIT',str(path))
            return original(path)
        with patch.object(aic_reproduction,'md_tables',current_only):
            rows=aic_reproduction.rows([], 'a'*40)
        original_butter=next(r for r in rows if r['column']=='Butteraugli-original')
        self.assertEqual(original_butter['implementation'],'butteraugli-original')
        self.assertTrue(original_butter['status'].startswith('reproduced'))
        self.assertLess(float(original_butter['maximum']),0.00005)
        for row in rows:
            if row['column'].startswith(('proposal-DVIFM','WD_s','A-FINE')):
                self.assertNotIn('missing',row['implementation'].lower())
                self.assertNotIn('MISSING',row['status'])
        pyiqa=next(r for r in rows if r['column']=='TOPIQ')
        self.assertIn('reference versus published',pyiqa['comparison_scope'])
        self.assertNotIn('missing',pyiqa['note'].lower())

    def test_qualified_snapshots_preserve_host_lineage_and_bind_input_files(self):
        result=aic_runtime.snapshot()
        self.assertEqual({r['host'] for r in result['hosts']},{'i265','mac'})
        for summary in result['hosts']:
            source=json.loads((ROOT/aic_runtime.SNAPSHOT/summary['host']/'snapshot.json').read_text())
            self.assertEqual(summary['qualified_attempt_rows'],source['qualified_attempt_rows'])
        self.assertTrue(any(r['width']==3840 and r['height']==2160 for r in result['observations']))
        for row in result['observations']:
            self.assertTrue(row['source_report'])
            self.assertTrue(row['impl_binary_sha'])
            self.assertTrue(row['reference_sha256'])
            self.assertNotIn('score',row)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/aic_runtime.SNAPSHOT/'i265';base.mkdir(parents=True)
            manifest=json.loads((ROOT/aic_runtime.SNAPSHOT/'i265/snapshot.json').read_text())
            (base/'snapshot.json').write_text(json.dumps(manifest))
            (base/'sources.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                aic_runtime.snapshot(Path(directory))

    def test_new_supplements_are_not_pooled_across_hosts_or_master_and_branch(self):
        rows=aic_runtime.supplements()
        self.assertEqual({r['host'] for r in rows},{'r5900xt'})
        self.assertTrue(any(r['metric']=='ahiq' and r['shape']=='3840x2160' for r in rows))
        luma=[r for r in rows if r['metric'].startswith('fsim-y-k')]
        self.assertEqual(sum(r['n'] for r in luma),36)
        self.assertTrue(all('Unmerged' in r['qualification'] for r in luma))
        self.assertEqual(sum(r['n'] for r in rows if r['metric'].startswith('dvifm-port-')),48)


if __name__=='__main__':unittest.main()
