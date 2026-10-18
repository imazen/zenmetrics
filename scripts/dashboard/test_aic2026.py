"""Publication boundaries, complete inventories and existing-viewer integration."""
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from catalog import ROOT, rows as catalog_rows
from model_cards import cards
import aic2026
from origins import check as check_origins
from publish_extension import merge


class AIC2026(unittest.TestCase):
    def test_pyiqa_cards_follow_the_actual_package(self):
        by_name={c['metric']:c for c in cards()}
        for name in ('topiq-fr','topiq-fr-pipal','wadiqam-pyiqa','ahiq-pyiqa','pieapp-pyiqa','cw-ssim-pyiqa','dmm-pyiqa'):
            self.assertIn('zenmetrics-pyiqa-ports/models/',by_name[name]['source'])
            self.assertIn('Noncommercial',by_name[name]['rust_license'])
            self.assertEqual(by_name[name]['parity']['tolerance_jnd'],0.001)
        self.assertIn('0.071',json.dumps(by_name['dmm-pyiqa']['parity']))

    def test_external_resources_fail_but_citation_navigation_passes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'index.html'
            for text in ['<script src="https://cdn.example/x.js"></script>', '<img src="//cdn.example/x.png">',
                         '<link rel="stylesheet" href="https://fonts.example/font.css">',
                         '<style>@import "https://cdn.example/font.css";</style>',
                         '<style>x{background:url(https://cdn.example/x)}</style>',
                         '<iframe src="https://external.example/"></iframe>']:
                p.write_text(text)
                with self.assertRaisesRegex(ValueError,'external origin'):check_origins(root)
            p.write_text('<a href="https://doi.org/example">Evidence</a><img src="local.svg"><link rel="icon" href="data:,">')
            check_origins(root)
            (root/'app.js').write_text('fetch("https://external.example/data")')
            with self.assertRaisesRegex(ValueError,'external origin'):check_origins(root)

    def test_screenshot_inventory_and_missing_work_stay_explicit(self):
        rankfile=ROOT/'benchmarks/dashboard/aic2026/screenshot-ranks.tsv'
        ranks=list(csv.DictReader(rankfile.read_text().splitlines(),delimiter='\t'))
        self.assertEqual(len(ranks),31);self.assertEqual(len({r['metric'] for r in ranks}),31)
        draft=json.loads((ROOT/'scripts/complexity/pedzisz_2026_draft_values.json').read_text())
        self.assertEqual(len(draft['values']),14)
        body,_=aic2026.screenshot_coverage({},[],[], 'a'*40)
        for r in ranks:self.assertIn(r['metric'],body)
        for r in draft['values']:self.assertIn(r['draft_name'],body)
        self.assertIn('pending (aicfix)',body)
        self.assertIn('skipped by owner',body)

    def test_full_population_density_and_canonical_fields(self):
        v=json.loads((ROOT/'benchmarks/dashboard/aic2026/scatter-example.json').read_text())
        self.assertEqual(sum(c[2] for c in v['density']['cells']),v['n'])
        self.assertEqual(v['n'],2192)
        self.assertEqual(v['srocc_ci_method']['units'],25)
        self.assertIn('target_cov',v['geo']);self.assertIn('residual_p99',v['raw'])
        self.assertNotIn('per_pair',v);self.assertNotIn('normalized_pred',v)
        self.assertIn('±4',aic2026.density_plot(v))

    def test_private_publication_has_only_aggregate_allowlist(self):
        data=json.loads((ROOT/'benchmarks/dashboard/aic2026/private-aggregates.json').read_text())
        for item in data['models']:
            self.assertEqual(set(item),{'metric','published_column','published_match','native_gate','licence_class','aggregate_record_git_blob'})
        text=json.dumps(data)
        for token in ('s3://','onnx','per_pair','weights.pth'):
            self.assertNotIn(token,text)

    def test_matching_conventions_and_native_failure_remain_distinct(self):
        mapping=json.loads((ROOT/'scripts/complexity/aic2026_columns.json').read_text())
        for col,name in [('TOPIQ','topiq-fr'),('TOPIQ-PIPAL','topiq-fr-pipal'),('WaDIQaM','wadiqam-pyiqa'),('DMM','dmm-pyiqa'),('AHIQ','ahiq-pyiqa'),('PieAPP','pieapp-pyiqa')]:
            self.assertEqual(mapping['map'][col],name)
        rows=aic2026.reproduction_rows([], 'a'*40)
        self.assertTrue(rows[0]['status'].startswith('MISSING'))
        self.assertEqual(next(r for r in rows if r['column']=='DMM')['status'],'fails native gate')
        for col in json.loads((ROOT/aic2026.B/'analysis/aic2026_agreement.json').read_text())['metrics']:
            self.assertTrue(any(col['column'] in r['column'] for r in rows),col['column'])

    def test_viewer_merge_keeps_assets_and_rejects_mixed_revisions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);viewer=root/'viewer';dash=root/'dash';viewer.mkdir();(dash/'aic2026').mkdir(parents=True)
            (viewer/'index.html').write_text('<h1>Existing viewer</h1>')
            (viewer/'worker.js').write_text('existing-worker')
            (viewer/'site-source.json').write_text(json.dumps({'source_commit':'a'*40,'files':{}}))
            (dash/'aic2026/manifest.json').write_text(json.dumps({'source_commit':'b'*40}))
            with self.assertRaisesRegex(ValueError,'commits differ'):merge(viewer,dash,'a'*40)
            (dash/'aic2026/manifest.json').write_text(json.dumps({'source_commit':'a'*40}))
            merge(viewer,dash,'a'*40)
            self.assertIn('Existing viewer',(viewer/'index.html').read_text())
            self.assertEqual((viewer/'worker.js').read_text(),'existing-worker')
            self.assertIn('aic2026/manifest.json',json.loads((viewer/'site-source.json').read_text())['files'])


if __name__=='__main__':unittest.main()
