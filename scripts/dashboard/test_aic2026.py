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
import repro
from origins import check as check_origins
from publish_extension import merge, refresh_landing


class AIC2026(unittest.TestCase):
    def test_explicit_aic_routes_keep_crop_qualification_and_features(self):
        by_name={c['metric']:c for c in cards()}
        self.assertEqual(by_name['hdrvdp3-aic2026']['inputs']['code_bits_before_srgb_eotf'],10)
        self.assertEqual(by_name['hdrvdp-aic2026']['inputs']['code_bits_before_srgb_eotf'],0)
        self.assertEqual(by_name['cambi-aic2026']['kind'],'NR')
        self.assertIn('cpu-hdrvdp',repro.FEATURES['hdrvdp3-aic2026'])
        self.assertIn('cpu-vmaf',repro.FEATURES['cambi-aic2026'])
        rows={r['column'].split(' / ',1)[0]:r for r in aic2026.reproduction_rows([], 'a'*40)}
        self.assertEqual(rows['HDR_VDP_3']['n'],'61')
        self.assertEqual(rows['HDR_VDP_3']['status'],'passes stated crop gate')
        self.assertEqual(rows['CAMBI']['status'],'passes stated crop median gate')
        self.assertEqual(rows['proposal-Butteraugli']['status'],'fails crop median gate')
        self.assertIn('not 9618 native scores',rows['proposal-Butteraugli']['note'])

    def test_pyiqa_cards_follow_the_actual_package(self):
        by_name={c['metric']:c for c in cards()}
        for name in ('topiq-fr','topiq-fr-pipal','wadiqam-pyiqa','ahiq-pyiqa','pieapp-pyiqa','cw-ssim-pyiqa','dmm-pyiqa'):
            self.assertIn('zenmetrics-pyiqa-ports/models/',by_name[name]['source'])
            self.assertIn('Noncommercial',by_name[name]['rust_license'])
            self.assertEqual(by_name[name]['parity']['tolerance_jnd'],0.001)
        self.assertIn('0.071',json.dumps(by_name['dmm-pyiqa']['parity']))

    def test_pyiqa_reproduction_features_and_reference_kind(self):
        inventory={r['metric']:r for r in catalog_rows()}
        by_name={c['metric']:c for c in cards()}
        for name in ('topiq-fr','topiq-fr-pipal','wadiqam-pyiqa','ahiq-pyiqa','pieapp-pyiqa','cw-ssim-pyiqa','dmm-pyiqa'):
            self.assertEqual(inventory[name]['nr'],'false')
            self.assertIn('pyiqa-ports',by_name[name]['repro']['features'])
            self.assertFalse(any('No-reference' in note for note in by_name[name]['repro']['notes']))
        topiq=by_name['topiq-fr']['repro']
        self.assertTrue(any('zenmetrics-pyiqa-ports/models/topiq/card.toml' in line for line in topiq['lines']))
        self.assertTrue(any('ZENMETRICS_NN_ACCEPT_WEIGHTS=topiq_fr' in note for note in topiq['notes']))
        self.assertEqual(by_name['cw-ssim-pyiqa']['family'],'Classical')
        self.assertFalse(any('MODEL_DIR' in line for line in by_name['cw-ssim-pyiqa']['repro']['lines']))

    def test_ssim_fixed_factors_are_current_dispatch_metadata(self):
        variants=aic2026.ssim_dispatch_inputs()
        self.assertEqual(variants['ssim-pyiqa-f32-k6']['fixed_box_factor'],6)
        self.assertEqual(variants['ssim-pyiqa-f32-k6']['rounding'],'unrounded')
        self.assertEqual(variants['ssim-pyiqa']['rounding'],'ties-to-even before downscale')
        self.assertEqual(variants['ms-ssim-pyiqa-f32-k2']['scales'],5)
        self.assertIn('f64',variants['ssimc-pyiqa-f32-k3']['arithmetic'])
        mapping=json.loads((ROOT/'scripts/complexity/aic2026_columns.json').read_text())
        self.assertEqual(mapping['map']['VMAF'],'vmaf-studio601')
        self.assertEqual(mapping['map']['VMAF-neg'],'vmaf-neg-studio601')

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
        self.assertIn('grayscale route differs',body)
        self.assertIn('Ponomarenko',body)
        self.assertIn('studio BT.601',body)
        self.assertIn('0 of 11 Haar variants identified',body)
        self.assertIn('figure does not label that statistic',body)

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

    def test_landing_refresh_preserves_component_sources_and_assets(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            viewer=Path(d)
            index='<html><head></head><body><header><h1>Viewer</h1></header></body></html>'
            (viewer/'index.html').write_text(index)
            (viewer/'worker.js').write_text('unchanged')
            original={'source_commit':'a'*40,'aic2026':{'source_commit':'b'*40},
                      'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in viewer.iterdir()}}
            (viewer/'site-source.json').write_text(json.dumps(original))
            refresh_landing(viewer,'c'*40)
            result=json.loads((viewer/'site-source.json').read_text())
            self.assertEqual(result['source_commit'],original['source_commit'])
            self.assertEqual(result['aic2026'],original['aic2026'])
            self.assertEqual(result['files']['worker.js'],original['files']['worker.js'])
            self.assertEqual(result['landing']['source_commit'],'c'*40)
            self.assertIn('https://gigapixels.imazen.org/',(viewer/'index.html').read_text())
            with self.assertRaisesRegex(ValueError,'already present'):refresh_landing(viewer,'c'*40)
            (viewer/'worker.js').write_text('changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):refresh_landing(viewer,'c'*40)

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
            self.assertIn('href="https://gigapixels.imazen.org/"',(viewer/'index.html').read_text())
            self.assertEqual((viewer/'worker.js').read_text(),'existing-worker')
            self.assertIn('aic2026/manifest.json',json.loads((viewer/'site-source.json').read_text())['files'])


if __name__=='__main__':unittest.main()
