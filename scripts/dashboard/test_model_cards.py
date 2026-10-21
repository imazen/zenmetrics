"""Evidence identity, missing-field and shared-output controls."""
import csv
import io
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from catalog import rows as catalog_rows
from model_cards import cards, exports, readme_section, documentation, comparison_results


class ModelCards(unittest.TestCase):
    def test_cold_numerical_miss_overrides_prior_pass_in_compact_check(self):
        with patch('model_cards.comparison_results',return_value={'brisque':dict(
                canonical='PIQ 0.8.0',check='170/170 pass',ratio=.12)}), \
             patch('model_cards.cold_batch_numerical_misses',return_value={'brisque':2}):
            text=readme_section([self.by_name['brisque']])
        compact=next(line for line in text.splitlines() if '| NR | PIQ' in line)
        self.assertIn('2 cold misses',compact)
        self.assertIn('≈0.12×',compact)
        self.assertNotIn('170/170 pass',compact)

    def test_cold_misses_separate_score_gates_from_process_errors(self):
        from model_cards import cold_numerical_misses
        (Path.home()/'tmp').mkdir(parents=True, exist_ok=True)  # a fresh CI runner has no ~/tmp
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);shard=root/'cold.json';index=root/'index.json'
            cases=[dict(case='finite',metric='example',native_source='rev',
                        absolute_bound=.01,relative_bound=0,blocks=[
                            dict(delta=.02,arms={'canonical':{'score':1}}),
                            dict(delta=.02,arms={'canonical':{'score':1}})]),
                   dict(case='refusal',metric='example',native_source='rev',
                        absolute_bound=.01,relative_bound=0,blocks=[
                            dict(delta=None,arms={},errors={'canonical':'exit 1'})])]
            shard.write_text(json.dumps(dict(cases=cases)))
            index.write_text(json.dumps(dict(shards=[dict(file=shard.name,
                cases=2,sha256=hashlib.sha256(shard.read_bytes()).hexdigest())])))
            self.assertEqual(cold_numerical_misses(index),{'example':1})
            shard.write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                cold_numerical_misses(index)

    def test_cold_batches_reject_duplicate_cases_and_changed_indexes(self):
        from model_cards import cold_batch_numerical_misses
        (Path.home()/'tmp').mkdir(parents=True, exist_ok=True)  # a fresh CI runner has no ~/tmp
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);shard=root/'cases.json';index=root/'index.json'
            case=dict(case='same',metric='example',native_source='rev',
                      absolute_bound=.01,relative_bound=0,blocks=[
                          dict(delta=.02,arms={'canonical':{'score':1}})])
            shard.write_text(json.dumps(dict(cases=[case])))
            index.write_text(json.dumps(dict(shards=[dict(file=shard.name,cases=1,
                sha256=hashlib.sha256(shard.read_bytes()).hexdigest())])))
            batch=dict(file=index.name,sha256=hashlib.sha256(index.read_bytes()).hexdigest())
            manifest=root/'batches.json';manifest.write_text(json.dumps(dict(batches=[batch])))
            self.assertEqual(cold_batch_numerical_misses(manifest),{'example':1})
            manifest.write_text(json.dumps(dict(batches=[batch,batch])))
            with self.assertRaisesRegex(ValueError,'duplicate cold case'):
                cold_batch_numerical_misses(manifest)
            manifest.write_text(json.dumps(dict(batches=[batch])))
            index.write_text('{}')
            with self.assertRaisesRegex(ValueError,'index hash mismatch'):
                cold_batch_numerical_misses(manifest)

    def test_comparative_shards_reject_changed_evidence(self):
        (Path.home()/'tmp').mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);shard=root/'results-00.json'
            shard.write_text(json.dumps({'example':{'ratio':.5}}))
            index=root/'results.json'
            index.write_text(json.dumps(dict(schema=2,metrics=1,shards=[dict(
                file=shard.name,metrics=1,sha256=hashlib.sha256(shard.read_bytes()).hexdigest())])))
            self.assertEqual(comparison_results(index)['example']['ratio'],.5)
            shard.write_text(json.dumps({'example':{'ratio':5}}))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                comparison_results(index)

    @classmethod
    def setUpClass(cls):cls.items=list(cards());cls.by_name={c['metric']:c for c in cls.items}

    def test_cli_membership_and_missing_citations(self):
        self.assertEqual(set(self.by_name),{r['metric'] for r in catalog_rows()})
        self.assertEqual(len(self.by_name),len(self.items))
        for row in catalog_rows():
            if row['nn']=='true':self.assertEqual(self.by_name[row['metric']]['family'],'Learned')
        psnr=self.by_name['psnr']
        self.assertIsNone(psnr['citation'])
        self.assertEqual(psnr['citation_status'],'citation needed')
        self.assertIn('citation needed',documentation([psnr]))

    def test_publisher_credits_are_not_source_card_shorthand(self):
        afine=self.by_name['afine']['citation']
        self.assertIn('Perfect Reference',afine['title'])
        self.assertIn('Chen, Du',afine['authors'])
        self.assertIn('Fu, Stephanie',self.by_name['dreamsim']['citation']['authors'])
        self.assertIn('Hyper Network',self.by_name['hyperiqa']['citation']['title'])
        self.assertNotIn('Gini',self.by_name['hyperiqa']['citation']['title'])

    def test_exact_named_original_cards_keep_their_evidence_and_limits(self):
        self.assertIn('IIGROUP/AHIQ',self.by_name['ahiq']['reference_implementation'])
        self.assertEqual(self.by_name['ahiq']['parity_status'],'Reported results')
        self.assertIn('6/24',self.by_name['dmm']['parity']['results'])
        self.assertEqual(self.by_name['dmm']['parity_status'],'Partial canonical parity')
        self.assertIn('measured_ladder',self.by_name['wadiqam-pyiqa']['parity'])
        self.assertEqual(self.by_name['wadiqam-pyiqa']['parity_status'],'Reported results')

    def test_local_pass_does_not_erase_broader_partial_parity(self):
        for metric in ['dmm', 'dmm-pyiqa']:
            with self.subTest(metric=metric):
                self.assertEqual(self.by_name[metric]['parity_status'],'Partial canonical parity')
                with patch('model_cards.comparison_results',return_value={metric:dict(check='2/2 pass',ratio=.5)}):
                    text=readme_section([self.by_name[metric]])
                self.assertIn('partial; 2/2 pass',text)

    def test_compact_readme_keeps_the_selected_cuda_reference_visible(self):
        with patch('model_cards.comparison_results',return_value={'cvvdp-gpu':dict(
                canonical='pycvvdp 0.5.7 / original callable',check='2/2 pass',ratio=.5,
                ratio_canonical_device='cuda')}):
            text=readme_section([self.by_name['cvvdp-gpu']])
        self.assertIn('| CUDA pycvvdp 0.5.7',text)
        self.assertIn('≈0.5×',text)

    def test_checkpoint_licence_and_oracle_scopes_survive(self):
        card=self.by_name['pieapp']
        self.assertIn('Non-Commercial',card['weights_license'])
        self.assertFalse(card['conventions']['oracle_convention']['cli_equivalence'])
        self.assertIn('tolerance_jnd',card['parity'])
        self.assertEqual(self.by_name['arniqa']['kind'],'NR')
        self.assertEqual(self.by_name['cambi']['kind'],'NR')
        self.assertEqual(self.by_name['wadiqam-nr-live']['kind'],'NR')
        self.assertIn('LicenseRef-Imazen-Commercial',self.by_name['niqe']['rust_license'])
        self.assertEqual(self.by_name['niqe']['weights_license'],'not recorded')
        self.assertFalse(any('afine' in p for p in self.by_name['lpips-alex']['license_evidence']))

    def test_all_exports_and_readme_use_same_identity(self):
        exported=exports(self.items)
        self.assertEqual(len(json.loads(exported['model-cards.json'])),len(self.items))
        table=list(csv.DictReader(io.StringIO(exported['model-cards.csv'])))
        self.assertEqual({r['metric'] for r in table},set(self.by_name))
        bibliography=exported['model-cards.bib']
        self.assertEqual(bibliography.count('@misc{'),len({c['citation']['url'] for c in self.items if c['citation']}))
        readme=readme_section(self.items)
        for metric in self.by_name:self.assertIn('`'+metric+'`',readme)

    def test_decimal_deltas_preserve_small_values_and_missing_medians(self):
        from model_cards import decimal_delta, readme_variant_tables
        self.assertEqual(decimal_delta(3e-5), '0.00003')
        self.assertEqual(decimal_delta(0.0), '0.0')
        self.assertEqual(decimal_delta(None), 'not recorded')
        text=readme_variant_tables([self.by_name['ssim2']],
            {'ssim2': {'canonical': 'explicit reference', 'max_abs_delta': 3e-5}})
        self.assertIn('0.00003 | not recorded', text)
        self.assertIn('explicit reference', text)
        self.assertIn('Reference code / weights', text)

    def test_historical_family_join_is_explicit_and_version_scoped(self):
        from build import HISTORICAL_QUALITY_FAMILY
        self.assertEqual(HISTORICAL_QUALITY_FAMILY['score_zensim_v0_16'],'zensim')
        self.assertEqual(HISTORICAL_QUALITY_FAMILY['score_cvvdp'],'cvvdp')
        self.assertNotIn('score_zensim_future',HISTORICAL_QUALITY_FAMILY)


class ReproAndUnimplemented(unittest.TestCase):
    def test_repro_features_come_from_cli_source(self):
        import repro
        from catalog import rows
        by_metric = {r['metric']: r for r in rows()}
        self.assertNotIn('--features', repro.commands('ssim2', by_metric['ssim2'], {})['lines'][2])
        lpips = repro.commands('lpips-alex', by_metric['lpips-alex'], {'_card_dir': 'lpips'})
        self.assertIn('--features nn', lpips['lines'][2])
        self.assertTrue(any('ZENMETRICS_NN_MODEL_DIR' in line for line in lpips['lines']))
        self.assertIn('--display-model', repro.commands('cvvdp', by_metric['cvvdp'], {})['lines'][-1])

    def test_unimplemented_methods_are_not_in_the_catalog(self):
        from build import unimplemented_methods
        from catalog import rows
        names = {r['metric'] for r in rows()}
        methods = unimplemented_methods()
        self.assertTrue(methods)
        for m in methods:
            self.assertNotIn(m['name'].lower(), names)
            self.assertTrue(m['url'].startswith('https://github.com/imazen/zenmetrics/blob/master/docs/'))
