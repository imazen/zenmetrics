"""Evidence identity, missing-field and shared-output controls."""
import csv
import io
import json
from pathlib import Path
import unittest
from catalog import rows as catalog_rows
from model_cards import cards, exports, readme_section, documentation


class ModelCards(unittest.TestCase):
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

    def test_historical_family_join_is_explicit_and_version_scoped(self):
        from build import HISTORICAL_QUALITY_FAMILY
        self.assertEqual(HISTORICAL_QUALITY_FAMILY['score_zensim_v0_16'],'zensim')
        self.assertEqual(HISTORICAL_QUALITY_FAMILY['score_cvvdp'],'cvvdp')
        self.assertNotIn('score_zensim_future',HISTORICAL_QUALITY_FAMILY)
