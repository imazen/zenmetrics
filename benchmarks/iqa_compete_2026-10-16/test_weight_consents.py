"""Native name propagation requires explicit caller acceptance, never a default."""
import unittest
from prepare_nn_durable import resolve_weight_consents


class WeightConsents(unittest.TestCase):
    def test_default_grants_nothing_even_for_gated_models(self):
        self.assertEqual(resolve_weight_consents('', ['afine','ckdn']), '')

    def test_explicit_all_expands_only_selected_named_terms(self):
        self.assertEqual(resolve_weight_consents('all', ['afine','ckdn','musiq','lpips-alex']), 'afine,ckdn')
        self.assertEqual(resolve_weight_consents('all', ['lpips-alex']), '')

    def test_explicit_names_are_preserved_and_canonical(self):
        self.assertEqual(resolve_weight_consents(' ckdn,afine,ckdn,pieapp ', ['afine']), 'afine,ckdn,pieapp')

    def test_typo_and_mixed_wildcard_refuse(self):
        for names in ['afnie','all,afine','*']:
            with self.assertRaisesRegex(ValueError, 'unknown explicit'):
                resolve_weight_consents(names, ['afine'])


if __name__ == '__main__': unittest.main()
