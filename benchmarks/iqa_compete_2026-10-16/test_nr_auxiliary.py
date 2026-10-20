import hashlib
from pathlib import Path
import tempfile
import unittest

from check_nr_auxiliary import reference_scores, validate_bindings


class AuxiliaryControls(unittest.TestCase):
    def test_retains_each_reference_observation(self):
        self.assertEqual(reference_scores('COLD 1 4\nPAIR_REFERENCE 2\nSAMPLE 3 4\nREFERENCE_SCORE 2.5\n'), [2, 2.5])

    def test_missing_and_nonfinite_reference_scores_fail(self):
        for text in ['SAMPLE 1 2', 'PAIR_REFERENCE nan', 'REFERENCE_SCORE inf']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                reference_scores(text)

    def test_relocation_preserves_old_preparer_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original=root/'prepare_case.py';historical=root/'retained.py'
            original.write_bytes(b'new');historical.write_bytes(b'old')
            plan={'bindings':{str(original):hashlib.sha256(b'old').hexdigest()},'cases':[{'canonical':{'command':['runtime.py']}}]}
            with self.assertRaises(ValueError):validate_bindings(plan,{})
            validate_bindings(plan,{str(original):str(historical)})
            historical.write_bytes(b'different')
            with self.assertRaises(ValueError):validate_bindings(plan,{str(original):str(historical)})

    def test_runtime_artifacts_cannot_be_relocated(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original=root/'runtime.py';historical=root/'retained.py'
            original.write_bytes(b'new');historical.write_bytes(b'old')
            plan={'bindings':{str(original):hashlib.sha256(b'old').hexdigest()},'cases':[{'canonical':{'command':[str(original)]}}]}
            with self.assertRaises(ValueError):validate_bindings(plan,{str(original):str(historical)})
            preparer=root/'prepare_runtime.py';plan['bindings']={str(preparer):hashlib.sha256(b'old').hexdigest()}
            plan['cases'][0]['canonical']['command']=[str(preparer)]
            with self.assertRaises(ValueError):validate_bindings(plan,{str(preparer):str(historical)})


if __name__=='__main__':unittest.main()
