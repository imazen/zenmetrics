from pathlib import Path
import tempfile
import unittest

from resume_canonical_comparison import unused_attempt


class RecoveryPaths(unittest.TestCase):
    def test_log_only_and_partial_attempts_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as name:
            root = Path(name)
            original = root/'block-04-wrapper.log'
            original.write_text('interrupted before child started')
            partial = root/'block-04-recovery-1'
            partial.mkdir()
            self.assertEqual(unused_attempt(root, 4), root/'block-04-recovery-2')
            self.assertEqual(original.read_text(), 'interrupted before child started')
            self.assertTrue(partial.is_dir())
