"""Input inventories must reject changed bytes and never skip failed reads."""
import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from canonical_input_hashes import stable_hash
from canonical_run import sha


class InputHashes(unittest.TestCase):
    def test_errno12_retries_whole_file_then_returns_actual_digest(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            path = Path(directory)/'input'
            path.write_text('retained bytes\n')
            expected = sha(path)
            with patch('canonical_input_hashes.sha', side_effect=[OSError(errno.ENOMEM, 'read failed'), expected]) as reader:
                with patch('canonical_input_hashes.time.sleep'):
                    self.assertEqual(stable_hash(path), expected)
            self.assertEqual(reader.call_args_list[0], reader.call_args_list[1])

    def test_exhausted_errno12_is_a_failure(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            path = Path(directory)/'input'
            path.write_text('retained bytes\n')
            with patch('canonical_input_hashes.sha', side_effect=OSError(errno.ENOMEM, 'read failed')) as reader:
                with patch('canonical_input_hashes.time.sleep'):
                    with self.assertRaises(OSError):
                        stable_hash(path)
            self.assertEqual(reader.call_count, 3)

    def test_changed_file_and_missing_file_fail(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            path = Path(directory)/'input'
            path.write_text('initial')
            def changed(source):
                source.write_text('different length')
                return '0'*64
            with patch('canonical_input_hashes.sha', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'changed while hashing'):
                    stable_hash(path)
            with self.assertRaises(FileNotFoundError):
                stable_hash(Path(directory)/'missing')


if __name__ == '__main__':
    unittest.main()
