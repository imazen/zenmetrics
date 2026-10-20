"""Resident scoring must use decoded arrays instead of the old file argv."""
from pathlib import Path
import tempfile
import unittest

from canonical_python import resident_adapter


class AdapterInputs(unittest.TestCase):
    def test_tuple_raw_input_load_is_removed_and_other_setup_preserved(self):
        scratch = Path.home() / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / 'cell.py'
            path.write_text('''
def main():
    r, d = [np.fromfile(path, dtype=np.uint8).reshape(a.height, a.width, 3)
            for path in (a.reference, a.distorted)]
    left, right = (2, 3)
    def score():
        return r - d + left + right
    for index in range(a.samples + 1):
        raise AssertionError('timing loop must not run')
''')
            score = resident_adapter({'path': str(path)}, 'fixture')
            self.assertEqual(score(10, 4), 11.0)
            self.assertEqual(score(7, 6), 6.0)


if __name__ == '__main__':
    unittest.main()
