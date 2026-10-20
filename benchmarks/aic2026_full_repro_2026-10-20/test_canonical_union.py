"""The cropped comparison cannot silently mix the two published views."""
from pathlib import Path
import tempfile
import unittest

from canonical_run import sha
from compare_fsim_vsi import published_crops, same_reference_metadata


class PublishedUnion(unittest.TestCase):
    def test_reference_bpp_formatting_preserves_exact_decimal_value(self):
        row = dict(distorted='same.png', bpp='1.060003890')
        self.assertTrue(same_reference_metadata(dict(row, bpp='1.06000389'), row, list(row)))
        self.assertFalse(same_reference_metadata(dict(row, bpp='1.060003891'), row, list(row)))
        self.assertFalse(same_reference_metadata(dict(row, distorted='other.png'), row, list(row)))

    def test_view_selection_and_duplicate_and_hash_refusals(self):
        scratch = Path.home()/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory)/'union.csv'
            path.write_text('view,distorted,FSIM\nfullres,same.png,0.1\ncropped,same.png,0.9\n')
            self.assertEqual(published_crops(path, sha(path))['same.png']['FSIM'], '0.9')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                published_crops(path, '0'*64)
            path.write_text(path.read_text()+'cropped,same.png,0.8\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                published_crops(path, sha(path))

    def test_missing_view_refuses_ambiguous_reference(self):
        scratch = Path.home()/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory)/'union.csv'
            path.write_text('distorted,FSIM\nsame.png,0.9\n')
            with self.assertRaisesRegex(ValueError, 'view column'):
                published_crops(path, sha(path))


if __name__ == '__main__':
    unittest.main()
