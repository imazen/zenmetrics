"""Negative controls for export qualification; no model runtimes needed."""
import math
from pathlib import Path
import runpy
import tempfile
import unittest

checks = runpy.run_path(str(Path(__file__).with_name('export_checks.py')))


class ExportChecks(unittest.TestCase):
    def test_mismatch_and_nonfinite_refused(self):
        for model, scale in [('musiq', 1.968925), ('hyperiqa', 4.461776)]:
            limit = scale * 1e-3
            self.assertEqual(checks['score_tolerance'](model), limit)
            self.assertEqual(checks['assert_parity'](model, limit, 0.0), limit)
            for got in [limit * 1.01, math.nan, math.inf]:
                with self.assertRaises(AssertionError):
                    checks['assert_parity'](model, got, 0.0)

    def test_missing_image_and_empty_goldens_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'goldens.tsv'
            path.write_text('image\tsize\thyperiqa\thyperiqa_ref\n')
            with self.assertRaises(ValueError):
                checks['golden_rows'](path, folder, 'hyperiqa')
            path.write_text('image\tsize\thyperiqa\thyperiqa_ref\nx\t64\t1\t2\n')
            with self.assertRaises(FileNotFoundError):
                checks['golden_rows'](path, folder, 'hyperiqa')
            for kind in ['ref', 'dist']:
                (Path(folder) / f'x_64_{kind}.png').touch()
            self.assertEqual(len(checks['golden_rows'](path, folder, 'hyperiqa')), 1)


if __name__ == '__main__':
    unittest.main()
