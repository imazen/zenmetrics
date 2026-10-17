"""Deploy regeneration tolerates stale copies; unrelated uploads stay gated."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from build import validate_inventory
from regenerate import regenerate, refresh


class Regeneration(unittest.TestCase):
    def test_stale_shards_refresh_and_unknown_upload_still_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            directory = root / 'benchmarks/dashboard/catalog'
            directory.mkdir(parents=True)
            (directory / 'catalog-00.tsv').write_text('stale')
            (directory / 'catalog-01.tsv').write_text('obsolete')
            # This is not owned by the generator, even inside its directory.
            unknown = directory / 'new-benchmark.csv'
            unknown.write_text('metric,ms\nexample,1\n')
            expected = {'catalog-00.tsv': 'fresh', 'catalog-02.tsv': 'new shard'}
            self.assertEqual(refresh(directory, expected, r'catalog-\d+\.tsv'), (1, 1, 1))
            self.assertEqual((directory / 'catalog-00.tsv').read_text(), 'fresh')
            self.assertFalse((directory / 'catalog-01.tsv').exists())
            self.assertTrue(unknown.is_file())
            self.assertEqual(refresh(directory, expected, r'catalog-\d+\.tsv'), (0, 0, 0))
            (root / 'benchmarks/dashboard/classification.tsv').write_text('path\tclassification\treason\n')
            registered = {f'benchmarks/dashboard/catalog/{name}' for name in expected}
            with self.assertRaisesRegex(ValueError, 'unclassified'):
                validate_inventory(root, registered)

    def test_summary_reports_both_stale_and_current_copies(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            summary = root / 'summary.md'
            with patch('regenerate.generate_model_cards'), patch('regenerate.ROOT', root), \
                 patch('regenerate.catalog_render', return_value={'catalog-00.tsv': 'catalog'}), \
                 patch('regenerate.markdown_render', return_value={'tables-00.tsv': 'table', 'sources.json': '{}'}), \
                 patch.dict(os.environ, {'GITHUB_STEP_SUMMARY': str(summary)}):
                with redirect_stdout(io.StringIO()) as output:
                    regenerate()
                self.assertIn('| Metric catalog | yes | 1 | 0 | 0 |', output.getvalue())
                self.assertIn('| Markdown extracts | yes | 2 | 0 | 0 |', summary.read_text())
                with redirect_stdout(io.StringIO()) as output:
                    regenerate()
                self.assertIn('| Metric catalog | no | 0 | 0 | 0 |', output.getvalue())
                self.assertIn('| Markdown extracts | no | 0 | 0 | 0 |', output.getvalue())

    def test_invalid_sources_still_fail_before_replacing_copies(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            directory = root / 'benchmarks/dashboard/catalog'
            directory.mkdir(parents=True)
            (directory / 'catalog-00.tsv').write_text('committed copy')
            with patch('regenerate.generate_model_cards'), patch('regenerate.ROOT', root), \
                 patch('regenerate.catalog_render', return_value={'catalog-00.tsv': 'fresh'}), \
                 patch('regenerate.markdown_render', side_effect=ValueError('malformed source')):
                with self.assertRaisesRegex(ValueError, 'malformed source'):
                    regenerate()
            self.assertEqual((directory / 'catalog-00.tsv').read_text(), 'committed copy')
