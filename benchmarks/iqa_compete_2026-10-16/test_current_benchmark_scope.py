import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import canonical_bench as bench


class CurrentBenchmarkScope(unittest.TestCase):
    def test_current_command_contains_all_forty_routes(self):
        names = bench.changed_metrics().split(',')
        self.assertEqual(len(names), 40)
        self.assertIn('hdrvdp', names)
        self.assertIn('hdrvdp3', names)
        self.assertEqual(names[0], 'psnr')
        self.assertEqual(names[-1], 'hdrvdp3-aic2026')

    def test_missing_duplicate_unsorted_or_nonfinite_order_refuses(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            root = Path(directory)
            (root / 'existing.txt').write_text('a\nb\n')
            (root / 'new.txt').write_text('')
            order = root / 'order.tsv'
            for rows in ['a\t1\n', 'a\t1\na\t2\nb\t3\n',
                         'a\t2\nb\t1\n', 'a\tnan\nb\t2\n']:
                order.write_text('metric\thistorical_ms\n' + rows)
                with patch.multiple(bench, SCOPE=root, CURRENT_ORDER=order):
                    with self.assertRaises(ValueError):
                        bench.changed_metrics()

    def run_preparation(self, root, sizes):
        args = ['canonical_bench.py', sizes, 'psnr', '--binary', '/worker',
                '--pairs', '/pairs.json', '--build-receipt', '/build.json',
                '--source-commit', 'a' * 40, '--bundle', str(root / 'bundle'),
                '--output', str(root / 'output'), '--lock', str(root / 'lock'),
                '--run-heavy', '/run-heavy', '--prepare-only']
        with patch.object(sys, 'argv', args), patch.object(bench.subprocess, 'run') as run:
            bench.main()
        return run.call_args.args[0]

    def test_requested_size_subset_reaches_the_freezer(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            command = self.run_preparation(Path(directory), 'cid22,4k')
        self.assertEqual(command[command.index('--sizes') + 1], 'cid22,4k')

    def test_unknown_and_duplicate_sizes_refuse(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            for sizes in ['unknown', 'cid22,cid22', 'cid22,']:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        self.run_preparation(Path(directory), sizes)


if __name__ == '__main__':
    unittest.main()
