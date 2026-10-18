"""Default scheduling omits costly routes, while explicit requests remain intact."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import runpy
import subprocess

from selection import select_metrics


class Selection(unittest.TestCase):
    def test_default_and_all_report_both_omissions(self):
        inventory = {'pieapp', 'pieapp-pyiqa', 'psnr', 'dists'}
        for request in (None, 'all'):
            with self.subTest(request=request), contextlib.redirect_stderr(io.StringIO()) as log:
                self.assertEqual(select_metrics(request, inventory), {'psnr', 'dists'})
                self.assertIn('pieapp,pieapp-pyiqa', log.getvalue())
                self.assertIn('explicit selection required', log.getvalue())

    def test_learned_group_omits_costly_routes(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(select_metrics('learned', {'psnr'}, learned={
                'pieapp', 'pieapp-pyiqa', 'dists'}), {'dists'})

    def test_explicit_selection_and_unknown_validation_are_preserved(self):
        for request in ('pieapp', 'pieapp-pyiqa', 'psnr,pieapp,pieapp-pyiqa', 'unknown'):
            with self.subTest(request=request), contextlib.redirect_stderr(io.StringIO()) as log:
                self.assertEqual(select_metrics(request, {'psnr'}), set(request.split(',')))
                self.assertEqual(log.getvalue(), '')

    def test_other_groups_are_unchanged(self):
        with contextlib.redirect_stderr(io.StringIO()) as log:
            self.assertEqual(select_metrics(None, {'psnr', 'dists'}), {'psnr', 'dists'})
            self.assertEqual(log.getvalue(), '')

    def test_direct_controller_default_all_learned_and_explicit_dispatch(self):
        import run
        inventory = [dict(metric_name=name, gpu=False, columns=[]) for name in
                     ['psnr', 'dists', 'pieapp', 'pieapp-pyiqa']]
        for request, expected in [(None, ['psnr', 'dists']), ('all', ['psnr', 'dists']),
                                  ('learned', ['dists']), ('pieapp,pieapp-pyiqa', ['pieapp', 'pieapp-pyiqa'])]:
            with self.subTest(request=request), tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as temp:
                root = Path(temp)
                (root / 'pixels').write_bytes(b'controlled input; mocked metric worker')
                pairs = root / 'pairs.json'
                pairs.write_text(json.dumps([dict(size_name='cid22', reference='pixels', distorted='pixels')]))
                argv = ['run.py', 'unused', str(pairs), str(root / 'output'), '--threads', '1']
                if request is not None:
                    argv += ['--metrics', request]
                with patch.object(sys, 'argv', argv), patch.object(run, 'metadata', return_value=dict(host='i265', os='Linux')), \
                     patch.object(run.subprocess, 'check_output', return_value=json.dumps(inventory)), \
                     patch.object(run, 'LEARNED', {'dists', 'pieapp', 'pieapp-pyiqa'}), \
                     patch.object(run, 'run_cell', return_value=[]) as cell, contextlib.redirect_stderr(io.StringIO()):
                    run.main()
                self.assertEqual([call.args[1]['metric_name'] for call in cell.call_args_list], expected)

    def test_coverage_launcher_does_not_add_an_explicit_pieapp_lane(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as temp:
            root = Path(temp)
            binary = root / 'worker'
            binary.write_bytes(b'launcher fixture')
            pairs = root / 'pairs.json'
            pairs.write_text('[]')
            argv = ['coverage_pass.py', str(binary), str(pairs), str(root / 'output'),
                    str(root / 'models'), str(root / 'model-map.json'),
                    '--ort-library', 'unused', '--cuda-libraries', 'unused']
            with patch.object(sys, 'argv', argv), patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0)) as launch, \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                runpy.run_path(str(Path(__file__).with_name('coverage_pass.py')), run_name='__main__')
            self.assertEqual(result.exception.code, 0)
            manifest = json.loads((root / 'output/launch.json').read_text())
            self.assertEqual(set(manifest['commands']), {'cpu', 'cuda', 'nn-cuda'})
            self.assertEqual(launch.call_count, 3)
            for command in manifest['commands'].values():
                self.assertNotIn('--metrics', command)


if __name__ == '__main__':
    unittest.main()
