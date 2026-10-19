"""Check recipe admission, fixed sample protocol and immutable output selection."""
import fcntl
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import canonical_bench as runner
import durable
from types import SimpleNamespace


class CanonicalCommands(unittest.TestCase):
    def test_initial_asset_hashing_holds_shared_admission(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            lock = Path(tmp) / 'heavy.lock'
            def preflight(_):
                with lock.open('a+') as probe:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                raise ValueError('fixture stops before measurements')
            with patch.object(durable, 'manifest', side_effect=preflight):
                with self.assertRaisesRegex(ValueError, 'fixture stops'):
                    durable.run(SimpleNamespace(lock=str(lock), manifest='fixture', output=str(Path(tmp)/'output')))
            with lock.open('a+') as probe:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_fixed_protocol_and_shared_lock_released_before_durable(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            argv = ['canonical_bench.py', runner.SIZES, 'changed', '--source-commit', 'fixture']
            for name in ('binary', 'pairs', 'build-receipt', 'bundle', 'output', 'lock', 'run-heavy'):
                argv += ['--' + name, str(root / name)]
            calls = []
            def execute(command, check):
                calls.append(command)
                with (root / 'lock').open('a+') as probe:
                    if len(calls) == 1:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    else:
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch('sys.argv', argv), patch.object(runner.subprocess, 'run', side_effect=execute):
                runner.main()
            prepare, run = calls
            for flag, value in [('--modes', 'score'), ('--threads', '1,8'), ('--cold-repeats', '3'), ('--sizes', runner.SIZES)]:
                self.assertEqual(prepare[prepare.index(flag) + 1], value)
            self.assertEqual(len(prepare[prepare.index('--metrics') + 1].split(',')), 27)
            self.assertEqual(run[run.index('--memory') + 1], '12G')
            self.assertEqual(run[run.index('--jobs') + 1], '8')
            (root / 'output').mkdir()
            with patch('sys.argv', argv), patch.object(runner.subprocess, 'run') as launch:
                with self.assertRaises(SystemExit):
                    runner.main()
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
