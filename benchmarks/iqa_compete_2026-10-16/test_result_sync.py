import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sync_saved_results import DESTINATION, FILES, fingerprint, public_checks, publish, recover_published, guarded_push


class SyncControls(unittest.TestCase):
    def test_initial_stale_tip_retains_forward_rebase_recovery(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            with patch('sync_saved_results.run', side_effect=['new', '', 'new', '', 'rebased', 'new']), patch(
                    'sync_saved_results.subprocess.run', side_effect=[subprocess.CompletedProcess([], 3),
                    subprocess.CompletedProcess([], 0)]):
                self.assertEqual(guarded_push(repo, 'snapshot', repo / 'push.log'), 'rebased')

    def test_post_guard_remote_advance_retries_every_guard(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            with patch('sync_saved_results.run', side_effect=['old', '', 'new', '', 'rebased', 'new']) as commands, patch(
                    'sync_saved_results.subprocess.run', side_effect=[subprocess.CompletedProcess([], 5),
                    subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 0)]) as child:
                self.assertEqual(guarded_push(repo, 'snapshot', repo / 'push.log'), 'rebased')
                self.assertEqual(child.call_args_list[0].args[0], ['bash', 'scripts/safe_push.sh', '-r', 'snapshot'])
                self.assertEqual(child.call_args_list[-1].args[0], ['bash', 'scripts/safe_push.sh', '-r', 'rebased'])
                self.assertIn(['jj', 'rebase', '-r', 'snapshot', '-d', 'master@origin'],
                              [c.args[0] for c in commands.call_args_list])

    def test_transport_failure_and_nonforward_remote_never_rebase(self):
        import subprocess
        for after, ancestry in [('old', 0), ('sideways', 1)]:
            with self.subTest(after=after), tempfile.TemporaryDirectory() as d:
                repo = Path(d)
                with patch('sync_saved_results.run', side_effect=['old', '', after]) as commands, patch(
                        'sync_saved_results.subprocess.run', side_effect=[subprocess.CompletedProcess([], 5),
                        subprocess.CompletedProcess([], ancestry)]):
                    with self.assertRaises(subprocess.CalledProcessError):
                        guarded_push(repo, 'snapshot', repo / 'push.log')
                    self.assertFalse(any(c.args[0][:2] == ['jj', 'rebase'] for c in commands.call_args_list))

    def test_guard_refusals_are_never_retried(self):
        import subprocess
        for code in (7, 8, 9):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as d:
                repo = Path(d)
                with patch('sync_saved_results.run', return_value='old') as commands, patch(
                        'sync_saved_results.subprocess.run', return_value=subprocess.CompletedProcess([], code)) as child:
                    with self.assertRaises(subprocess.CalledProcessError):
                        guarded_push(repo, 'snapshot', repo / 'push.log')
                    self.assertEqual(commands.call_count, 1)
                    self.assertEqual(child.call_count, 1)

    def test_repeated_races_stop_after_three_guarded_attempts(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            with patch('sync_saved_results.run', side_effect=['a', '', 'b', '', 'v2', 'b', '', 'c', '', 'v3', 'c']), patch(
                    'sync_saved_results.subprocess.run', side_effect=[subprocess.CompletedProcess([], c)
                    for c in (5, 0, 5, 0, 5)]) as child:
                with self.assertRaises(subprocess.CalledProcessError):
                    guarded_push(repo, 'snapshot', repo / 'push.log')
                self.assertEqual(sum(c.args[0][0] == 'bash' for c in child.call_args_list), 3)

    def test_recovery_requires_exact_published_bytes_and_remote_ancestry(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            folder = repo / DESTINATION
            folder.mkdir(parents=True)
            for host in ('mac', 'i265', 'r7900x'):
                (folder / (host + '-observations.csv')).write_text(
                    'protocol,only_mode,threads,width,height\ncanonical-cold3-warm10-100ms-scalar,score,8,512,512\n')
            def freeze():
                (folder / 'SHA256SUMS.json').write_text(json.dumps({f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in folder.glob('*.csv')}))
            freeze()
            committed = {p.name:p.read_bytes() for p in folder.iterdir()}
            def show(argv, **kwargs):
                return committed[argv[-1].rsplit('/', 1)[-1]]
            revision = 'a' * 40
            with patch('sync_saved_results.subprocess.run') as ancestry, patch('sync_saved_results.subprocess.check_output', side_effect=show):
                result = recover_published(repo, revision)
                self.assertEqual(result['revision'], revision)
                ancestry.assert_called_once_with(['git', 'merge-base', '--is-ancestor', revision, 'origin/master'], cwd=repo, check=True)
                target = folder / 'mac-observations.csv'
                target.write_text(target.read_text().replace(',8,', ',1,'))
                freeze()
                with self.assertRaisesRegex(ValueError, 'working data differs'):
                    recover_published(repo, revision)
            with patch('sync_saved_results.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'ancestry')):
                with self.assertRaises(subprocess.CalledProcessError):
                    recover_published(repo, revision)

    def test_staged_capture_has_its_sealed_validation_dependencies(self):
        import shutil
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in FILES:
                shutil.copy2(Path(__file__).parent / name, root / name)
            subprocess.run([sys.executable, '-c', 'import capture_saved_results; import durable'],
                           cwd=root, check=True)

    def test_capture_time_alone_does_not_create_another_commit(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            f = p / 'mac-observations.csv'
            f.write_text('metric_name,median_ns,capture_utc,source\nx,10,old,a\n')
            first = fingerprint(p)
            f.write_text('metric_name,median_ns,capture_utc,source\nx,10,new,b\n')
            self.assertEqual(first, fingerprint(p))
            f.write_text('metric_name,median_ns,capture_utc,source\nx,11,new,b\n')
            self.assertNotEqual(first, fingerprint(p))

    def test_foreign_owner_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / '.workongoing').write_text('another lane\n')
            result = publish(p, p, 'digest', p)
            self.assertIn('another lane', result['state'])
            self.assertEqual((p / '.workongoing').read_text(), 'another lane\n')

    def test_dirty_checkout_is_never_modified_or_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            with patch('sync_saved_results.run', return_value='M user-source.rs') as run:
                result = publish(p, p, 'digest', p)
            self.assertIn('uncommitted', result['state'])
            self.assertEqual(run.call_count, 1)
            self.assertFalse((p / '.workongoing').exists())

    def test_projection_checks_detect_tamper_and_excluded_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            for host in ('mac', 'i265', 'r7900x'):
                (p / (host + '-observations.csv')).write_text(
                    'protocol,only_mode,threads,width,height\ncanonical-cold3-warm10-100ms-scalar,score,8,512,512\n')
            def freeze():
                (p / 'SHA256SUMS.json').write_text(json.dumps({f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in p.glob('*.csv')}))
            freeze()
            public_checks(p)
            target = p / 'mac-observations.csv'
            target.write_text(target.read_text().replace('512,512', '7680,4320'))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                public_checks(p)
            freeze()
            with self.assertRaisesRegex(ValueError, 'geometry'):
                public_checks(p)
