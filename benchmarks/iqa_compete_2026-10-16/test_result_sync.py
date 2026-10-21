import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sync_saved_results import (BLOCKED_EXIT, DESTINATION, FILES, KEEP_CYCLES, capture, fingerprint, main, prune, public_checks,
                                publish, push_failure, recover_published, guarded_push)


class SyncControls(unittest.TestCase):
    def run_cycle(self, root, outcome):
        """One main() cycle with capture/export stubbed; returns (exit code, status.json)."""
        import sys
        config = root / 'config.json'
        config.write_text(json.dumps(dict(scratch=str(root / 'scratch'), repo=str(root))))
        def publish_stub(repo, projection, digest, cycle, config=None):
            if isinstance(outcome, Exception):
                raise outcome
            return dict(outcome)
        with patch.object(sys, 'argv', ['sync', str(config)]), patch('sync_saved_results.run', return_value='rev'), patch(
                'sync_saved_results.subprocess.check_output', return_value=b''), patch(
                'sync_saved_results.capture'), patch('sync_saved_results.subprocess.run'), patch(
                'sync_saved_results.public_checks'), patch('sync_saved_results.fingerprint', return_value='new'), patch(
                'sync_saved_results.publish', side_effect=publish_stub):
            try:
                main()
                code = 0
            except SystemExit as exit_:
                code = exit_.code
        return code, json.loads((root / 'scratch' / 'status.json').read_text())

    def test_blocked_cycles_fail_loudly_until_a_cycle_publishes(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dirty = dict(state='captured; uncommitted repository changes left untouched', blocked=True,
                         reason='checkout has uncommitted changes', detail='M benchmarks/x')
            code, first = self.run_cycle(root, dirty)
            self.assertEqual(code, BLOCKED_EXIT)
            self.assertEqual((first['blocked_cycles'], first['first_reason']), (1, 'checkout has uncommitted changes'))
            refused = subprocess.CalledProcessError(9, 'safe_push; inspect push.log')
            code, second = self.run_cycle(root, refused)
            self.assertEqual(code, BLOCKED_EXIT)
            self.assertEqual(second['reason'], 'push refused by dashboard guard')
            self.assertEqual((second['blocked_cycles'], second['stalled_since']), (2, first['stalled_since']))
            self.assertEqual(second['first_reason'], 'checkout has uncommitted changes')
            code, third = self.run_cycle(root, dict(state='committed and remote verified', revision='r'))
            self.assertEqual(code, 0)
            self.assertNotIn('blocked', third)
            self.assertNotIn('stalled_since', third)

    def test_no_new_evidence_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'scratch').mkdir()
            (root / 'scratch' / 'last-published.json').write_text(json.dumps(dict(evidence_fingerprint='new', revision='r')))
            code, status = self.run_cycle(root, AssertionError('publish must not run'))
            self.assertEqual((code, status['state']), (0, 'no new measured evidence'))

    def test_retired_host_reuses_its_frozen_capture_without_contact(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frozen = root / 'frozen-mac'
            frozen.mkdir()
            (frozen / 'capture-receipt.json').write_text(json.dumps(dict(host='mac', capture_finished_utc='t')))
            config = dict(hosts=dict(mac=dict(ssh='mac', campaigns={}, retired=dict(since='2026-10-20', capture=str(frozen))),
                                     i265=dict(campaigns={}, retired=dict(since='x', capture=str(frozen)))))
            cycle = root / 'cycle'
            (cycle / 'captures').mkdir(parents=True)
            with patch('sync_saved_results.ssh', side_effect=AssertionError('retired host contacted')), patch(
                    'sync_saved_results.subprocess.run', side_effect=AssertionError('retired host contacted')):
                self.assertEqual(capture(config, 'mac', cycle, root)['host'], 'mac')
                self.assertEqual((cycle / 'captures' / 'mac').resolve(), frozen.resolve())
                with self.assertRaisesRegex(ValueError, 'belongs to mac'):
                    capture(config, 'i265', cycle, root)

    def test_prune_keeps_newest_cycles_and_the_published_one(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            names = ['20261020T%06d000000Z' % i for i in range(KEEP_CYCLES + 5)]
            for name in names:
                (root / name).mkdir()
            for other in ('retired', 'status.json'):
                (root / other).mkdir()
            prune(root, {names[1]})
            left = sorted(p.name for p in root.iterdir())
            self.assertEqual(left, sorted(names[-KEEP_CYCLES:] + [names[1], 'retired', 'status.json']))

    def test_push_failure_names_guard_and_keeps_log_tail(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            cycle = Path(d)
            (cycle / 'push.log').write_text('earlier\nsafe_push: 2 unclassified benchmark uploads\n')
            for code, reason in ((7, 'push refused by hygiene guard'), (9, 'push refused by dashboard guard'),
                                 (42, 'push failed (exit 42)')):
                try:
                    raise subprocess.CalledProcessError(code, 'safe_push; inspect push.log')
                except subprocess.CalledProcessError as error:
                    got, detail = push_failure(error, cycle)
                self.assertEqual(got, reason)
                self.assertIn('unclassified benchmark uploads', detail)
            try:
                raise subprocess.CalledProcessError(255, ['ssh', '-o', 'BatchMode=yes', 'host', 'true'])
            except subprocess.CalledProcessError as error:
                self.assertEqual(push_failure(error, Path(d) / 'none')[0], 'host capture failed')

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

    def test_coverage_counts_change_the_evidence_but_capture_windows_do_not(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'mac-observations.csv').write_text('metric_name,median_ns\nx,10\n')
            cov = p / 'coverage.json'
            cov.write_text(json.dumps(dict(mac=dict(capture_window_utc=['a'], campaigns=[]))))
            base = fingerprint(p)
            cov.write_text(json.dumps(dict(mac=dict(capture_window_utc=['b'], campaigns=[]))))
            self.assertEqual(base, fingerprint(p))
            cov.write_text(json.dumps(dict(mac=dict(capture_window_utc=['b'], cells=dict(planned=2, ok=2)))))
            withcells = fingerprint(p)
            self.assertNotEqual(base, withcells)
            cov.write_text(json.dumps(dict(mac=dict(capture_window_utc=['c'], cells=dict(planned=2, ok=1, pending=1)))))
            self.assertNotEqual(withcells, fingerprint(p))

    def test_foreign_owner_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / '.workongoing').write_text('another lane\n')
            result = publish(p, p, 'digest', p)
            self.assertIn('another lane', result['state'])
            self.assertTrue(result['blocked'])
            self.assertEqual((p / '.workongoing').read_text(), 'another lane\n')

    def test_dirty_checkout_is_never_modified_or_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            with patch('sync_saved_results.run', return_value='M user-source.rs') as run:
                result = publish(p, p, 'digest', p)
            self.assertIn('uncommitted', result['state'])
            self.assertEqual((result['blocked'], result['detail']), (True, 'M user-source.rs'))
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
