"""Bundle identity and service recovery controls; launchctl calls are mocked."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import mac_service
import prepare_durable
import select_durable
import durable
import linux_service
import fcntl


def write_hashes(bundle):
    hashes = {str(path.relative_to(bundle)): prepare_durable.sha(path)
              for path in bundle.rglob('*') if path.is_file() and path.name != 'SHA256SUMS.json'}
    (bundle / 'SHA256SUMS.json').write_text(json.dumps(hashes))
    return hashes


def make_bundle(bundle):
    (bundle / 'bin').mkdir(parents=True)
    (bundle / 'inputs').mkdir()
    for name in ('durable.py', 'mac_service.py', 'bin/iqa-compete', 'build.json', 'inputs.meta.json', 'select_durable.py', 'admission_recovery.py'):
        (bundle / name).write_bytes(b'fixture ' + name.encode())
    (bundle / 'inputs/reference.rgb').write_bytes(b'xyz')
    (bundle / 'inputs/distorted.rgb').write_bytes(b'abc')
    (bundle / 'manifest.json').write_text(json.dumps(dict(
        schema=1, binary='bin/iqa-compete', binary_sha256=prepare_durable.sha(bundle / 'bin/iqa-compete'),
        input_sha256={str(path.relative_to(bundle)): prepare_durable.sha(path)
                      for path in (bundle / 'inputs').iterdir()}, cells=[])))
    write_hashes(bundle)
    return bundle


class BundleTests(unittest.TestCase):
    def test_linux_sequence_refuses_changed_plan_and_incomplete_predecessor(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            bundle = make_bundle(root / 'bundle')
            phases = [dict(name=str(i), bundle=str(bundle), output=str(root / str(i)),
                           manifest_sha256=prepare_durable.sha(bundle / 'manifest.json')) for i in range(2)]
            plan = root / 'sequence.json'
            plan.write_text(json.dumps(dict(phases=phases, lock=str(root / 'heavy.lock'))))
            with patch.object(linux_service, 'supervise') as run:
                with self.assertRaisesRegex(ValueError, 'sequence changed'):
                    linux_service.execute(plan, 'wrong')
                run.assert_not_called()
            with patch.object(linux_service, 'supervise', return_value=0) as run, \
                 patch.object(linux_service, 'predecessor_complete', return_value=False):
                with self.assertRaisesRegex(ValueError, 'predecessor incomplete'):
                    linux_service.execute(plan, prepare_durable.sha(plan))
                self.assertEqual(run.call_count, 1)

    def test_linux_sequence_skips_only_terminal_predecessors(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            bundle = make_bundle(root / 'bundle')
            phases = [dict(name=str(i), bundle=str(bundle), output=str(root / str(i)),
                           manifest_sha256=prepare_durable.sha(bundle / 'manifest.json')) for i in range(2)]
            plan = root / 'sequence.json'
            plan.write_text(json.dumps(dict(phases=phases, lock=str(root / 'heavy.lock'))))
            with patch.object(linux_service, 'supervise') as run, \
                 patch.object(linux_service, 'predecessor_complete', return_value=True):
                self.assertEqual(linux_service.execute(plan, prepare_durable.sha(plan)), 0)
                run.assert_not_called()

    def test_progress_distinguishes_pending_failed_attempts_from_exhausted_cells(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            output = Path(tmp)
            durable.atomic(output / 'progress.json', dict(complete=False, statuses=dict(ok=1, failed=0, pending=1)))
            attempt = output / 'cells/a/attempt-0001'
            attempt.mkdir(parents=True)
            durable.save_result(attempt / 'result.json', dict(status='failed', row=None))
            second = output / 'cells/b/attempt-0001'
            second.mkdir(parents=True)
            durable.save_result(second / 'result.json', dict(status='ok', row=dict(timing_phase='warm')))
            with patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, ' state = running\n', '')):
                row = mac_service.progress(output, 'user/fixture')
            self.assertEqual(row['attempt_statuses'], dict(failed=1, ok=1))
            self.assertEqual(row['successful_phases'], dict(warm=1))
            self.assertEqual(row['progress']['statuses']['failed'], 0)

    def test_check_in_saves_remote_failure_without_an_available_agent(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            plan = root / 'plan.json'
            plan.write_text(json.dumps(dict(machine='mac', script='benchmark/mac_service.py',
                phases=[dict(bundle='bundle', output='output')])) )
            with patch.object(subprocess, 'run', side_effect=subprocess.TimeoutExpired('ssh', 60)), \
                 patch.object(mac_service.shutil, 'which', return_value=None), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(mac_service.check_in(plan, root / 'checks'), 124)
            receipt = json.loads(next((root / 'checks').glob('*.json')).read_text())
            self.assertEqual(receipt['returncode'], 124)
            self.assertIn('timed out', receipt['stderr'])

    def test_predecessor_requires_exact_terminal_inventory_and_controller_exit(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            output = Path(tmp)
            gate = dict(output=tmp, manifest_sha256='frozen-manifest')
            self.assertFalse(mac_service.predecessor_complete(gate))
            cell = dict(metric_name='fixture')
            saved = dict(frozen=dict(manifest_sha256='frozen-manifest', manifest=dict(cells=[cell])), policy={})
            durable.atomic(output / 'identity.json', saved)
            row = dict(cell_id='000000-' + durable.digest(durable.encoded(cell))[:16], status='failed')
            report = dict(identity=durable.digest(durable.encoded(saved)), complete=False, cells=[row])
            durable.atomic(output / 'report.json', report)
            (output / 'coordinator.lock').touch()
            self.assertFalse(mac_service.predecessor_complete(gate))
            report['complete'] = True
            durable.atomic(output / 'report.json', report)
            with (output / 'coordinator.lock').open('rb') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self.assertFalse(mac_service.predecessor_complete(gate))
            self.assertTrue(mac_service.predecessor_complete(gate))
            report['cells'] = []
            durable.atomic(output / 'report.json', report)
            with self.assertRaisesRegex(ValueError, 'inventory'):
                mac_service.predecessor_complete(gate)
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                mac_service.predecessor_complete(dict(gate, manifest_sha256='other'))

    def test_subset_preserves_all_phases_and_refuses_unknown_selection(self):
        cells = [dict(metric_name=m, size_name='small', threads=t, timing_phase=p)
                 for m in ('psnr', 'gmsd') for t in (1, 8) for p in ('cold', 'warm')]
        selected = select_durable.select(cells, ['gmsd'], threads=[1])
        self.assertEqual([c['timing_phase'] for c in selected], ['cold', 'warm'])
        self.assertEqual(len(cells), 8)
        with self.assertRaisesRegex(ValueError, 'unknown metric_name'):
            select_durable.select(cells, ['missing'])

    def test_hash_verification_survives_relocation_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            original = make_bundle(root / 'original')
            moved = root / 'moved'
            original.rename(moved)
            mac_service.verify(moved)
            (moved / 'bin/iqa-compete').write_bytes(b'changed worker')
            with self.assertRaisesRegex(ValueError, 'bundle changed'):
                mac_service.verify(moved)

    def test_verify_refuses_path_escape(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = make_bundle(Path(tmp) / 'bundle')
            hashes = write_hashes(root)
            hashes['../outside'] = '0' * 64
            (root / 'SHA256SUMS.json').write_text(json.dumps(hashes))
            with self.assertRaisesRegex(ValueError, 'escapes bundle'):
                mac_service.verify(root)

    def test_verify_refuses_empty_and_incomplete_checksum_inventory(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = make_bundle(Path(tmp) / 'bundle')
            complete = write_hashes(root)
            for missing in (None, 'manifest.json', 'durable.py', 'mac_service.py',
                            'bin/iqa-compete', 'build.json', 'inputs.meta.json', 'select_durable.py'):
                with self.subTest(missing=missing):
                    hashes = {} if missing is None else {k: v for k, v in complete.items() if k != missing}
                    (root / 'SHA256SUMS.json').write_text(json.dumps(hashes))
                    with self.assertRaisesRegex(ValueError, 'required files'):
                        mac_service.verify(root)

    def test_verify_requires_each_manifest_input_in_checksum_inventory(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = make_bundle(Path(tmp) / 'bundle')
            hashes = write_hashes(root)
            del hashes['inputs/reference.rgb']
            (root / 'SHA256SUMS.json').write_text(json.dumps(hashes))
            with self.assertRaisesRegex(ValueError, 'input hashes missing'):
                mac_service.verify(root)

    def test_prepare_refuses_input_tampering_before_creating_bundle(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            (root / 'reference.rgb').write_bytes(b'xyz')
            (root / 'distorted.rgb').write_bytes(b'abc')
            pair = dict(size_name='fixture', width=1, height=1, method='test-only', reference='reference.rgb', distorted='distorted.rgb')
            (root / 'pairs.json').write_text(json.dumps([pair]))
            known = dict(pair, sha256=dict(reference=hashlib.sha256(b'bad').hexdigest(), distorted=hashlib.sha256(b'abc').hexdigest()))
            (root / 'meta.json').write_text(json.dumps(dict(derivatives=[known])))
            args = ['prepare_durable.py', '--binary', str(root / 'worker'), '--pairs', str(root / 'pairs.json'),
                    '--inputs-meta', str(root / 'meta.json'), '--output', str(root / 'out'), '--source-commit', 'test', '--build-receipt', str(root / 'build.json'), '--sizes', 'fixture']
            with patch.object(sys, 'argv', args), patch.object(subprocess, 'check_output', return_value=b'[{"metric_name":"psnr","compiled":true,"gpu":false}]'):
                with self.assertRaisesRegex(ValueError, 'input bytes'):
                    prepare_durable.main()
            self.assertFalse((root / 'out').exists())

    def test_prepared_bundle_has_worker_compatible_budgets_and_frozen_identity(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            source = make_bundle(root / 'source')
            recipe = (prepare_durable.HERE / 'bundle.just').read_bytes()
            (source / 'bundle.just').write_bytes(recipe)
            pair = dict(size_name='fixture', width=1, height=1, method='test-only',
                        reference='inputs/reference.rgb', distorted='inputs/distorted.rgb')
            (source / 'pairs.json').write_text(json.dumps([pair]))
            known = dict(pair, sha256={role: prepare_durable.sha(source / pair[role])
                                     for role in ('reference', 'distorted')})
            (source / 'inputs.meta.json').write_text(json.dumps(dict(derivatives=[known])))
            (source / 'build.json').write_text(json.dumps(dict(source_commit='fixture-commit',
                binary_sha256=prepare_durable.sha(source / 'bin/iqa-compete'),
                features=['compete', 'metrics-parallel'])))
            args = ['prepare_durable.py', '--binary', str(source / 'bin/iqa-compete'),
                    '--pairs', str(source / 'pairs.json'), '--inputs-meta', str(source / 'inputs.meta.json'),
                    '--output', str(root / 'out'), '--source-commit', 'fixture-commit',
                    '--build-receipt', str(source / 'build.json'), '--sizes', 'fixture',
                    '--threads', '1,8', '--cold-repeats', '2']
            inventory = [dict(metric_name='psnr', compiled=True, gpu=False),
                         dict(metric_name='gmsd', compiled=True, gpu=False),
                         dict(metric_name='dists', compiled=True, gpu=False),
                         dict(metric_name='ahiq', compiled=True, gpu=False),
                         dict(metric_name='dmm', compiled=True, gpu=False),
                         dict(metric_name='ssim2-gpu', compiled=True, gpu=True)]
            with patch.object(prepare_durable, 'HERE', source), patch.object(sys, 'argv', args), \
                    patch.object(subprocess, 'check_output', return_value=json.dumps(inventory).encode()), \
                    patch.object(prepare_durable.platform, 'platform', return_value='fixture OS'), \
                    contextlib.redirect_stdout(io.StringIO()):
                prepare_durable.main()
            bundle = root / 'out'
            mac_service.verify(bundle)
            self.assertEqual((bundle / 'justfile').read_bytes(), recipe)
            self.assertIn('justfile', json.loads((bundle / 'SHA256SUMS.json').read_text()))
            manifest = json.loads((bundle / 'manifest.json').read_text())
            self.assertEqual(manifest['metadata']['metrics'], ['psnr', 'gmsd'])
            self.assertEqual(manifest['binary_sha256'], prepare_durable.sha(bundle / 'bin/iqa-compete'))
            self.assertEqual(manifest['metadata']['build_receipt_sha256'], prepare_durable.sha(bundle / 'build.json'))
            self.assertEqual(len(manifest['cells']), 24)
            for cell in manifest['cells']:
                # The worker rejects a budget above 850000 ms before zenbench starts.
                self.assertGreaterEqual(cell['chunk_budget_ms'], 1)
                self.assertLessEqual(cell['chunk_budget_ms'], 850_000)
                self.assertEqual(cell['required_samples'], 1 if cell['timing_phase'] == 'cold' else 10)
                for role in ('reference', 'distorted'):
                    self.assertFalse(Path(cell[role]).is_absolute())
                    self.assertEqual(manifest['input_sha256'][cell[role]], prepare_durable.sha(bundle / cell[role]))


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.bundle = make_bundle(self.root / 'bundle')
        self.output = self.root / 'output'
        self.service_dir = self.output.with_name(self.output.name + '.service')
        self.home = self.root / 'home'
        self.command = [sys.executable, str(self.bundle / 'durable.py'), 'run',
                        str(self.bundle / 'manifest.json'), str(self.output),
                        '--run-heavy', str(self.root / 'run-heavy'), '--lock', str(self.root / 'heavy.lock')]

    def execute(self, command=None, expected_sha=None, returncode=0):
        raw = (json.dumps(command if command is not None else self.command) + '\n').encode()
        self.service_dir.mkdir(exist_ok=True)
        (self.service_dir / 'service-command.json').write_bytes(raw)
        digest = expected_sha if expected_sha is not None else hashlib.sha256(raw).hexdigest()
        args = ['mac_service.py', 'execute', str(self.bundle), str(self.output), '--command-sha256', digest]
        stderr = io.StringIO()
        awake = Mock()
        with patch.object(sys, 'argv', args), patch.object(subprocess, 'Popen', return_value=awake) as spawn, \
                patch.object(mac_service, 'supervise', return_value=returncode) as child, contextlib.redirect_stderr(stderr):
            result = mac_service.main()
        return result, child, spawn, awake, stderr.getvalue()

    def test_stalled_controller_is_killed_without_manual_resume(self):
        with contextlib.redirect_stderr(io.StringIO()):
            rc = mac_service.supervise([sys.executable, '-c', 'import time; time.sleep(30)'], self.output, startup=.3)
        self.assertLess(rc, 0)

    def test_healthy_controller_completion_is_preserved(self):
        rc = mac_service.supervise([sys.executable, '-c', 'pass'], self.output, startup=2)
        self.assertEqual(rc, 0)

    def test_start_refuses_installed_other_bundle(self):
        directory = self.home / 'Library/LaunchAgents'
        directory.mkdir(parents=True)
        token = hashlib.sha256(str(self.output.resolve()).encode()).hexdigest()[:16]
        plist = directory / ('io.imazen.iqa-durable.' + token + '.plist')
        with plist.open('wb') as stream:
            plistlib.dump(dict(ProgramArguments=[sys.executable, '/other/mac_service.py', 'execute', '/other', str(self.output)]), stream)
        args = ['mac_service.py', 'start', str(self.bundle), str(self.output)]
        with patch.object(Path, 'home', return_value=self.home), patch.object(sys, 'argv', args), patch.object(subprocess, 'call') as launch:
            with self.assertRaisesRegex(ValueError, 'another bundle/output'):
                mac_service.main()
        launch.assert_not_called()

    def test_execute_runs_frozen_command_and_preserves_crash_restart_status(self):
        for rc, expected in ((0, 0), (2, 0), (137, 137), (-9, -9)):
            with self.subTest(child_returncode=rc):
                result, child, spawn, awake, _ = self.execute(returncode=rc)
                self.assertEqual(result, expected)
                child.assert_called_once_with(self.command, self.output)
                spawn.assert_called_once_with(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
                awake.terminate.assert_called_once_with()
                awake.wait.assert_called_once_with()

    def test_execute_refuses_mutated_command_with_original_frozen_hash(self):
        original = (json.dumps(self.command) + '\n').encode()
        changed = self.command + ['--jobs', '123']
        result, child, spawn, _, stderr = self.execute(changed, hashlib.sha256(original).hexdigest())
        self.assertEqual(result, 0)
        self.assertIn('service command changed', stderr)
        child.assert_not_called()
        spawn.assert_not_called()

    def test_execute_refuses_other_bundle_manifest_or_output_even_with_matching_hash(self):
        for index, replacement in ((0, '/different/python'), (1, str(self.root / 'other/durable.py')),
                                   (3, str(self.root / 'other/manifest.json')), (4, str(self.root / 'other-output'))):
            with self.subTest(argument=index):
                changed = self.command.copy()
                changed[index] = replacement
                result, child, spawn, _, stderr = self.execute(changed)
                self.assertEqual(result, 0)
                self.assertIn('another experiment', stderr)
                child.assert_not_called()
                spawn.assert_not_called()

    def install_args(self):
        return ['mac_service.py', 'install', str(self.bundle), str(self.output),
                '--run-heavy', str(self.root / 'run-heavy'), '--lock', str(self.root / 'heavy.lock')]

    def test_install_freezes_command_hash_in_plist_and_start_reuses_it(self):
        with patch.object(Path, 'home', return_value=self.home), patch.object(sys, 'platform', 'darwin'), \
                patch.object(sys, 'argv', self.install_args()), patch.object(subprocess, 'run') as launch, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(mac_service.main(), 0)
        plists = list((self.home / 'Library/LaunchAgents').glob('*.plist'))
        self.assertEqual(len(plists), 1)
        plist = plists[0]
        payload = plistlib.loads(plist.read_bytes())
        self.assertFalse(self.output.exists(), 'install must leave controller output uncreated')
        self.assertEqual(payload['StandardOutPath'], str(self.service_dir / 'service.stdout.log'))
        self.assertEqual(payload['StandardErrorPath'], str(self.service_dir / 'service.stderr.log'))
        raw = (self.service_dir / 'service-command.json').read_bytes()
        self.assertEqual(payload['ProgramArguments'][-2:], ['--command-sha256', hashlib.sha256(raw).hexdigest()])
        self.assertEqual(payload['KeepAlive'], dict(SuccessfulExit=False))
        domain = 'user/' + str(os.getuid())
        launch.assert_called_once_with(['launchctl', 'bootstrap', domain, str(plist)], check=True)
        before = plist.read_bytes(), raw
        args = ['mac_service.py', 'start', str(self.bundle), str(self.output)]
        with patch.object(Path, 'home', return_value=self.home), patch.object(sys, 'argv', args), \
                patch.object(subprocess, 'call', return_value=0) as start:
            self.assertEqual(mac_service.main(), 0)
        start.assert_called_once_with(['launchctl', 'bootstrap', domain, str(plist)])
        self.assertEqual((plist.read_bytes(), (self.service_dir / 'service-command.json').read_bytes()), before)

    def test_failed_bootstrap_removes_new_plist_and_install_can_retry(self):
        with patch.object(Path, 'home', return_value=self.home), patch.object(sys, 'platform', 'darwin'), \
                patch.object(sys, 'argv', self.install_args()), patch.object(subprocess, 'run', \
                    side_effect=subprocess.CalledProcessError(5, ['launchctl', 'bootstrap'])) as launch:
            with self.assertRaises(subprocess.CalledProcessError):
                mac_service.main()
        self.assertEqual(list((self.home / 'Library/LaunchAgents').glob('*.plist')), [])
        self.assertTrue((self.service_dir / 'service-command.json').exists())
        with patch.object(Path, 'home', return_value=self.home), patch.object(sys, 'platform', 'darwin'), \
                patch.object(sys, 'argv', self.install_args()), patch.object(subprocess, 'run') as retry, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(mac_service.main(), 0)
        retry.assert_called_once()
        launch.assert_called_once()

    def test_install_refusal_preserves_existing_plist_and_command(self):
        directory = self.home / 'Library/LaunchAgents'
        directory.mkdir(parents=True)
        token = hashlib.sha256(str(self.output.resolve()).encode()).hexdigest()[:16]
        plist = directory / ('io.imazen.iqa-durable.' + token + '.plist')
        plist.write_bytes(b'existing service fixture')
        self.service_dir.mkdir()
        command = self.service_dir / 'service-command.json'
        command.write_bytes(b'existing command fixture')
        with patch.object(Path, 'home', return_value=self.home), patch.object(sys, 'platform', 'darwin'), \
                patch.object(sys, 'argv', self.install_args()), patch.object(subprocess, 'run') as launch, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                mac_service.main()
        self.assertEqual(error.exception.code, 2)
        launch.assert_not_called()
        self.assertEqual(plist.read_bytes(), b'existing service fixture')
        self.assertEqual(command.read_bytes(), b'existing command fixture')


if __name__ == '__main__':
    unittest.main()
