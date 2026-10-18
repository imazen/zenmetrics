"""Lightweight scheduler/row controls and real owned-process termination."""
import csv
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import missing_metrics
import perf_deltas

HERE = Path(__file__).resolve().parent


class Driver(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.home() / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def row(self, metric, value='1', shape='3840x2160', threads=8):
        return '\t'.join([metric, shape, str(threads), '0', '1'] + [value] * 9 + ['0.5']) + '\n'

    def test_headerless_rows_survive_exactly_and_first_error_is_readable(self):
        path = self.root / 'results.tsv'
        original = self.row('afine', 'error') + self.row('dists') * 3
        path.write_text(original)
        perf_deltas.prepare_output(path)
        self.assertEqual(path.with_name(path.name + '.headerless.bak').read_bytes(), original.encode())
        self.assertEqual(path.read_bytes(), ('\t'.join(perf_deltas.COLS) + '\n' + original).encode())
        perf_deltas.prepare_output(path)
        with path.open() as source:
            rows = list(csv.DictReader(source, delimiter='\t'))
        self.assertEqual([r['metric'] for r in rows], ['afine', 'dists', 'dists', 'dists'])
        self.assertEqual(missing_metrics.missing(path, ['afine', 'dists', 'wd'], 8), ['wd'])

    def test_new_and_empty_files_get_headers_before_error_rows(self):
        for exists in (False, True):
            path = self.root / f'{exists}.tsv'
            if exists:
                path.touch()
            perf_deltas.append_error(path, 'afine', 8, 1, 'controlled timeout')
            with path.open() as source:
                rows = list(csv.DictReader(source, delimiter='\t'))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['metric'], 'afine')
            self.assertEqual(rows[0]['score'], 'controlled timeout')

    def test_completion_filters_threads_shape_and_partial_counts(self):
        path = self.root / 'rows.tsv'
        path.write_text(self.row('dists') * 2 + self.row('dists', shape='1024x1024')
                        + self.row('dists', threads=1) + self.row('afine', 'error'))
        self.assertEqual(missing_metrics.missing(path, ['dists', 'afine', 'wd'], 8), ['dists', 'wd'])

    def test_completion_checker_failure_stops_instead_of_skipping_work(self):
        checker = self.root / 'broken-checker.py'
        checker.write_text('raise SystemExit(7)\n')
        env = dict(os.environ, CX4K_OUTPUT=str(self.root / 'out'), CX4K_LOGS=str(self.root / 'logs'),
                   CX4K_MISSING_DRIVER=str(checker), CX4K_METRICS_ENG='wd')
        result = subprocess.run(['bash', str(HERE / 'drive_nn_4k.sh'), 'cuda'],
                                env=env, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(list((self.root / 'logs').iterdir()), [])

    def test_engine_scheduler_runs_missing_and_skips_completed_preserving_rows(self):
        wrapper = self.root / 'resource-wrapper'
        wrapper.write_text('#!/bin/sh\nshift 5\nexec "$@"\n')
        wrapper.chmod(0o755)
        fake = self.root / 'perf.py'
        fake.write_text('''import argparse, os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--out'); p.add_argument('--metrics'); p.add_argument('--repeats', type=int)
a, _=p.parse_known_args()
with open(os.environ['CALLS'], 'a') as calls: calls.write(a.metrics+'\\n')
with open(a.out, 'a') as out:
 for i in range(a.repeats): out.write('\\t'.join([a.metrics,'3840x2160','8',str(i),'1']+['1']*9+['0.5'])+'\\n')
''')
        for phase, engine in [('cuda', 'ort-cuda'), ('webgpu', 'ort-webgpu')]:
            with self.subTest(phase=phase):
                output = self.root / phase
                output.mkdir()
                path = output / f'i265_nn_{engine}_3840x2160.tsv'
                original = self.row('dists') * 3 + self.row('afine', 'error')
                path.write_text(original)
                calls = output / 'calls'
                env = dict(os.environ, CX4K_OUTPUT=str(output), CX4K_LOGS=str(output / 'logs'),
                           CX4K_LOCK=str(output / 'heavy.lock'), CX4K_RUN_HEAVY=str(wrapper),
                           CX4K_PERF_DRIVER=str(fake), CX4K_METRICS_ENG='dists,afine,wd', CALLS=str(calls))
                result = subprocess.run(['bash', str(HERE / 'drive_nn_4k.sh'), phase],
                                        env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertEqual(calls.read_text(), 'wd\n')
                self.assertTrue(path.read_text().startswith(original))
                self.assertEqual(missing_metrics.missing(path, ['dists', 'afine', 'wd'], 8), [])
                result = subprocess.run(['bash', str(HERE / 'drive_nn_4k.sh'), phase],
                                        env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertEqual(calls.read_text(), 'wd\n')

    def test_hold_errors_are_terminal_unqualified_and_explicit_pieapp_is_announced(self):
        wrapper = self.root / 'wrapper'
        wrapper.write_text('#!/bin/sh\nshift 5\nexec "$@"\n')
        wrapper.chmod(0o755)
        fake = self.root / 'fake_perf.py'
        fake.write_text(f'''import argparse, sys
sys.path.insert(0, {str(HERE)!r})
from perf_deltas import append_error
p=argparse.ArgumentParser(); p.add_argument('--out'); p.add_argument('--metrics')
a, _=p.parse_known_args()
append_error(a.out, a.metrics, 8, 0, 'controlled probe failure')
''')
        for phase, filename in [('rten', 'i265_nn_3840x2160.tsv'),
                                ('cuda', 'i265_nn_ort-cuda_3840x2160.tsv')]:
            output = self.root / phase
            env = dict(os.environ, CX4K_OUTPUT=str(output), CX4K_LOGS=str(output / 'logs'),
                       CX4K_LOCK=str(output / 'heavy.lock'), CX4K_RUN_HEAVY=str(wrapper),
                       CX4K_PERF_DRIVER=str(fake), CX4K_METRICS_ENG='pieapp', CX4K_METRICS_RTEN='pieapp')
            result = subprocess.run(['bash', str(HERE / 'drive_nn_4k.sh'), phase],
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('BENCHMARK EXPLICITLY SELECTED (slow): pieapp\n', result.stderr)
            self.assertNotIn('required): pieapp\n', result.stderr)
            self.assertEqual(len(list((output / 'logs').glob('*.log'))), 1)
            with (output / filename).open() as source:
                rows = list(csv.DictReader(source, delimiter='\t'))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['wall_ms_per_call'], 'error')
            self.assertEqual(rows[0]['score'], 'controlled probe failure')

    def test_outer_term_and_probe_timeout_kill_owned_child_only(self):
        # Fake perf runs a real sleeping child in the session owned by run().
        # Another identically named process must survive both controls.
        perf = self.root / 'perf'
        perf.write_text(f'''#!{sys.executable}
import os, subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
with open(os.environ['CHILD_PID'], 'w') as f: f.write(str(child.pid))
child.wait()
''')
        perf.chmod(0o755)
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        try:
            for outer_term in (False, True):
                pidfile = self.root / f'child-{outer_term}'
                env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ['PATH'],
                           CHILD_PID=str(pidfile))
                args = [sys.executable, str(HERE / 'perf_deltas.py'), '--bin', 'unused',
                        '--out', str(self.root / f'{outer_term}.tsv'), '--metrics', 'wd',
                        '--extra', '1', '--repeats', '1', '--timeout-s', '1' if not outer_term else '30']
                process = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    deadline = time.monotonic() + 5
                    while not pidfile.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(pidfile.exists())
                    child = int(pidfile.read_text())
                    if outer_term:
                        process.send_signal(signal.SIGTERM)
                    process.communicate(timeout=5)
                    self.assertEqual(process.returncode, 143 if outer_term else 0)
                    state = Path(f'/proc/{child}/stat')
                    deadline = time.monotonic() + 2
                    while state.exists() and state.read_text().split()[2] != 'Z' and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(not state.exists() or state.read_text().split()[2] == 'Z')
                    self.assertIsNone(unrelated.poll())
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=5)
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_interrupted_holds_without_rows_retry_once_and_keep_both_logs(self):
        wrapper = self.root / 'wrapper'
        wrapper.write_text('#!/bin/sh\nshift 5\nexec "$@"\n')
        wrapper.chmod(0o755)
        fake = self.root / 'interrupted.py'
        fake.write_text(f'''import sys
sys.path.insert(0, {str(HERE)!r})
from perf_deltas import append_error
if __name__ == '__main__': raise SystemExit(124)
''')
        output = self.root / 'out'
        env = dict(os.environ, CX4K_OUTPUT=str(output), CX4K_LOGS=str(output / 'logs'),
                   CX4K_LOCK=str(output / 'heavy.lock'), CX4K_RUN_HEAVY=str(wrapper),
                   CX4K_PERF_DRIVER=str(fake), CX4K_METRICS_RTEN='wd')
        result = subprocess.run(['bash', str(HERE / 'drive_nn_4k.sh'), 'rten'],
                                env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        logs = list((output / 'logs').glob('*.log'))
        self.assertEqual(len(logs), 2)
        self.assertTrue(all('rc=124' in log.read_text() for log in logs))
        with (output / 'i265_nn_3840x2160.tsv').open() as source:
            rows = list(csv.DictReader(source, delimiter='\t'))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['wall_ms_per_call'], 'error')
        self.assertIn('last exit=124', rows[0]['score'])


if __name__ == '__main__':
    unittest.main()
