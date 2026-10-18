"""Real worker controls. Caller supplies IQA_TIMING_BINARY; no implicit skips."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


class TimingWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = str(Path(os.environ['IQA_TIMING_BINARY']).resolve(strict=True))
        cls.scratch = tempfile.TemporaryDirectory(prefix='iqa-timing-', dir=os.environ['TMPDIR'])
        cls.root = Path(cls.scratch.name)
        cls.reference = cls.root / 'reference.rgb'
        cls.distorted = cls.root / 'distorted.rgb'
        cls.reference.write_bytes(bytes((x * 17 + y * 31 + c * 7) % 240 for y in range(32) for x in range(32) for c in range(3)))
        cls.distorted.write_bytes(bytes(v + (i % 5 == 0) * 9 for i, v in enumerate(cls.reference.read_bytes())))

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def worker(self, metric, mode, phase, **extra):
        config = dict(metric_name=metric, width=32, height=32,
                      reference=str(self.reference), distorted=str(self.distorted),
                      only_mode=mode, locked_process_accounting=True,
                      device='cpu', required_samples=10, sample_target_ns=100_000,
                      chunk_budget_ms=30_000, coverage_first=True)
        config.update(extra)
        if phase is not None:
            config['timing_phase'] = phase
        path = self.root / 'config.json'
        path.write_text(json.dumps(config))
        result = subprocess.run([self.binary, str(path)], text=True, capture_output=True, timeout=90)
        print(result.stdout, end='')
        print(result.stderr, end='')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(s.removeprefix('IQA_ROW ')) for s in result.stdout.splitlines() if s.startswith('IQA_ROW ')]
        self.assertEqual(len(rows), 1, result.stdout + result.stderr)
        row = rows[0]
        for key in ('metric_name', 'reference', 'distorted', 'width', 'height'):
            self.assertEqual(row[key], config[key])
        self.assertEqual(row['mode'], mode)
        self.assertEqual(row['timing_phase'], phase or 'warm')
        self.assertGreater(row['input_load_ns'], 0)
        if phase == 'cold':
            self.assertNotIn('IQA_WARMUP ', result.stdout)
        if row['status'] == 'ok':
            self.assertEqual(row['score_bits'], struct.pack('>d', row['score']).hex())
            self.assertGreater(row['median_ns'], 0)
            if phase == 'cold':
                self.assertEqual(row['n'], 1)
                self.assertEqual(row['cold_elapsed_ns'], row['median_ns'])
            else:
                self.assertGreaterEqual(row['n'], 10 if phase == 'warm' else config['required_samples'])
        return row

    def test_psnr_cold_warm_identity(self):
        cold = self.worker('psnr', 'score', 'cold')
        warm = self.worker('psnr', 'score', 'warm')
        self.assertEqual(cold['status'], 'ok')
        self.assertEqual(warm['status'], 'ok')
        self.assertEqual(cold['score_bits'], warm['score_bits'])

    def test_gmsd_both_modes_cold_warm_identity(self):
        for mode in ('score', 'score+map'):
            cold = self.worker('gmsd', mode, 'cold')
            warm = self.worker('gmsd', mode, 'warm')
            self.assertEqual(cold['status'], 'ok')
            self.assertEqual(warm['status'], 'ok')
            self.assertEqual(cold['score_bits'], warm['score_bits'])

    def test_map_unsupported_has_no_timing(self):
        for phase in ('cold', 'warm'):
            row = self.worker('psnr', 'score+map', phase, median_ns=123, cold_elapsed_ns=123)
            self.assertEqual(row['status'], 'unsupported')
            self.assertEqual(row['n'], 0)
            self.assertNotIn('median_ns', row)
            self.assertNotIn('cold_elapsed_ns', row)

    def test_returned_refusal_has_no_timing(self):
        row = self.worker('ssim2-gpu', 'score', 'cold', gpu_runtime='refusal-control')
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['n'], 0)
        self.assertIn('error', row)
        self.assertNotIn('median_ns', row)

    def test_nonfinite_has_no_timing(self):
        row = self.worker('psnr', 'score', 'cold', distorted=str(self.reference))
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['n'], 0)
        self.assertIn('error', row)
        self.assertNotIn('median_ns', row)

    def test_explicit_warm_has_ten_rounds(self):
        row = self.worker('psnr', 'score', 'warm', required_samples=1)
        self.assertEqual(row['status'], 'ok')
        self.assertEqual(row['n'], 10)

    def test_legacy_phase_defaults_to_warm(self):
        row = self.worker('psnr', 'score', None, required_samples=1)
        self.assertEqual(row['status'], 'ok')
        self.assertEqual(row['n'], 1)


if __name__ == '__main__':
    unittest.main()
