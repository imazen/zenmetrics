import json
from pathlib import Path
import tempfile
import unittest
from collect_portable_results import collect


class CollectionContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.home() / 'tmp')
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.row = dict(metric_name='psnr', size_name='cid22', width=512, height=512,
            threads=1, impl_binary_sha='worker-hash', pair_sha256='pair-hash',
            mode='score', status='ok', required_samples=5, n=5,
            zenbench_samples=[dict(gate_clean=True) for _ in range(5)])
        self.config = {k: self.row[k] for k in ('metric_name', 'size_name', 'width', 'height', 'threads', 'impl_binary_sha', 'pair_sha256')}
        self.config['only_mode'] = 'score'

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        (self.source / 'psnr-cid22-1-0.json').write_text(json.dumps(self.config))
        (self.source / 'rows.jsonl').write_text(json.dumps(self.row) + '\n')
        (self.source / 'psnr-cid22-1-0.log').write_text('IQA_ROW ' + json.dumps(self.row) + '\nrun-heavy: done rc=0 1s | peak-RSS 0.04GiB | min-avail 20000MiB\n')

    def test_bound_resource_receipt_and_gate_flags(self):
        self.row['zenbench_samples'][0]['gate_clean'] = False
        self.write()
        collect([self.source], self.root / 'out')
        saved = json.loads((self.root / 'out/observations.jsonl').read_text())
        self.assertFalse(saved['qualified_timing'])
        self.assertEqual(saved['evidence']['peak_rss_gib_rounded'], 0.04)
        self.assertFalse(json.loads((self.root / 'out/snapshot.json').read_text())['sources'][0]['complete'])

    def test_tampered_identity_refuses(self):
        self.config['impl_binary_sha'] = 'different-worker'
        self.write()
        with self.assertRaisesRegex(ValueError, 'config/row mismatch'):
            collect([self.source], self.root / 'out')

    def test_unbound_log_refuses(self):
        self.write()
        (self.source / 'psnr-cid22-1-0.log').write_text('run-heavy: done rc=0 1s | peak-RSS 0.04GiB\n')
        with self.assertRaisesRegex(ValueError, 'bound worker log'):
            collect([self.source], self.root / 'out')

    def test_existing_export_refuses_overwrite(self):
        self.write()
        collect([self.source], self.root / 'out')
        before = (self.root / 'out/observations.jsonl').read_bytes()
        with self.assertRaises(FileExistsError):
            collect([self.source], self.root / 'out')
        self.assertEqual(before, (self.root / 'out/observations.jsonl').read_bytes())
