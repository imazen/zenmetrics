import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sync_saved_results import FILES, fingerprint, public_checks, publish


class SyncControls(unittest.TestCase):
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
