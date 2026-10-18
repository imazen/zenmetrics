"""Bundle controls: refuse changed inputs, preserve relocation and phase identities."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import mac_service
import prepare_durable


class BundleTests(unittest.TestCase):
    def test_hash_verification_survives_relocation_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            original = root / 'original'
            original.mkdir()
            (original / 'worker').write_bytes(b'worker fixture')
            (original / 'SHA256SUMS.json').write_text(json.dumps({'worker': prepare_durable.sha(original / 'worker')}))
            moved = root / 'moved'
            original.rename(moved)
            mac_service.verify(moved)
            (moved / 'worker').write_bytes(b'changed worker')
            with self.assertRaisesRegex(ValueError, 'bundle changed'):
                mac_service.verify(moved)

    def test_verify_refuses_path_escape(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            (root / 'SHA256SUMS.json').write_text(json.dumps({'../outside': '0' * 64}))
            with self.assertRaisesRegex(ValueError, 'escapes bundle'):
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
                    '--inputs-meta', str(root / 'meta.json'), '--output', str(root / 'out'), '--source-commit', 'test', '--sizes', 'fixture']
            with patch.object(sys, 'argv', args), patch.object(subprocess, 'check_output', return_value=b'[{"metric_name":"psnr","compiled":true,"gpu":false}]'):
                with self.assertRaisesRegex(ValueError, 'input bytes'):
                    prepare_durable.main()
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
