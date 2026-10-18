import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from record_durable_build import verify_source_file

class ChainTests(unittest.TestCase):
    def test_chain_binds_both_overlays_and_rejects_reorder_and_tampering(self):
        h=lambda b:hashlib.sha256(b).hexdigest()
        first={'files':{'Cargo.lock':{'original_sha256':h(b'base'),'adapted_sha256':h(b'private')}}}
        second={'files':{'Cargo.lock':{'original_sha256':h(b'private'),'adapted_sha256':h(b'corrected')}}}
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            path=Path(directory)/'Cargo.lock'
            path.write_bytes(b'corrected')
            verify_source_file('Cargo.lock',b'base',path,(first,second))
            with self.assertRaisesRegex(ValueError,'provenance chain'):
                verify_source_file('Cargo.lock',b'base',path,(second,first))
            with self.assertRaisesRegex(ValueError,'differs'):
                verify_source_file('Cargo.lock',b'base',path,(first,))
            path.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'differs'):
                verify_source_file('Cargo.lock',b'base',path,(first,second))
