"""Provenance must refuse sources changed after their inventory was captured."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from canonical_provenance import checked_document


class SourceBindings(unittest.TestCase):
    def test_source_inventory_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            source = root/'reference.m'
            source.write_text('original reference source\n')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            inventory = root/'bindings.json'
            inventory.write_text(json.dumps({str(source): digest}))
            self.assertEqual(checked_document(inventory)['entries'][str(source)], digest)
            source.write_text('changed reference source\n')
            with self.assertRaisesRegex(ValueError, 'reference source changed'):
                checked_document(inventory)

    def test_method_inventory_keeps_input_and_internal_dtype(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            source = root/'vsi.py'
            source.write_text('x = x.double()\n')
            entry = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         dtype='float32 input; explicit internal float64')
            inventory = root/'methods.json'
            inventory.write_text(json.dumps({'vsi': entry}))
            self.assertEqual(checked_document(inventory, methods=True)['entries']['vsi'], entry)

    def test_shared_source_cannot_hide_contradictory_method_hash(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            source = root/'fsim.py'
            source.write_text('shared implementation\n')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            inventory = root/'methods.json'
            inventory.write_text(json.dumps({
                'fsim': dict(source=str(source), source_sha256='0'*64),
                'fsimc': dict(source=str(source), source_sha256=digest)}))
            with self.assertRaisesRegex(ValueError, 'reference source changed'):
                checked_document(inventory, methods=True)


if __name__ == '__main__':
    unittest.main()
