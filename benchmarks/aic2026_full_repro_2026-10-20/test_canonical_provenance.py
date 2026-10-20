"""Provenance must refuse sources changed after their inventory was captured."""
import hashlib
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from canonical_provenance import checked_document, checked_runtime_qualifications, main


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

    def test_runtime_qualification_retains_scope_and_rejects_changed_packet(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            packet = root/'observations.jsonl'
            packet.write_text('{"runtime":"Octave","finite":false}\n')
            entry = dict(qualification='comparison-semantics diagnostic only',
                         scope='two retained controls; no MATLAB execution',
                         native_matlab_execution=False,
                         artifacts={str(packet): hashlib.sha256(packet.read_bytes()).hexdigest()})
            inventory = root/'runtime.json'
            inventory.write_text(json.dumps({'mad': entry}))
            self.assertEqual(checked_runtime_qualifications(inventory)['entries']['mad'], entry)
            packet.write_text('{"runtime":"changed"}\n')
            with self.assertRaisesRegex(ValueError, 'runtime qualification artifact changed'):
                checked_runtime_qualifications(inventory)

    def test_unbound_runtime_note_is_refused(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            inventory = Path(directory)/'runtime.json'
            inventory.write_text(json.dumps({'mad': {'qualification': 'unbound note'}}))
            with self.assertRaisesRegex(ValueError, 'requires a note and artifacts'):
                checked_runtime_qualifications(inventory)

    def test_export_preserves_runtime_limits_and_distinguishes_octave_from_wrapper(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            config = root/'config.json'
            config.write_text('{}')
            versions = root/'versions.json'
            octave = dict(runtime='GNU Octave', version='test version')
            python = dict(runtime='Python wrapper', version='test wrapper version')
            versions.write_text(json.dumps({'python3': python, 'octave-cli': octave}))
            packet = root/'diagnostic.json'
            packet.write_text('{"test_observation":true}\n')
            note = dict(qualification='diagnostic only; no MATLAB execution',
                        artifacts={str(packet): hashlib.sha256(packet.read_bytes()).hexdigest()})
            qualifications = root/'qualification.json'
            qualifications.write_text(json.dumps({'mad': note}))
            schedule = dict(rows_per_worker=32)
            predecessor = dict(manifest_sha256='test predecessor')
            scoring_manifest = root/'frozen-scoring.json'
            scoring_manifest.write_text('{"invented_frozen_fixture":true}\n')
            analysis_predecessor = dict(path=str(scoring_manifest),
                sha256=hashlib.sha256(scoring_manifest.read_bytes()).hexdigest(),
                reason='reporting view; frozen scoring bytes unchanged')
            manifest = dict(config=str(config), scratch=str(root), output=str(root/'work'),
                            cohort_rows=2, source_commit='test source', binary='test binary',
                            binary_sha256='test digest', batch_precision='test precision',
                            driver_source='test driver', bindings={}, execution=schedule,
                            schedule_predecessor=predecessor, addendum={},
                            analysis_predecessor=analysis_predecessor,
                            metrics={'mad': dict(kind='octave', python='python3', gate=None)})
            path = root/'manifest.json'
            path.write_text(json.dumps(manifest))
            argv = ['canonical_provenance.py', str(path), str(versions),
                    '--runtime-qualifications', str(qualifications)]
            with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            report = json.loads((root/'provenance.canonical.json').read_text())
            entry = report['metrics']['mad']
            self.assertEqual(entry['installed_runtime_metadata'], octave)
            self.assertEqual(entry['adapter_runtime_metadata'], python)
            self.assertEqual(entry['reference_runtime_qualification'], note)
            self.assertIsNone(entry['gate'])
            self.assertEqual(report['execution'], schedule)
            self.assertEqual(report['schedule_predecessor'], predecessor)
            self.assertEqual(report['analysis_predecessor'], analysis_predecessor)
            qualifications.write_text(json.dumps({'unplanned-metric': note}))
            with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'unplanned metrics'):
                main()
            scoring_manifest.write_text('{"changed_fixture":true}\n')
            with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'analysis predecessor'):
                main()


if __name__ == '__main__':
    unittest.main()
