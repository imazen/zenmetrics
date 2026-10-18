"""Private dependency overlays cannot touch the public source or accept new structure."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import patch_dvifmish as overlay


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path.home() / 'tmp')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.original, self.snapshot = self.root / 'original', self.root / 'snapshot'
        self.manifest = '[package]\nname="probe"\nversion="0.0.0"\n[features]\ncompete=[]\n[dependencies]\n'
        for tree in (self.original, self.snapshot):
            (tree / overlay.RELATIVE).parent.mkdir(parents=True)
            (tree / overlay.RELATIVE).write_text(self.manifest)
            (tree / 'Cargo.lock').write_text('original lock')

    def test_overlay_binds_both_changed_files_and_keeps_original_bytes(self):
        def metadata(*args, **kwargs):
            self.assertEqual(kwargs['cwd'], self.snapshot)
            (self.snapshot / 'Cargo.lock').write_text('private snapshot lock')
        receipt = self.root / 'receipt.json'
        with patch.object(overlay.subprocess, 'run', side_effect=metadata):
            overlay.apply(self.snapshot, self.original, receipt)
        self.assertEqual((self.original / overlay.RELATIVE).read_text(), self.manifest)
        self.assertEqual((self.original / 'Cargo.lock').read_text(), 'original lock')
        value = json.loads(receipt.read_text())
        self.assertEqual(set(value['files']), {overlay.RELATIVE, 'Cargo.lock'})
        self.assertIn(overlay.REVISION, (self.snapshot / overlay.RELATIVE).read_text())
        self.assertNotEqual(value['files']['Cargo.lock']['original_sha256'], value['files']['Cargo.lock']['adapted_sha256'])

    def test_source_alias_and_already_patched_manifest_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'aliases'):
            overlay.apply(self.original, self.original, self.root / 'receipt.json')
        (self.snapshot / overlay.RELATIVE).write_text(self.manifest + '\n# dvifmish already present\n')
        with patch.object(overlay.subprocess, 'run') as cargo:
            with self.assertRaisesRegex(ValueError, 'patch refused'):
                overlay.apply(self.snapshot, self.original, self.root / 'receipt.json')
            cargo.assert_not_called()


if __name__ == '__main__':
    unittest.main()
