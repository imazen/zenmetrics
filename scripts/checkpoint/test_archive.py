"""Exercise host-independent manifest paths and the real archive verifier."""
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

from package import artifact_hashes

spec = importlib.util.spec_from_file_location('preparer', Path(__file__).with_name('prepare-release.py'))
preparer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparer)


class ArchiveControls(unittest.TestCase):
    def test_windows_relative_paths_are_portable(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch:
            payload = Path(scratch) / 'payload'
            payload.write_bytes(b'checkpoint executor')
            artifact = Mock()
            artifact.is_file.return_value = True
            artifact.relative_to.return_value = PureWindowsPath('drivers/latency.exe')
            artifact.open = payload.open
            package = Mock()
            package.rglob.return_value = [artifact]
            self.assertEqual(artifact_hashes(package), {
                'drivers/latency.exe': hashlib.sha256(payload.read_bytes()).hexdigest()})

    def test_nested_archive_roundtrip_and_tamper_refusal(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as scratch:
            root = Path(scratch)
            package = root / 'package'
            (package / 'drivers').mkdir(parents=True)
            (package / 'drivers/latency.exe').write_bytes(b'executor fixture')
            source = 'a' * 40
            manifest = {'schema': 1, 'source_commit': source,
                        'files': artifact_hashes(package)}
            (package / 'checkpoint.json').write_text(json.dumps(manifest))
            archive = root / 'checkpoint.tar.gz'
            command = [sys.executable, str(Path(__file__).with_name('archive.py')),
                       str(package), str(archive)]
            subprocess.run(command, check=True)
            self.assertEqual(preparer.verify_archive(archive, source), manifest)
            (package / 'drivers/latency.exe').write_bytes(b'changed executor')
            rejected = subprocess.run(command[:-1] + [str(root / 'tampered.tar.gz')],
                                      capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('Artifact hash mismatch', rejected.stderr)
            self.assertFalse((root / 'tampered.tar.gz').exists())


if __name__ == '__main__':
    unittest.main()
