"""Publication controls use fake commands; no GitHub or Git mutation occurs."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('publisher', Path(__file__).with_name('publish-release.py'))
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class PublicationControls(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(dir=Path.home() / 'tmp')
        self.addCleanup(self.scratch.cleanup)
        self.out = Path(self.scratch.name) / 'assets'
        self.readme = b'Reviewed source README\n'
        self.source = 'a' * 40
        self.tag = 'v0.1.0-checkpoint.control'
        self.argv = ['publish-release.py', '--run', '123', '--source', self.source,
                     '--tag', self.tag, '--out', str(self.out), '--readme-approved',
                     '--readme-sha256', hashlib.sha256(self.readme).hexdigest()]
        self.commands = []
        self.existing_tag = False
        self.refuse_preparation = False

    def output(self, command, **kwargs):
        if command[0] == 'git':
            self.assertEqual(command, ['git', 'show', f'{self.source}:README.md'])
            return self.readme
        self.assertEqual(command[:2], ['gh', 'api'])
        if 'releases/generate-notes' in ' '.join(command):
            return json.dumps({'body': 'Changes by @contributor and @lilith'})
        return json.dumps([{'ref': f'refs/tags/{self.tag}'}] if self.existing_tag else [])

    def command(self, command, **kwargs):
        self.commands.append(command)
        if command[1].endswith('prepare-release.py'):
            if self.refuse_preparation:
                raise subprocess.CalledProcessError(2, command)
            self.out.mkdir()
            archive = self.out / 'checkpoint-control.tar.gz'
            archive.write_bytes(b'verified by preparer control')
            (self.out / 'release-assets.json').write_text(json.dumps(
                {'schema': 1, 'source': self.source, 'archives': [str(archive)]}))

    def invoke(self, argv=None):
        with patch.object(sys, 'argv', argv or self.argv), \
             patch.object(publisher.subprocess, 'check_output', self.output), \
             patch.object(publisher.subprocess, 'run', self.command), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            publisher.main()

    def test_missing_approval_refuses_before_commands(self):
        with self.assertRaises(SystemExit):
            self.invoke([arg for arg in self.argv if arg != '--readme-approved'])
        self.assertEqual(self.commands, [])

    def test_wrong_readme_refuses_before_preparation(self):
        self.readme = b'Unreviewed replacement\n'
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.commands, [])

    def test_failed_verification_never_tags(self):
        self.refuse_preparation = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke()
        self.assertEqual(len(self.commands), 1)

    def test_existing_tag_never_mutates(self):
        self.existing_tag = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(len(self.commands), 1)

    def test_dry_run_never_tags_or_publishes(self):
        self.invoke(self.argv + ['--dry-run'])
        self.assertEqual(len(self.commands), 1)

    def test_public_index_contains_portable_release_filenames(self):
        self.invoke(self.argv + ['--dry-run'])
        index = json.loads((self.out / 'release-assets.json').read_text())
        self.assertEqual(index['archives'], ['checkpoint-control.tar.gz'])

    def test_tag_then_push_then_attach_archives_and_checksums(self):
        self.invoke()
        self.assertEqual(self.commands[1], ['git', 'tag', self.tag, self.source])
        self.assertEqual(self.commands[2], ['git', 'push', 'origin', f'refs/tags/{self.tag}'])
        release = self.commands[3]
        self.assertEqual(release[:4], ['gh', 'release', 'create', self.tag])
        archive = str(self.out / 'checkpoint-control.tar.gz')
        self.assertIn(archive, release)
        self.assertIn(archive + '.sha256', release)
        self.assertIn(str(self.out / 'release-assets.json'), release)
        self.assertIn('--prerelease', release)
        self.assertNotIn('--clobber', release)
        self.assertEqual((self.out / 'release-notes.md').read_text(),
                         'Changes by contributor and @lilith\n')


if __name__ == '__main__':
    unittest.main()
