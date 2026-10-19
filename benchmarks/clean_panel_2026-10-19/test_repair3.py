"""Actual venv/.pth archive-prefix discovery and fresh-interpreter controls."""
import importlib._bootstrap_external
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import venv
import zipfile
import bytecode_inventory


class ArchivePrefix(unittest.TestCase):
    def test_discovered_archive_prefix_mutation_deletion_and_retarget(self):
        for suffix in ('.zip', '.egg'):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                scripts = root/'scripts'; scripts.mkdir()
                envroot = root/'env'
                venv.EnvBuilder(with_pip=False).create(envroot)
                python = str(envroot/'bin/python')
                site = Path(subprocess.check_output([python, '-B', '-c',
                    'import sysconfig;print(sysconfig.get_path("purelib"))'],
                    env=bytecode_inventory.environment(), text=True).strip())
                archive = root/('external'+suffix)
                prefix = str(archive)+'/inside/nested'
                pth = site/'owned_archive.pth'
                pth.write_text('import sys;sys.path.append('+repr(prefix)+')\n')
                pth_bytes = pth.read_bytes()
                def save(path, value):
                    code = compile(f'VALUE={value}\n', 'owned_zip_module.py', 'exec')
                    pyc = importlib._bootstrap_external._code_to_timestamp_pyc(code, 0, 0)
                    with zipfile.ZipFile(path, 'w') as z:
                        z.writestr('inside/nested/owned_zip_module.pyc', pyc)
                def imported():
                    return subprocess.check_output([python, '-B', '-c',
                        'import owned_zip_module;print(owned_zip_module.VALUE)'],
                        cwd=scripts, env=bytecode_inventory.environment(), text=True).strip()
                probes = [dict(executable=python, script_directory=str(scripts))]
                save(archive, 1)
                binding = bytecode_inventory.capture(probes)
                self.assertEqual(imported(), '1')
                bytecode_inventory.verify(binding)
                state = next(s for s in binding['inventory']['roots'] if s['path']==prefix)
                self.assertEqual(state['kind'], 'archive-prefix')
                self.assertEqual(state['prefix'], 'inside/nested')
                self.assertEqual(state['container_resolved'], str(archive.resolve()))
                save(archive, 2)
                self.assertEqual(imported(), '2')
                with self.assertRaises(ValueError): bytecode_inventory.verify(binding)
                binding = bytecode_inventory.capture(probes)
                archive.unlink()
                with self.assertRaises(subprocess.CalledProcessError): imported()
                with self.assertRaises(ValueError): bytecode_inventory.verify(binding)
                one = root/'one'; two = root/'two'
                save(one, 3); save(two, 4)
                archive.symlink_to(one)
                binding = bytecode_inventory.capture(probes)
                self.assertEqual(imported(), '3')
                bytecode_inventory.verify(binding)
                archive.unlink(); archive.symlink_to(two)
                self.assertEqual(imported(), '4')
                with self.assertRaises(ValueError): bytecode_inventory.verify(binding)
                # Identical bytes with a changed target must also reject.
                two.write_bytes(one.read_bytes())
                archive.unlink(); archive.symlink_to(one)
                binding = bytecode_inventory.capture(probes)
                archive.unlink(); archive.symlink_to(two)
                self.assertEqual(imported(), '3')
                with self.assertRaises(ValueError): bytecode_inventory.verify(binding)
                self.assertEqual(pth.read_bytes(), pth_bytes)

    def test_genuinely_absent_root_and_unsupported_file_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root/'python999.zip'
            binding = bytecode_inventory.capture([], [str(missing)])
            self.assertEqual(binding['inventory']['roots'][0]['kind'], 'absent')
            bytecode_inventory.verify(binding)
            missing.write_bytes(b'not an archive')
            with self.assertRaises(ValueError): bytecode_inventory.verify(binding)
            with self.assertRaisesRegex(ValueError, 'unsupported file-prefix'):
                bytecode_inventory.capture([], [str(missing/'inside')])
