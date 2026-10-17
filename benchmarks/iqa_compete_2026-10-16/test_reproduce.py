"""Negative controls for reproduction identity, input integrity and process cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from bench import key, checked
from resources import scoped

class Reproduce(unittest.TestCase):
    def test_identity_changes_with_provenance(self):
        base=dict(commit='a',host='i265',binaries={'rust':'x'},models={},input_manifest_sha256='b')
        for change in [dict(commit='b'),dict(host='mac'),dict(binaries={'rust':'y'}),dict(models={'weights':'new'}),dict(input_manifest_sha256='c')]:
            self.assertNotEqual(key(base),key(base|change))
        self.assertEqual(key(base),key(dict(reversed(list(base.items())))))

    def test_checksum_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as root:
            path=Path(root)/'corpus';path.write_bytes(b'changed corpus')
            with self.assertRaisesRegex(ValueError,'Checksum differs'):
                checked(path,'0'*64)

    def test_missing_wrapper_runs_plain_with_timeout(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as root,patch('resources.Path.home',return_value=Path(root)):
            command=scoped(['example'],seconds=4,memory='1G',jobs=1)
            self.assertEqual(command[-3:],['timeout','4','example'])
            self.assertNotIn('run-heavy',' '.join(command))

    def test_timeout_reaps_real_child(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as root:
            pid=Path(root)/'pid'
            command=[sys.executable,str(Path(__file__).with_name('resources.py')),'timeout','.3',sys.executable,'-c',
                     'import os,pathlib,time;pathlib.Path('+repr(str(pid))+').write_text(str(os.getpid()));time.sleep(30)']
            result=subprocess.run(command,timeout=5)
            self.assertEqual(result.returncode,124)
            child=int(pid.read_text())
            with self.assertRaises(ProcessLookupError):os.kill(child,0)

if __name__=='__main__':unittest.main()
