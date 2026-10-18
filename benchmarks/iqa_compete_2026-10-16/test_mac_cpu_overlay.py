import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from patch_mac_cpu_counters import adapt

class OverlayTests(unittest.TestCase):
    def test_unknown_source_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'exact frozen'):
            adapt('different source')

    def test_original_checkout_alias_is_refused_before_any_mutation(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            root=Path(directory)
            (root/'Cargo.toml').write_text('original manifest')
            (root/'Cargo.lock').write_text('original lock')
            script=Path(__file__).with_name('patch_mac_cpu_counters.py')
            result=subprocess.run([sys.executable,str(script),str(root),str(root),str(root/'receipt.json')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('snapshot aliases original',result.stderr)
            self.assertEqual((root/'Cargo.toml').read_text(),'original manifest')
            self.assertEqual((root/'Cargo.lock').read_text(),'original lock')
            self.assertFalse((root/'receipt.json').exists())
