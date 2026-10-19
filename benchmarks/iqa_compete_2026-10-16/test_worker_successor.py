"""Replacement bundles preserve assets and reject a mismatched executable receipt."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare_worker_successor as successor
from mac_service import verify


class Successor(unittest.TestCase):
    def test_preserved_assets_new_runner_and_tamper_refusal(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            source = root/'old'
            (source/'bin').mkdir(parents=True)
            for name in ['durable.py','mac_service.py','select_durable.py','build.json','inputs.meta.json','asset','bin/iqa-compete']:
                (source/name).write_text('original-'+name)
            manifest = dict(binary_sha256=successor.sha(source/'bin/iqa-compete'),
                cells=[dict(metric_name='nn')],metadata={},input_sha256=dict(asset=successor.sha(source/'asset')))
            (source/'manifest.json').write_text(json.dumps(manifest))
            (source/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(source)):successor.sha(p) for p in source.rglob('*') if p.is_file()}))
            before = {str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            binary = root/'worker'
            binary.write_bytes(b'new-binary')
            inventory = b'[{"metric_name":"nn","compiled":true}]'
            build = root/'build.json'
            build.write_text(json.dumps(dict(binary_sha256=successor.sha(binary),inventory_sha256=hashlib.sha256(inventory).hexdigest(),source_commit='test',features='test')))
            target = root/'new'
            with patch('sys.argv',['prepare',str(source),str(binary),str(build),str(target)]), patch.object(successor.subprocess,'check_output',return_value=inventory):
                successor.main()
            verify(target)
            self.assertEqual((target/'asset').read_bytes(),before['asset'])
            self.assertEqual({str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()},before)
            self.assertEqual((target/'durable.py').read_bytes(),(successor.HERE/'durable.py').read_bytes())
            binary.write_bytes(b'tampered')
            with patch('sys.argv',['prepare',str(source),str(binary),str(build),str(root/'bad')]):
                with self.assertRaisesRegex(ValueError,'differs'):
                    successor.main()
            self.assertFalse((root/'bad').exists())
            binary.write_bytes(b'new-binary')
            manifest['cells'][0]['worker_environment'] = dict(ZENMETRICS_NN_ENGINE='ort-webgpu')
            (source/'manifest.json').write_text(json.dumps(manifest))
            hashes = json.loads((source/'SHA256SUMS.json').read_bytes())
            hashes['manifest.json'] = successor.sha(source/'manifest.json')
            (source/'SHA256SUMS.json').write_text(json.dumps(hashes))
            with patch('sys.argv',['prepare',str(source),str(binary),str(build),str(root/'bad-engine')]), patch.object(successor.subprocess,'check_output',return_value=inventory):
                with self.assertRaisesRegex(ValueError,'WebGPU feature'):
                    successor.main()
            self.assertFalse((root/'bad-engine').exists())


if __name__ == '__main__': unittest.main()
