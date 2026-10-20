import json
from pathlib import Path
import tempfile
import unittest

from compare_canonical import sha
from prepare_canonical_grid import prepare


class GridPlanTests(unittest.TestCase):
    def test_contract_hashes_and_existing_outputs(self):
        scratch = Path.home() / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            ref, dis = root/'ref.rgb', root/'dis.rgb'
            ref.write_bytes(bytes(12))
            dis.write_bytes(bytes([1])*12)
            original = dict(bindings={}, binary='fixture-worker', rust_source='fixture-source',
                            cases=[dict(metric='fixture', threads=1, matched_scope=True,
                                        absolute_bound=1e-9, relative_bound=1e-12,
                                        scope='resident RGB8', canonical=dict(command=['{width}', '{reference}']))])
            source, pairs, output = root/'source.json', root/'pairs.json', root/'output.json'
            source.write_text(json.dumps(original))
            pairs.write_text(json.dumps(dict(pairs=[dict(pair='A', content='fixture', width=2,
                height=2, reference=ref.name, reference_sha256=sha(ref),
                distorted=dis.name, distorted_sha256=sha(dis))])))
            self.assertEqual(prepare(source, pairs, output, ['fixture'], [1, 8]), 2)
            plan = json.loads(output.read_text())
            self.assertEqual(json.loads(source.read_text()), original)
            self.assertEqual([c['threads'] for c in plan['cases']], [1, 8])
            for case in plan['cases']:
                self.assertEqual(case['absolute_bound'], 1e-9)
                self.assertEqual(case['relative_bound'], 1e-12)
                self.assertEqual(case['canonical'], original['cases'][0]['canonical'])
                self.assertEqual(case['reference'], str(ref))
            with self.assertRaises(FileExistsError):
                prepare(source, pairs, output, ['fixture'], [1])
            ref.write_bytes(bytes([2])*12)
            with self.assertRaisesRegex(ValueError, 'input hash mismatch'):
                prepare(source, pairs, root/'invalid.json', ['fixture'], [1])
            self.assertFalse((root/'invalid.json').exists())


if __name__ == '__main__':
    unittest.main()
