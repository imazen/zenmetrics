"""A scheduling successor must retain completed work and the scorer contract."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from canonical_run import sha
from canonical_successor import prepare


class Successor(unittest.TestCase):
    def test_completed_native_and_reference_rows_are_carried_without_scorers(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            old_scratch, old_output, source = root/'old', root/'results', root/'source'
            (old_scratch/'logs').mkdir(parents=True)
            old_output.mkdir()
            relative = Path('benchmarks/aic2026_full_repro_2026-10-20')
            (source/relative).mkdir(parents=True)
            for name in ('canonical_python.py', 'canonical_octave.py'):
                (source/relative/name).write_text('raise RuntimeError("scorers must never run during successor preparation")\n')
            binary = root/'binary'
            binary.write_text('invented binary binding')
            pairs = root/'pairs.tsv'
            pairs.write_text('distorted\tref_path\tdist_path\na.png\tr.png\ta.png\nb.png\tr.png\tb.png\n')
            native = old_output/'mad.native.tsv'
            native.write_text('distorted\tref_path\tdist_path\tscore\na.png\tr.png\ta.png\t1\nb.png\tr.png\tb.png\t2\n')
            (old_scratch/'logs/mad.native.receipt.json').write_text(json.dumps(dict(
                returncode=0, rows=2, result=str(native), result_sha256=sha(native))))
            chunks = old_output/'mad.canonical.chunks'
            chunks.mkdir()
            result = chunks/'worker.tsv'
            result.write_text('distorted\tvalue\terror\na.png\t1.25\t\n')
            (chunks/'bundle-0000.receipt.json').write_text(json.dumps(dict(returncode=0, outputs={str(result): sha(result)})))
            metadata = old_scratch/'metrics_cropped.csv'
            metadata.write_text('distorted,codec,bpp\na.png,JPG,1.00\nb.png,JPG,2.00\n')
            manifest = old_scratch/'manifest.json'
            original = dict(scratch=str(old_scratch), output=str(old_output), driver_freeze=str(source),
                binary=str(binary), binary_sha256=sha(binary), bindings={str(binary): sha(binary)},
                pairs=str(pairs), cohort_rows=2, metrics={'mad': {'kind': 'octave'}}, addendum={},
                execution={'default_workers': 3, 'workers': {}, 'rows_per_worker': 256})
            manifest.write_text(json.dumps(original))
            before = sha(manifest)
            new_path = prepare(manifest, root/'successor', root/'new-results', root/'new-source',
                               [str(root/'first.lock'), str(root/'second.lock')], 32)
            new = json.loads(new_path.read_text())
            self.assertEqual(sha(manifest), before)
            self.assertEqual(new['execution']['rows_per_worker'], 32)
            self.assertEqual(new['pairs'], str(pairs))
            self.assertEqual((root/'successor/metrics_cropped.csv').read_bytes(), metadata.read_bytes())
            self.assertEqual(new['bindings'][str(root/'successor/metrics_cropped.csv')], sha(metadata))
            self.assertEqual(new['metrics'], original['metrics'])
            self.assertEqual(new['execution']['environment_root'], str(old_scratch))
            self.assertEqual(new['schedule_predecessor']['retained_rows'], {'mad.native': 2, 'mad.canonical': 1})
            self.assertEqual(new['continuations']['mad.canonical']['rows'], 1)
            self.assertEqual((root/'new-results/mad.native.tsv').read_bytes(), native.read_bytes())
            for name in ('canonical_python.py', 'canonical_octave.py'):
                self.assertEqual(sha(source/relative/name), sha(root/'new-source'/relative/name))
            with self.assertRaisesRegex(ValueError, 'must be new'):
                prepare(manifest, root/'successor', root/'other-results', root/'other-source', [], 32)


if __name__ == '__main__':
    unittest.main()
