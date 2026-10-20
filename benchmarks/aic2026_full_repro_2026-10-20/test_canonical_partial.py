"""Progress must count retained and successor rows once, without qualifying them."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from canonical_partial import main
from canonical_run import sha


class PartialContinuation(unittest.TestCase):
    @staticmethod
    def counts_fixture(root):
        (root/'logs').mkdir()
        work = root/'work'
        work.mkdir()
        pairs = root/'pairs.tsv'
        pairs.write_text('distorted\tref_path\tdist_path\n'+''.join(
            f'{key}.png\tr.png\t{key}.png\n' for key in 'abcde'))
        native = work/'metric.native.tsv'
        native.write_text('distorted\tref_path\tdist_path\tscore\n'+''.join(
            f'{key}.png\tr.png\t{key}.png\t{value}\n'
            for key, value in zip('abcde', ['1', '2', 'nan', '4', '5'])))
        (root/'logs/metric.native.receipt.json').write_text(json.dumps(dict(
            returncode=0, rows=5, result=str(native), result_sha256=sha(native))))
        chunks = work/'metric.canonical.chunks'
        chunks.mkdir()
        result = chunks/'worker.jsonl'
        observations = [dict(distorted='a.png', value=1.25),
            dict(distorted='b.png', error='invented reference refusal'),
            dict(distorted='c.png', value=3), dict(distorted='d.png', value='nan'),
            dict(distorted='e.png', value='invalid')]
        result.write_text(''.join(json.dumps(row)+'\n' for row in observations))
        receipt = chunks/'bundle-0000.receipt.json'
        receipt.write_text(json.dumps(dict(returncode=0, outputs={str(result): sha(result)})))
        manifest = root/'manifest.json'
        manifest.write_text(json.dumps(dict(scratch=str(root), output=str(work), pairs=str(pairs),
            cohort_rows=5, metrics={'metric': dict(kind='python', reference='invented fixture')})))
        return manifest, native, result, receipt

    def test_counts_preserve_refusals_and_nonfinite_values_without_statistics(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            manifest, _, _, _ = self.counts_fixture(root)
            output = root/'counts.json'
            with patch('sys.argv', ['partial', str(manifest), 'metric', str(output), '--counts-only']), \
                    patch('canonical_partial.compare', side_effect=AssertionError('statistics must not run')), \
                    redirect_stdout(io.StringIO()):
                main()
            report = json.loads(output.read_text())
            counts = report['identity_counts']
            self.assertEqual(counts['native_rows'], 5)
            self.assertEqual(counts['native_nonfinite_values'], 1)
            self.assertEqual(counts['reference_rows'], 5)
            self.assertEqual(counts['reference_errors'], 1)
            self.assertEqual(counts['reference_nonfinite_values'], 2)
            self.assertEqual(counts['finite_comparisons'], 1)
            self.assertIsNone(report['comparison'])
            self.assertEqual(report['worst_five'], [])
            self.assertEqual(report['full_cohort_qualification'], 'pending')

    def test_counts_refuse_changed_outputs_and_false_complete_native_receipts(self):
        for changed in ('worker', 'native-identities'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
                root = Path(directory)
                manifest, native, result, _ = self.counts_fixture(root)
                if changed == 'worker':
                    result.write_text(result.read_text().replace('1.25', '1.5'))
                    message = 'reference chunk changed'
                else:
                    native.write_text(native.read_text().replace('e.png\tr.png\te.png\t5\n', ''))
                    receipt = root/'logs/metric.native.receipt.json'
                    data = json.loads(receipt.read_text())
                    data['result_sha256'] = sha(native)
                    receipt.write_text(json.dumps(data))
                    message = 'complete requested cohort'
                with patch('sys.argv', ['partial', str(manifest), 'metric', str(root/'counts.json'), '--counts-only']), \
                        self.assertRaisesRegex(ValueError, message):
                    main()

    def test_progress_combines_retained_and_new_completed_bundles(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            (root/'logs').mkdir()
            work = root/'work'
            work.mkdir()
            pairs = root/'pairs.tsv'
            pairs.write_text('distorted\tref_path\tdist_path\na.png\tr.png\ta.png\nb.png\tr.png\tb.png\n')
            native = work/'metric.native.tsv'
            native.write_text('distorted\tref_path\tdist_path\tscore\na.png\tr.png\ta.png\t1\nb.png\tr.png\tb.png\t2\n')
            (root/'logs/metric.native.receipt.json').write_text(json.dumps(dict(
                returncode=0, rows=2, result=str(native), result_sha256=sha(native))))
            old_result = root/'old.jsonl'
            old_result.write_text(json.dumps(dict(distorted='a.png', value=1.25))+'\n')
            old_receipt = root/'old.receipt.json'
            old_receipt.write_text(json.dumps(dict(returncode=0, outputs={str(old_result): sha(old_result)})))
            chunks = work/'metric.canonical.chunks'
            chunks.mkdir()
            result = chunks/'worker.jsonl'
            result.write_text(json.dumps(dict(distorted='b.png', value=2.5))+'\n')
            (chunks/'bundle-0000.receipt.json').write_text(json.dumps(dict(returncode=0, outputs={str(result): sha(result)})))
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(dict(scratch=str(root), output=str(work), pairs=str(pairs),
                cohort_rows=2, metrics={'metric': dict(kind='python', reference='invented fixture')},
                continuations={'metric.canonical': dict(path=str(old_result), sha256=sha(old_result), rows=1,
                    receipts=[dict(path=str(old_receipt), sha256=sha(old_receipt))])})))
            report_path = root/'partial.json'
            with patch('sys.argv', ['partial', str(manifest), 'metric', str(report_path)]), redirect_stdout(io.StringIO()):
                main()
            report = json.loads(report_path.read_text())
            self.assertEqual((report['reference_rows'], report['comparison']['n'], report['bundles']), (2, 2, 2))
            self.assertEqual(report['full_cohort_qualification'], 'pending')
            self.assertEqual(report['comparison']['max_d'], .5)
            self.assertEqual(report['files'][str(old_receipt)], sha(old_receipt))


if __name__ == '__main__':
    unittest.main()
