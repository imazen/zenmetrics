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
