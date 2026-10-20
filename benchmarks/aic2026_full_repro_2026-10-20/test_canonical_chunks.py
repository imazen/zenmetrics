"""Partial output recovery must preserve exact records and reject changed controls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from canonical_chunks import continuation_records, merge, records, scoring_environment
from canonical_run import admitted_command, sha
import json


class ChunkRecovery(unittest.TestCase):
    def test_admission_preserves_configured_lock_order(self):
        locks = [str(Path.home()/'tmp/first.lock'), str(Path.home()/'tmp/second.lock')]
        command = admitted_command({'heavy_locks': locks}, ['scorer', 'pairs.tsv'])
        self.assertEqual(command[:4], ['flock', locks[0], 'flock', locks[1]])
        self.assertEqual(command[-7:], ['--mem', '12G', '--jobs', '12', '--', 'scorer', 'pairs.tsv'])

    def test_admission_refuses_missing_relative_and_duplicate_locks(self):
        lock = str(Path.home()/'tmp/first.lock')
        for manifest in ({}, {'heavy_locks': ['relative.lock']}, {'heavy_locks': [lock, lock]}):
            with self.assertRaisesRegex(ValueError, 'heavy_locks'):
                admitted_command(manifest, ['scorer'])

    def test_successor_imports_only_exact_completed_bundle_rows(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            result = root/'result.tsv'
            result.write_text('distorted\tvalue\terror\na.png\t1.25\t\n')
            receipt = root/'receipt.json'
            receipt.write_text(json.dumps(dict(returncode=0, outputs={str(result): sha(result)})))
            continuation = dict(path=str(result), sha256=sha(result), rows=1,
                                receipts=[dict(path=str(receipt), sha256=sha(receipt))])
            observed, _ = continuation_records(continuation, False)
            self.assertEqual(list(observed), ['a.png'])
            result.write_text('distorted\tvalue\terror\na.png\t2.5\t\n')
            continuation['sha256'] = sha(result)
            with self.assertRaisesRegex(ValueError, 'worker output changed'):
                continuation_records(continuation, False)

    def test_successor_rejects_unreceipted_or_failed_rows(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            result = root/'result.tsv'
            result.write_text('distorted\tvalue\terror\na.png\t1.25\t\n')
            continuation = dict(path=str(result), sha256=sha(result), rows=1, receipts=[])
            with self.assertRaisesRegex(ValueError, 'completed bundle identities'):
                continuation_records(continuation, False)
            receipt = root/'receipt.json'
            receipt.write_text(json.dumps(dict(returncode=1, outputs={str(result): sha(result)})))
            continuation['receipts'] = [dict(path=str(receipt), sha256=sha(receipt))]
            with self.assertRaisesRegex(ValueError, 'successful bundles'):
                continuation_records(continuation, False)

    def test_successor_can_preserve_exact_scoring_environment(self):
        original = Path.home()/'tmp/original-attempt'
        successor = Path.home()/'tmp/successor-attempt'
        old = scoring_environment({}, {}, original)
        new = scoring_environment({'execution': {'environment_root': str(original)}}, {}, successor)
        self.assertEqual(old, new)

    def test_inherited_secret_is_not_serialized_in_worker_settings(self):
        with patch.dict('os.environ', {'AIC_TEST_SECRET': 'invented-control-value'}):
            settings = scoring_environment({}, {}, Path.home()/'tmp')
        self.assertNotIn('AIC_TEST_SECRET', settings)
        self.assertEqual(settings['RAYON_NUM_THREADS'], '1')

    def test_partial_tail_never_becomes_a_valid_numeric_result(self):
        scratch = Path.home()/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory)/'scores.tsv'
            path.write_text('distorted\tscore\na.png\t0.123456\nb.png\t0.12')
            recovered, _ = records(path, partial=True)
            self.assertEqual(list(recovered), ['a.png'])
            self.assertEqual(recovered['a.png']['score'], '0.123456')
            with self.assertRaisesRegex(ValueError, 'unterminated'):
                records(path)

    def test_exact_overlap_control_and_changed_control(self):
        retained = {'a.png': {'distorted': 'a.png', 'score': '0.123456'}}
        merge(retained, {'a.png': dict(retained['a.png'])})
        with self.assertRaisesRegex(ValueError, 'overlap control changed'):
            merge(retained, {'a.png': dict(retained['a.png'], score='0.123457')})

    def test_duplicate_complete_records_refuse(self):
        scratch = Path.home()/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory)/'scores.jsonl'
            path.write_text('{"distorted":"a.png","value":1}\n'*2)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                records(path, jsonl=True)


if __name__ == '__main__':
    unittest.main()
