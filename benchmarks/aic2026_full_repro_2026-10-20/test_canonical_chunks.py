"""Partial output recovery must preserve exact records and reject changed controls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from canonical_chunks import merge, records, scoring_environment
from canonical_run import admitted_command


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
