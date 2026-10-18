#!/usr/bin/env python3
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import durable
import prepare_coverage as coverage


class Retention(unittest.TestCase):
    def test_valid_attempt_survives_stale_pending_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = dict(durable_cell_id='one', metric_name='psnr', timing_phase='warm')
            attempt = root / 'cells/one/attempt-0001'
            attempt.mkdir(parents=True)
            (attempt / 'result.json').touch()
            saved = dict(frozen={'binary': 'original'})
            report = dict(identity=durable.digest(durable.encoded(saved)), cells=[dict(cell_id='one', status='pending')])
            def load(path):
                return saved if path.name == 'identity.json' else report
            with patch.object(durable, 'manifest', return_value=(None, [config], saved['frozen'])), patch.object(durable, 'load', side_effect=load), patch.object(durable, 'read_result', return_value={'status': 'ok'}), patch.object(durable, 'attempts', return_value=(attempt, [{'status': 'ok'}])) as verify:
                self.assertEqual(coverage.prior_completions(root, root / 'report.json'), {coverage.key(config)})
                verify.assert_called_once()
                verify.side_effect = durable.Refusal('retained log changed')
                with self.assertRaises(durable.Refusal):
                    coverage.prior_completions(root, root / 'report.json')

    def test_phase_and_repeat_have_distinct_identity(self):
        cell = dict(metric_name='psnr', timing_phase='cold', cold_repeat=0)
        self.assertNotEqual(coverage.key(cell), coverage.key(dict(cell, cold_repeat=1)))
        self.assertNotEqual(coverage.key(cell), coverage.key(dict(cell, timing_phase='warm')))


if __name__ == '__main__':
    unittest.main()
