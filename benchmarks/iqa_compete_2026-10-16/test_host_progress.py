"""Historical remote cohorts must not overstate sample or map coverage."""
import unittest
from host_progress import summarize


class Progress(unittest.TestCase):
    def test_medium_three_samples_do_not_meet_five(self):
        row = dict(metric_name='example', size_name='aic-medium', threads=1,
                   mode='score', status='ok', n=3, coverage_minimum_n=3)
        result = summarize([row], [dict(metric_name='example', gpu=False)], [1], False)
        self.assertEqual(result['per_size']['aic-medium']['timed_minimum_met'], 0)
        self.assertEqual(result['per_size']['aic-medium']['refused_or_short'], 1)

    def test_unsupported_adapter_is_not_metric_map_unavailability(self):
        row = dict(metric_name='fsim-y', size_name='cid22', threads=1,
                   mode='score+map', status='unsupported', n=0)
        result = summarize([row], [dict(metric_name='fsim-y', gpu=False)], [1], False)
        self.assertEqual(result['per_size']['cid22']['adapter_omissions'], 1)

    def test_duplicate_attempts_must_be_collected_first(self):
        row = dict(metric_name='example', size_name='cid22', threads=1, mode='score')
        with self.assertRaises(ValueError):
            summarize([row, row], [dict(metric_name='example', gpu=False)], [1], False)


if __name__ == '__main__':
    unittest.main()
