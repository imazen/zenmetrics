"""Owner-authorized finite Metal timing policy; strict historical path retained."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import durable


def bits(v):
    return struct.pack('>d', v).hex()


class FiniteGpuPolicy(unittest.TestCase):
    def setUp(self):
        self.config = dict(metric_name='stlpips-alex', device='metal', timing_phase='warm', only_mode='score',
            gpu_score_repeatability_policy=durable.FINITE_GPU_POLICY,
            worker_environment=dict(WGPU_BACKEND='metal', ZENMETRICS_NN_ENGINE='ort-webgpu'))
        self.row = dict(self.config, mode='score', status='variant_mismatch', n=10, median_ns=100,
            score=1.16, score_bits=bits(1.16), round_means_ns=[100]*10,
            zenbench_samples=[dict(gate_clean=True, iterations=2)]*10,
            gpu_call_observation_count=30, gpu_finite_observation_count=30,
            gpu_nonfinite_observation_count=0, gpu_call_bit_variation_count=3,
            gpu_reference_score_bits=bits(1.), gpu_min_score_bits=bits(1.), gpu_max_score_bits=bits(1.16),
            gpu_one_different_score_bits=bits(1.16), gpu_max_abs_delta_bits=bits(1.16-1.),
            gpu_max_abs_delta=1.16-1., gpu_max_relative_delta=1.16-1., gpu_delta_overflow=False,
            gpu_score_range_includes_baseline=True)

    def validate(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            p = Path(d)/'worker.log'
            p.write_text('IQA_ROW '+json.dumps(self.row)+'\n')
            return durable.validate_row(p, self.config)

    def test_finite_large_change_is_timing_only_and_flagged(self):
        r = self.validate()
        self.assertEqual(r['status'], 'ok')
        self.assertEqual(r['raw_status'], 'variant_mismatch')
        self.assertTrue(r['numerical_instability_flag'])
        self.assertFalse(r['score_repeatability_qualified'])
        self.assertEqual(self.row['status'], 'variant_mismatch')

    def test_no_opt_in_keeps_original_refusal(self):
        self.config.pop('gpu_score_repeatability_policy')
        with self.assertRaisesRegex(durable.Refusal, 'variant_mismatch'):
            self.validate()

    def test_missing_full_observations_refused(self):
        self.row.pop('gpu_nonfinite_observation_count')
        with self.assertRaises(durable.Refusal): self.validate()

    def test_hidden_nonfinite_refused_even_with_finite_headline(self):
        self.row.update(gpu_nonfinite_observation_count=1, gpu_finite_observation_count=29)
        with self.assertRaises(durable.Refusal): self.validate()

    def test_wrong_scope_refused(self):
        for key, value in [('device','cuda'), ('only_mode','score+map'), ('timing_phase','cold')]:
            old = self.config[key]
            self.config[key] = value
            with self.assertRaises(durable.Refusal): self.validate()
            self.config[key] = old

    def test_dirty_or_missing_or_short_rounds_refused(self):
        original = copy.deepcopy(self.row)
        for samples in [[dict(gate_clean=False,iterations=2)]*10, [], [dict(gate_clean=True,iterations=4)]*10]:
            self.row['zenbench_samples'] = samples
            with self.assertRaises(durable.Refusal): self.validate()
        self.row = original
        self.row['n'] = 9
        with self.assertRaises(durable.Refusal): self.validate()

    def test_incoherent_delta_or_extrema_refused(self):
        for key, value in [('gpu_max_abs_delta', 0), ('gpu_min_score_bits', bits(1.1)), ('gpu_max_relative_delta', 0)]:
            old = self.row[key]
            self.row[key] = value
            with self.assertRaises(durable.Refusal): self.validate()
            self.row[key] = old

    def test_finite_extreme_delta_overflow_is_explicit(self):
        high = float.fromhex('0x1.fffffffffffffp+1023')
        self.row.update(score=-high, score_bits=bits(-high), gpu_reference_score_bits=bits(high),
            gpu_min_score_bits=bits(-high), gpu_max_score_bits=bits(high),
            gpu_one_different_score_bits=bits(-high), gpu_max_abs_delta_bits=bits(float('inf')),
            gpu_max_abs_delta=None, gpu_max_relative_delta=None, gpu_delta_overflow=True)
        self.assertTrue(self.validate()['numerical_instability_flag'])


if __name__ == '__main__': unittest.main()
