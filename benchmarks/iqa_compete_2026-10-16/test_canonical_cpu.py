import unittest
from prepare_canonical_cpu import cells, SIZES


class CanonicalCpu(unittest.TestCase):
    def test_full_scalar_protocol(self):
        pairs = {s: dict(width=512, height=512, method='fixture') for s in SIZES}
        rows = cells(['psnr', 'dvifmish-port-min-v0.2'], pairs)
        self.assertEqual(len(rows), 96)
        for metric in ('psnr', 'dvifmish-port-min-v0.2'):
            for size in SIZES:
                for threads in (1, 8):
                    group = [r for r in rows if (r['metric_name'], r['size_name'], r['threads']) == (metric, size, threads)]
                    self.assertEqual([r['cold_repeat'] for r in group if r['timing_phase'] == 'cold'], [0, 1, 2])
                    warm = [r for r in group if r['timing_phase'] == 'warm']
                    self.assertEqual(len(warm), 1)
                    self.assertEqual(warm[0]['required_samples'], 10)
        self.assertTrue(all(r['only_mode'] == 'score' and r['sample_target_ns'] == 100_000_000 and not r['coverage_first'] for r in rows))

    def test_geometry_refuses_8k_even_with_other_label(self):
        pairs = {s: dict(width=512, height=512, method='fixture') for s in SIZES}
        pairs['aic-small']['width'] = 7680
        with self.assertRaisesRegex(ValueError, '8K'):
            cells(['psnr'], pairs)


if __name__ == '__main__':
    unittest.main()
