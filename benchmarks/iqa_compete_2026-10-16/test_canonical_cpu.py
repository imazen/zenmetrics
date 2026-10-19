import unittest
from prepare_canonical_cpu import cells, SIZES
from rebuild_canonical_mac import checked_control


class CanonicalCpu(unittest.TestCase):
    def test_control_status_ok_does_not_accept_flagged_rounds(self):
        report = dict(complete=True, cells=[dict(status='ok',
            config=dict(threads=t, timing_phase=phase),
            row=dict(score_bits='3ff0000000000000', n=10,
                zenbench_samples=[dict(gate_clean=True) for _ in range(10)]))
            for t in (1, 8) for phase in ('cold', 'warm')])
        checked_control(report)
        report['cells'][-1]['row']['zenbench_samples'][0]['gate_clean'] = False
        with self.assertRaisesRegex(ValueError, 'flagged'):
            checked_control(report)

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
