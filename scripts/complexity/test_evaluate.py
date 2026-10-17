import unittest
from evaluate import evaluate, validate

def tally(stages, **extra):
    return dict(schema='iqa-complexity-tally/1', metric='t', counter='test', source_commit='0', kind='classic-fr',
                assumptions=[], stages=stages, **extra)

def stage(id, **kw):
    s = dict(id=id, name=id, source=['x.rs:1-2'], elements='W*H', precision='f32', ops={}, access='pointwise')
    s.update(kw)
    return s

class Evaluate(unittest.TestCase):
    def test_separable_halo_is_per_axis(self):
        t = tally([stage('h', access='sep_stencil', radius=5, axis='x', ops={'mul': 11, 'add': 10}),
                   stage('v', access='sep_stencil', radius=5, axis='y', ops={'mul': 11, 'add': 10}, depends_on=['h'])])
        self.assertEqual(validate(t), [])
        r = evaluate(t, 64, 64)
        self.assertEqual((r['halo_x_px'], r['halo_y_px'], r['halo_px']), (5, 5, 5))
        self.assertEqual(r['flops_per_px'], 42)

    def test_chained_pyramid_reach_and_per_level_blur(self):
        t = tally([stage('down', access='downsample', levels=3, chain=True, radius=1, scale='2**l',
                         elements='(W/2**(l+1))*(H/2**(l+1))', ops={'add': 3}),
                   stage('blur', access='sep_stencil', levels=4, radius=2, scale='2**l', axis='x',
                         elements='(W/2**l)*(H/2**l)', ops={'fma': 1}, depends_on=['down'])])
        r = evaluate(t, 64, 64)
        self.assertEqual(r['halo_x_px'], (1 + 2 + 4) + 2 * 8)  # chain reach, then the coarsest blur
        self.assertEqual(r['halo_y_px'], 1 + 2 + 4)

    def test_implementation_class_is_reported_separately(self):
        t = tally([stage('a', ops={'add': 2}), stage('copy', ops={'mul': 1}, **{'class': 'implementation'})])
        r = evaluate(t, 8, 8)
        self.assertEqual((r['intrinsic_flops_per_px'], r['implementation_flops_per_px']), (2, 1))

    def test_rejects_unknown_classes_and_missing_sources(self):
        self.assertTrue(validate(tally([stage('a', ops={'flop': 1})])))
        self.assertTrue(validate(tally([stage('a', source=['nowhere'])])))
        self.assertTrue(validate(tally([stage('a', axis='z')])))

if __name__ == '__main__':
    unittest.main()
