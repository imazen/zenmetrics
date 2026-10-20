#!/usr/bin/env python3
"""Static HDR entry/negative controls; no Octave, native build or inference.

Caller holds shared heavy admission. These checks do not qualify original-call
bit identity or native numerical parity.
"""
import copy
from pathlib import Path
import tempfile
import unittest

from first_boundary_hdrvdp_octave import script
from first_boundary_hdrvdp_plan import sha


class EntryControls(unittest.TestCase):
    def setUp(self):
        # The session's disk-backed TMPDIR is part of caller admission.
        import os
        if not os.environ.get('TMPDIR'):
            raise ValueError('explicit disk-backed TMPDIR required')
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tree = self.root / 'official'
        tree.mkdir()
        # Invented source identities are fixtures, not claimed metric bodies.
        paths = []
        for name in ('hdrvdp.m', 'hdrvdp3.m'):
            path = tree / name
            path.write_text('% synthetic binding fixture; never executed\n')
            paths.append(path)
        for name in ('reference.rgb', 'distorted.rgb', 'octave', 'geomean.m'):
            path = self.root / name
            path.write_bytes(bytes(range(18)) if name.endswith('.rgb') else b'fixture\n')
            paths.append(path)
        self.config = dict(metric='hdrvdp', width=3, height=2, threads=1, samples=1,
            phase='first', tree=str(tree), reference=str(self.root / 'reference.rgb'),
            distorted=str(self.root / 'distorted.rgb'), octave=str(self.root / 'octave'),
            geomean=str(self.root / 'geomean.m'), octave_version='11.1.0', image_version='2.20.1',
            bindings={str(path): sha(path) for path in paths})

    def test_first_v2_uses_original_srgb_call_without_priming(self):
        text = script(self.config)
        call = "result=hdrvdp(distorted,reference,'sRGB-display',30,{});score=result.Q;"
        self.assertEqual(text.count(call), 1)
        self.assertLess(text.index('clock=tic();'), text.index(call))
        self.assertIn("fprintf('COLD ", text)
        for wrong_convention in ('1023', '200*', "'XYZ'", "'rgb-bt.709'"):
            self.assertNotIn(wrong_convention, text)

    def test_first_v3_preserves_quality_jod_and_declared_viewing(self):
        config = dict(self.config, metric='hdrvdp3')
        text = script(config)
        self.assertIn("result=hdrvdp3('quality',distorted,reference,'sRGB-display',30,", text)
        self.assertIn("'age',24,'surround','none'", text)
        self.assertIn('score=result.Q_JOD;', text)
        self.assertNotIn('score=result.Q;', text)

    def test_warm_priming_is_explicit_and_outside_the_clock(self):
        text = script(dict(self.config, phase='warm', samples=5))
        call = "result=hdrvdp(distorted,reference,'sRGB-display',30,{});score=result.Q;"
        self.assertEqual(text.count(call), 2)
        self.assertLess(text.index(call), text.index('clock=tic();'))
        self.assertIn('for iteration=1:5', text)
        self.assertIn("fprintf('SAMPLE ", text)
        self.assertNotIn("fprintf('COLD ", text)

    def test_direct_control_has_independent_rectangular_ingress(self):
        candidate = script(self.config)
        direct = script(self.config, direct_control=True)
        self.assertIn('[r,n]=fread(f,[3,6],"*uint8")', direct)
        self.assertIn("reshape(double(r'),[3,2,3]),[2,1,3]", direct)
        self.assertIn('reshape(r,[3,3,2]),[3,2,1]', candidate)
        # Each upstream call still has precisely the same declared arguments.
        self.assertEqual([s for s in direct.splitlines() if s.startswith('result=')],
                         [s for s in candidate.splitlines() if s.startswith('result=')])

    def test_changed_source_and_unbound_file_refuse(self):
        path = Path(self.config['tree']) / 'hdrvdp.m'
        path.write_text('changed\n')
        with self.assertRaises(ValueError):
            script(self.config)
        self.config['bindings'][str(path)] = sha(path)
        (path.parent / 'unexpected.m').write_text('unbound\n')
        with self.assertRaises(ValueError):
            script(self.config)

    def test_aic_wrong_phase_and_wrong_geometry_refuse(self):
        for overrides in (dict(metric='hdrvdp-aic2026'), dict(metric='hdrvdp3-aic2026'),
                          dict(phase='first', samples=2), dict(phase='unknown'), dict(width=4)):
            config = copy.deepcopy(self.config)
            config.update(overrides)
            with self.assertRaises(ValueError):
                script(config)


if __name__ == '__main__':
    unittest.main()
