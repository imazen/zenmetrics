"""Study identity, missing coverage and qualification refusal controls."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import build_benchmark_study as study


class StudyTests(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / 'tmp'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='metric-study-', dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ['r', 'd']:
            (self.root / name).write_bytes(bytes(range(12)))
        self.spec = {'schema': 1, 'study': 'fixture', 'metrics': ['ssim2', 'iwssim'],
                     'pairs': [{'id': 'pair', 'origin': 'origin', 'content': 'textured',
                                'width': 2, 'height': 2, 'stride_bytes': 6, 'pixel_format': 'rgb8_srgb',
                                'reference': self.bound('r'), 'distorted': self.bound('d')}],
                     'testbeds': [{'id': 'cpu', 'machine': 'cpu', 'domain': 'uniform',
                                  'target': 'x86_64-unknown-linux-gnu'}],
                     'reuse': ['cold', 'warm_reference'], 'threads': [1, 2],
                     'passes': ['wall', 'driver'], 'replicates': 3, 'operations': 2,
                     'timeout_seconds': 10, 'admission_seconds': 0, 'peak_mem_bytes': 1024**3}

    def bound(self, name):
        return {'path': name, 'sha256': study.file_sha(self.root / name)}

    def write(self):
        path = self.root / 'spec.json'
        path.write_text(json.dumps(self.spec))
        return path

    def qualify(self):
        (self.root / 'binary').write_bytes(b'fixture executable identity; not a real build')
        build = {'schema': 1, 'target': self.spec['testbeds'][0]['target'],
                 'binary_sha256': study.file_sha(self.root / 'binary'), 'source_revision': 'fixture',
                 'dependency_lock_sha256': 'a' * 64, 'compiler': 'fixture', 'features': [],
                 'rustflags': [], 'profile': 'release', 'instrumented': False}
        (self.root / 'build.json').write_text(json.dumps(build))
        bed = self.spec['testbeds'][0]
        bed['artifacts'] = {'latency': {'binary': self.bound('binary'), 'build': self.bound('build.json')}}
        inv, _, rows = study.expand(self.write())
        cases = {}
        for row in rows:
            if row['status'] == 'unsupported' or row['measurement'] == 'driver':
                continue
            proof = study.digest({'case': row['case'], 'inputs': [self.spec['pairs'][0][s]['sha256'] for s in ['reference', 'distorted']],
                                  'binary_sha256': build['binary_sha256'], 'operations': self.spec['operations']})
            cases[proof] = {'score_bits': [123, 123]}
        q = {'schema': 1, 'target': bed['target'], 'native_arch_verified': True,
             'inventory_sha': study.digest(inv), 'cases': cases}
        (self.root / 'qualification.json').write_text(json.dumps(q))
        bed['qualification'] = self.bound('qualification.json')

    def test_source_inventory_and_exact_expansion(self):
        inv, summary, rows = study.expand(self.write())
        # Every `#[value(name)]` in the CLI's MetricKind; bump with each metric added
        # (51: +vmaf-studio601, vmaf-neg-studio601, psnr-ycbcr611, ciede2000, ssimc;
        # 53: +psnr-cb, psnr-cr;
        # 60: +lpips-alex, lpips-vgg, lpips-squeeze, stlpips-alex, stlpips-vgg, dists, deepdc;
        # 65: +ckdn, afine, dreamsim, piqe, niqe).
        self.assertEqual(len(inv['metrics']), 65)
        self.assertEqual(summary['declared_cells'], 48)
        self.assertEqual(summary['statuses'], {'ready': 0, 'unsupported': 24, 'unqualified': 24})
        self.assertEqual(len({r['cell_sha'] for r in rows}), 48)
        self.assertEqual(summary['planned_driver_launches'], 0)

    def test_qualified_artifacts_only_and_immutable_outputs(self):
        self.qualify()
        _, summary, rows = study.expand(self.write())
        self.assertEqual(summary['statuses']['ready'], 12)
        output = self.root / 'output'
        study.write_study(self.write(), output)
        bundles = list(output.glob('*/bundle-spec.json'))
        self.assertEqual(len(bundles), 12)
        bundle = json.loads(bundles[0].read_bytes())
        self.assertEqual(bundle['plan']['expected_correctness'], {'score_bits': [123, 123]})
        self.assertEqual(bundle['plan']['operations'], 2)
        with self.assertRaises(FileExistsError):
            study.write_study(self.write(), output)

    def test_relocation_preserves_identity_and_changes_are_bound(self):
        self.qualify()
        first = study.expand(self.write())[1]['study_sha']
        other = self.root / 'relocated'; other.mkdir()
        for path in self.root.iterdir():
            if path.is_file(): shutil.copyfile(path, other / path.name)
        self.assertEqual(study.expand(other / 'spec.json')[1]['study_sha'], first)
        self.spec['threads'] = [1]
        self.assertNotEqual(study.expand(self.write())[1]['study_sha'], first)

    def test_input_mutation_and_invalid_schema_refused(self):
        (self.root / 'r').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            study.expand(self.write())
        (self.root / 'r').write_bytes(bytes(range(12)))
        self.spec['metrics'].append('unknown')
        with self.assertRaisesRegex(ValueError, 'unknown CLI'):
            study.expand(self.write())

    def test_no_empty_or_duplicate_axes_and_ids(self):
        original = copy.deepcopy(self.spec)
        for mutate in [lambda s: s.update(threads=[1, 1]), lambda s: s.update(operations=True),
                       lambda s: s['pairs'].append(dict(s['pairs'][0], content='different')),
                       lambda s: s.update(replicates=1001), lambda s: s.update(extra='unknown')]:
            self.spec = copy.deepcopy(original); mutate(self.spec)
            with self.assertRaises(ValueError): study.expand(self.write())

    def test_qualification_cannot_be_reused_for_wrong_native_or_build(self):
        self.qualify()
        q = json.loads((self.root / 'qualification.json').read_bytes())
        q['native_arch_verified'] = False
        (self.root / 'qualification.json').write_text(json.dumps(q))
        self.spec['testbeds'][0]['qualification'] = self.bound('qualification.json')
        with self.assertRaisesRegex(ValueError, 'native execution'):
            study.expand(self.write())

    def test_packaging_refuses_changed_cases_and_runner_before_writing(self):
        self.qualify()
        spec = self.write()
        output = self.root / 'output'
        study.write_study(spec, output)
        packaged = self.root / 'packaged'
        runner = self.root / 'binary'
        with self.assertRaisesRegex(ValueError, 'runner hash'):
            study.package_study(spec, output, runner, 'a' * 64, packaged)
        case = next(output.glob('*/case.json'))
        value = json.loads(case.read_bytes()); value['threads'] = 17
        case.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'case or bundle plan'):
            study.package_study(spec, output, runner, study.file_sha(runner), packaged)
        self.assertFalse(packaged.exists())

    def test_hdr_requires_nits_and_explicit_matching_viewing(self):
        self.spec['metrics'] = ['hdrvdp']
        self.assertTrue(all(row['status'] == 'unsupported' for row in study.expand(self.write())[2]))
        pair = self.spec['pairs'][0]
        pair.update(stride_bytes=24, pixel_format='rgb_f32le_bt709_linear_nits',
                    viewing={'model': 'hdrvdp2-defaults-bt709', 'pixels_per_degree': 45.0})
        for name in ['r', 'd']:
            (self.root / name).write_bytes(bytes(48))
        pair.update(reference=self.bound('r'), distorted=self.bound('d'))
        rows = study.expand(self.write())[2]
        cold = [row for row in rows if row['case']['reuse'] == 'cold']
        self.assertTrue(all(row['status'] == 'unqualified' and row['case']['schema'] == 3 for row in cold))
        self.assertTrue(all(row['status'] == 'unsupported' for row in rows if row not in cold))
        pair['viewing']['pixels_per_degree'] = True
        with self.assertRaises(ValueError): study.expand(self.write())

    def test_gpu_requires_explicit_backend_build_and_verified_device(self):
        self.spec['metrics'] = ['ssim2-gpu']
        self.spec['reuse'] = ['cold', 'warm_instance']
        self.assertTrue(all(row['status'] == 'unsupported' for row in study.expand(self.write())[2]))
        bed=self.spec['testbeds'][0]
        bed.update(gpu_backend='wgpu',gpu_device='fixture-device',gpu_adapter='fixture GPU')
        self.assertTrue(all(row['status'] == 'unqualified' for row in study.expand(self.write())[2]))
        self.qualify()
        self.assertTrue(all(row['status'] == 'unsupported' for row in study.expand(self.write())[2] if row['measurement']=='wall'))
        build=json.loads((self.root/'build.json').read_bytes());build['features']=['study-gpu-wgpu']
        (self.root/'build.json').write_text(json.dumps(build))
        bed['artifacts']['latency']['build']=self.bound('build.json')
        with self.assertRaisesRegex(ValueError,'GPU qualification'):
            study.expand(self.write())
        q=json.loads((self.root/'qualification.json').read_bytes())
        q.update(testbed='cpu',gpu_backend='wgpu',gpu_device='fixture-device',gpu_device_verified=True)
        (self.root/'qualification.json').write_text(json.dumps(q));bed['qualification']=self.bound('qualification.json')
        rows=study.expand(self.write())[2]
        self.assertTrue(all(row['case']['schema']==4 for row in rows))
        self.assertTrue(all(row['status']=='unqualified' for row in rows))

    def test_nonlinux_heaptrack_is_visible(self):
        self.spec['metrics'] = ['ssim2']; self.spec['passes'] = ['heaptrack']
        self.spec['testbeds'][0]['target'] = 'aarch64-apple-darwin'
        _, summary, rows = study.expand(self.write())
        self.assertEqual(summary['statuses']['unsupported'], summary['declared_cells'])
        self.assertTrue(all('heaptrack interception backend requires Linux' in r['reasons'] for r in rows))


if __name__ == '__main__':
    unittest.main()
