#!/usr/bin/env python3
"""Controls for retained scalar scheduling, gates and frozen attempt budgets."""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import scalar_priority as priority


def record(status='failed', warm=False, clean=True):
    row = dict(timing_phase='warm' if warm else 'cold', n=10,
               zenbench_samples=[dict(gate_clean=clean)] * 10)
    return dict(record=dict(status=status, row=row))


class ScalarPriorityTests(unittest.TestCase):
    def test_dimension_disable_ignores_labels_and_orientation(self):
        policy = dict(disabled_geometries=[[7680,4320]])
        self.assertTrue(priority.disabled_geometry(dict(width=7680,height=4320,size_name="other"),policy))
        self.assertTrue(priority.disabled_geometry(dict(width=4320,height=7680,size_name="portrait"),policy))
        self.assertFalse(priority.disabled_geometry(dict(width=3840,height=2160,size_name="8k"),policy))

    def test_attempt_budget_never_resets(self):
        self.assertEqual(priority.decision([]), ('pending', 2))
        self.assertEqual(priority.decision([record()]), ('pending', 1))
        self.assertEqual(priority.decision([record(), record()]), ('exhausted', 0))

    def test_flagged_warm_not_reused(self):
        self.assertEqual(priority.decision([record('ok', warm=True, clean=False)]), ('pending', 1))
        self.assertEqual(priority.decision([record('ok', warm=True)]), ('retained', 0))
        self.assertEqual(priority.decision([record('unsupported')]), ('retained', 0))

    def test_family_identity(self):
        for name in ['haarpsi', 'haarpsi-y', 'ssim2', 'dssim', 'butteraugli', 'butteraugli-original', 'cvvdp', 'vmaf-neg', 'vmaf-v1']:
            self.assertTrue(priority.requested_family(name), name)
        for name in ['ssim', 'msssim', 'ssim-pyiqa-f32-k6', 'psnr-y-libvmaf']:
            self.assertFalse(priority.requested_family(name), name)

    def test_prepare_preserves_maps_and_orders_scalar_tiers(self):
        import json
        cells = []
        for name in ['psnr', 'dvifmish-talk-faithful-luma-int', 'ssim2', 'slow']:
            for mode in ['score', 'score+map']:
                cells.append(dict(metric_name=name, only_mode=mode, timing_phase='cold', cold_repeat=0,
                                  width=10, height=10, size_name='small', threads=1, reference='r', distorted='d'))
        configs = [dict(c, durable_cell_id=str(i)) for i, c in enumerate(cells)]
        evidence = {'0': [dict(record=dict(status='ok', row=dict(timing_phase='cold')), path='old', sha256='hash')]}
        manifest = dict(cells=cells, input_sha256=dict(r='same', d='same'))
        frozen = dict(manifest=manifest)
        captured = []
        def fake_freeze(source, target, selected, lineage):
            captured.append((target.name, selected, lineage))
            return dict(bundle=str(target), manifest_sha256='test', cells=len(selected))
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            (source / 'manifest.json').write_text(json.dumps(manifest))
            config = dict(sources=[dict(bundle=str(source), output='old')], results=str(root/'results'), lock='lock')
            with patch.object(priority, 'retained', return_value=(None, frozen, configs, evidence)), patch.object(priority, 'freeze', side_effect=fake_freeze):
                priority.prepare(config, root, [dict(metric='psnr')])
            self.assertEqual([x[0].split('-source')[0] for x in captured], ['dvifmish', 'requested-family', 'remaining-scalar'])
            self.assertTrue(all(c['only_mode']=='score' for _, selected, _ in captured for c in selected))
            self.assertNotIn('psnr', [c['metric_name'] for _, selected, _ in captured for c in selected])
            self.assertEqual(len(json.loads((root/'retained-coverage.json').read_text())), 8)


    def test_nn_successor_retains_caps_and_disables_dimensions(self):
        import json
        cells = [dict(metric_name='nn', only_mode=mode, width=w, height=h,
                      timing_phase='cold', cold_repeat=0, threads=8, size_name='misleading')
                 for mode,w,h in [('score',7680,4320),('score',4320,7680),
                                  ('score',3840,2160),('score+map',3840,2160)]]
        configs = [dict(c,durable_cell_id=str(i)) for i,c in enumerate(cells)]
        frozen = dict(manifest=dict(cells=cells))
        evidence = {'2':[dict(record=dict(status='failed'),path='old',sha256='hash')]}
        captured=[]
        def freeze(source,target,selected,lineage):
            captured.extend(selected)
            return dict(bundle=str(target),manifest_sha256='hash',cells=len(selected))
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory)
            config=dict(sources=[],nn=[dict(name='nn-cpu',bundle='old',output='old',max_attempts=2)],
                        disabled_geometries=[[7680,4320]],results=str(root/'results'),lock='lock')
            with patch.object(priority,'retained',return_value=(None,frozen,configs,evidence)), patch.object(priority,'freeze',side_effect=freeze):
                plan=priority.prepare(config,root,[])
            self.assertEqual(captured,[cells[2]])
            self.assertEqual(plan['phases'][0]['max_attempts'],1)
            coverage=json.loads((root/'retained-coverage.json').read_text())
            self.assertEqual([c['status'] for c in coverage],['disabled-by-user','disabled-by-user','pending','maps-not-scheduled'])

    def test_nn_replacement_binds_pixels_and_preserves_original_lineage(self):
        import json
        from durable import FINITE_GPU_POLICY
        cell = dict(metric_name='stlpips-alex', only_mode='score', width=2048, height=1080,
                    timing_phase='warm', cold_repeat=0, threads=8, device='metal', reference='r', distorted='d',
                    worker_environment=dict(WGPU_BACKEND='metal', ZENMETRICS_NN_ENGINE='ort-webgpu'))
        manifest = dict(cells=[cell], input_sha256=dict(r='rhash', d='dhash'))
        frozen = dict(manifest=manifest)
        evidence = {'old-id':[dict(record=dict(status='failed'), path='old', sha256='hash')]}
        captured = []
        def freeze(source, target, selected, lineage):
            captured.append((source, selected, lineage))
            return dict(bundle=str(target), manifest_sha256='hash', cells=len(selected))
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            replacement = root/'replacement'
            replacement.mkdir()
            (replacement/'manifest.json').write_text(json.dumps(manifest))
            config = dict(sources=[], nn=[dict(name='metal',bundle='old',output='old',max_attempts=2,
                          execution_bundle=str(replacement), gpu_score_repeatability_policy=FINITE_GPU_POLICY)],
                          disabled_geometries=[[7680,4320]],results=str(root/'results'),lock='lock')
            with patch.object(priority,'retained',return_value=(None,frozen,[dict(cell,durable_cell_id='old-id')],evidence)), patch.object(priority,'freeze',side_effect=freeze):
                plan = priority.prepare(config,root,[])
            self.assertEqual(plan['phases'][0]['max_attempts'],1)
            self.assertEqual(captured[0][1][0]['gpu_score_repeatability_policy'],FINITE_GPU_POLICY)
            self.assertNotIn('gpu_score_repeatability_policy',captured[0][2]['cells'][0]['original_config'])
            manifest['input_sha256']['r'] = 'different'
            (replacement/'manifest.json').write_text(json.dumps(manifest))
            with patch.object(priority,'retained',return_value=(None,dict(manifest=dict(cells=[cell],input_sha256=dict(r='rhash',d='dhash'))),[dict(cell,durable_cell_id='old-id')],evidence)), patch.object(priority,'freeze',side_effect=freeze):
                with self.assertRaisesRegex(ValueError,'input differs'):
                    priority.prepare(config,root,[])

    def test_execute_refuses_any_upcoming_map(self):
        import json
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            root = Path(directory)
            bundle = root / 'bundle'
            bundle.mkdir()
            (bundle / 'manifest.json').write_text(json.dumps(dict(cells=[dict(only_mode='score+map')])))
            plan = dict(scheduler_files={}, source_sha256=priority.sha(Path(priority.__file__)),
                        phases=[dict(bundle=str(bundle), output=str(root/'output'), name='map',
                                     manifest_sha256=priority.sha(bundle/'manifest.json'))])
            path = root / 'sequence.json'
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'upcoming map'):
                priority.execute(path, priority.sha(path))
            self.assertFalse((root/'output').exists())

    def test_execute_refuses_changed_scheduler(self):
        import json
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            root = Path(directory)
            helper = root / 'helper.py'
            helper.write_text('changed')
            path = root / 'sequence.json'
            path.write_text(json.dumps(dict(scheduler_files={'helper.py': 'wrong'}, phases=[])))
            with self.assertRaisesRegex(ValueError, 'frozen scheduler changed'):
                priority.execute(path, priority.sha(path))


    def test_boundary_accepts_confirmed_absent_transient_unit(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            receipt = Path(directory) / 'boundary.json'
            result = subprocess.CompletedProcess([], 0, 'LoadState=not-found\nActiveState=inactive\n', '')
            with patch.object(priority, 'processes', return_value=[]), patch.object(priority.sys, 'platform', 'linux'), patch.object(priority.subprocess, 'run', return_value=result) as run:
                priority.boundary(dict(sources=[], nn=[], lock='shared', stop_services=['gone.service']), receipt)
            self.assertEqual(run.call_count, 1)
            self.assertTrue(receipt.exists())

    def test_checkin_tracks_replacement_waits_without_false_completion(self):
        from mac_service import _check_in_rows
        value = dict(service='scalar-service', phases=[dict(output='fast', progress=dict(complete=True), wait={}),
                   dict(output='nn', progress='queued', wait=dict(pending=['remaining-scalar']))])
        rows = _check_in_rows(value)
        self.assertEqual(rows[1]['waiting_on'], ['remaining-scalar'])
        self.assertFalse(all(r['progress'].get('complete') for r in rows))
        self.assertEqual(_check_in_rows(rows), rows)


    def test_new_requested_routes_and_slow_tail_precedence(self):
        for name in ['vmaf-neg', 'vmaf-v1', 'psnrhvs-daala', 'fsim', 'fsim-y', 'fsim-k2', 'fsim-nodownscale']:
            self.assertTrue(priority.requested_family(name))
            self.assertEqual(priority.queue_tier(name, []), 2)
        for name in ['hdrvdp', 'hdrvdp3', 'mdctpsnr']:
            self.assertEqual(priority.queue_tier(name, [name]), 5)
        self.assertLess(priority.priority_order('vmaf-neg'), priority.priority_order('fsim-y'))

    def test_successor_one_attempt_budget_is_not_reset(self):
        self.assertEqual(priority.decision([], cap=1), ('pending', 1))
        self.assertEqual(priority.decision([record()], cap=1), ('exhausted', 0))

    def test_deadline_selects_smallest_missing_pair_without_repeat(self):
        entries = []
        for size in [20, 40]:
            for phase, repeat in [('cold', 0), ('cold', 1), ('cold', 2), ('warm', 0)]:
                c = dict(metric_name='fsim-y', width=size, height=size, size_name=str(size), threads=8,
                         timing_phase=phase, cold_repeat=repeat, only_mode='score', required_samples=10)
                entries.append(dict(source=0, original_cell_id=f'{size}-{phase}-{repeat}', original_config=c,
                                    status='pending', queue='requested-first'))
        qualified = {('fsim-y', 20, 20, '20', 8, p) for p in ['cold', 'warm']}
        qualified.add(('fsim-y', 40, 40, '40', 8, 'cold'))
        selected, choices = priority.deadline_selection(entries, qualified)
        self.assertEqual(selected, {(0, '40-warm-0')})
        self.assertEqual(choices[0]['geometry'], (40, 40, '40'))
        self.assertTrue(all(e['original_config']['required_samples'] == 10 for e in entries))

    def test_successor_lineage_preserves_parent_bytes(self):
        import json
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            parent = source / 'scalar-lineage.json'
            parent.write_text(json.dumps(dict(prior_attempts=1)))
            parent_hash = priority.sha(parent)
            cell = dict(metric_name='vmaf-v1', size_name='small')
            manifest = dict(cells=[cell], metadata={}, input_sha256={'scalar-lineage.json': parent_hash})
            (source/'manifest.json').write_text(json.dumps(manifest))
            (source/'SHA256SUMS.json').write_text(json.dumps({'manifest.json': priority.sha(source/'manifest.json'), 'scalar-lineage.json': parent_hash}))
            target = root / 'successor'
            priority.freeze(source, target, [cell], dict(source='parent', remaining_attempts=1))
            self.assertEqual(priority.sha(parent), parent_hash)
            saved = priority.load(target/'manifest.json')
            ancestor = 'scalar-parent-lineage-' + parent_hash + '.json'
            self.assertEqual(saved['input_sha256'][ancestor], parent_hash)
            self.assertNotEqual(saved['input_sha256']['scalar-lineage.json'], parent_hash)
            self.assertEqual(priority.load(target/'scalar-lineage.json')['remaining_attempts'], 1)


if __name__ == '__main__':
    unittest.main()
