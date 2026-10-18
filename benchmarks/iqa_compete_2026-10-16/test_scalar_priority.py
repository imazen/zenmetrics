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
                                  width=10, height=10, threads=1, reference='r', distorted='d'))
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


if __name__ == '__main__':
    unittest.main()
