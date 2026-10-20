"""Protocol, resume and corruption controls; native scoring has a separate real panel."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import signal
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('aic_rescore', HERE / 'aic_rescore.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

FAKE = '''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
if sys.argv[1]=='list-metrics':
 print('name backend requires_gpu\\nhdrvdp-aic2026 CPU no\\nhdrvdp3-aic2026 CPU no\\npsnr CPU no\\nws-hv CPU no'); sys.exit(0)
a=sys.argv; routes=[a[i+1] for i,v in enumerate(a) if v=='--metric']
log=Path(os.environ['AIC_FAKE_LOG']); prior=log.read_text().splitlines() if log.exists() else []
with log.open('a') as f:f.write(json.dumps(routes)+'\\n')
if os.environ.get('AIC_FAKE_SLEEP'):
 import time; time.sleep(60)
scores={}
for r in routes:
 key={'hdrvdp-aic2026':'hdrvdp_aic2026_q','hdrvdp3-aic2026':'hdrvdp3_aic2026_q','ws-hv':'ws_hv','psnr':'psnr'}[r]
 scores[key]=None if r=='psnr' and not prior and os.environ.get('AIC_FAKE_FAIL') else 0.12345678901234566
print(json.dumps(dict(metrics=routes,results=[dict(reference=a[a.index('--reference')+1],variant=a[a.index('--variant')+1],scores=scores)])))
if os.environ.get('AIC_FAKE_PANIC'):
 print('thread panicked at runtime failure',file=sys.stderr)
sys.exit(101 if os.environ.get('AIC_FAKE_EXIT') else 1 if any(v is None for v in scores.values()) else 0)
'''


class Rescore(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ.get('TMPDIR', Path.home() / 'tmp'))
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch, prefix='aic-rescore-test-')
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        for name in ['ref.png', 'a.png', 'b.png']:
            (self.images / name).write_bytes(name.encode())
        self.binary = self.root / 'zenmetrics'
        self.binary.write_text(FAKE)
        self.binary.chmod(0o755)
        self.csv = self.root / 'aic.csv'
        self.csv.write_text('source,distorted,HDR_VDP_2,HDR_VDP_3,PSNR,codec\nref.png,a.png,99,99,99,AVIF\nref.png,b.png,99,99,99,JXL\n')
        self.plan = self.root / 'plan.json'
        self.output = self.root / 'output'
        self.env = dict(os.environ, AIC_FAKE_LOG=str(self.root / 'calls'))
        self.call('prepare', '--aic-csv', self.csv, '--images', self.images,
                  '--output', self.plan, '--extra-metric', 'ws-hv', expected=0)

    def tearDown(self):
        self.temp.cleanup()

    def call(self, *args, expected):
        p = subprocess.run([sys.executable, str(HERE / 'aic_rescore.py'), *map(str, args)],
                           capture_output=True, text=True, env=self.env)
        self.assertEqual(p.returncode, expected, p.stdout + p.stderr)
        return p

    def run_panel(self, *args, expected):
        return self.call('run', '--plan', self.plan, '--binary', self.binary,
                         '--source-commit', 'fixture-source', '--output', self.output,
                         *args, expected=expected)

    def calls(self):
        return [json.loads(l) for l in (self.root / 'calls').read_text().splitlines()]

    def test_pair_resume_and_exact_csv(self):
        self.run_panel('--limit-pairs', '1', expected=2)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertFalse(summary['complete'])
        self.assertEqual(len(self.calls()), 1)
        self.assertTrue(self.calls()[0][0].startswith('hdrvdp'))
        self.run_panel(expected=0)
        self.assertEqual(len(self.calls()), 2)
        self.run_panel(expected=0)
        self.assertEqual(len(self.calls()), 2)
        with (self.output / 'scores.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertEqual(float(r['HDR_VDP_2']), 0.12345678901234566)
            self.assertEqual(float(r['HDR_VDP_3']), 0.12345678901234566)
            self.assertEqual(float(r['WS-HV']), 0.12345678901234566)
            self.assertNotEqual(r['PSNR'], '99')
        self.assertEqual(rows[0]['codec'], 'AVIF')

    def test_failed_metric_retries_without_repeating_successes(self):
        self.env['AIC_FAKE_FAIL'] = '1'
        self.run_panel(expected=2)
        self.run_panel(expected=0)
        self.assertEqual(self.calls()[-1], ['psnr'])
        self.assertEqual(len(self.calls()), 3)
        self.assertEqual(len(list((self.output / 'attempts').iterdir())), 3)

    def test_binary_input_and_plan_drift_refuse(self):
        self.run_panel(expected=0)
        original = self.binary.read_text()
        self.binary.write_text(original + '\n# changed\n')
        self.assertIn('identity changed', self.run_panel(expected=1).stderr)
        self.binary.write_text(original)
        (self.images / 'a.png').write_bytes(b'changed')
        self.assertIn('input hash mismatch', self.run_panel(expected=1).stderr)
        (self.images / 'a.png').write_bytes(b'a.png')
        plan = json.loads(self.plan.read_text()); plan['pairs'][0]['codec'] = 'changed'
        self.plan.write_text(json.dumps(plan))
        self.assertIn('identity changed', self.run_panel(expected=1).stderr)
        self.assertEqual(len(self.calls()), 2)

    def test_checkpoint_corruption_refuses(self):
        self.run_panel(expected=0)
        cell = next((self.output / 'cells').iterdir())
        obj = json.loads(cell.read_text()); obj['payload']['PSNR']['value'] = 42
        cell.write_text(json.dumps(obj))
        self.assertIn('corrupt checkpoint', self.run_panel(expected=1).stderr)
        self.assertEqual(len(self.calls()), 2)

    def test_nonfinite_and_ambiguous_columns_refuse(self):
        for scores in [{'x': float('nan')}, {'x': None}, {'x': True}, {'x1': 1, 'x2': 2}]:
            with self.assertRaises(ValueError):
                mod.select(scores, 'x*')

    def test_overlapping_writer_refused(self):
        self.run_panel(expected=0)
        with mod.writer_lock(self.output):
            self.run_panel(expected=1)
        self.assertEqual(len(self.calls()), 2)

    def test_missing_route_remains_explicit(self):
        obj = json.loads(self.plan.read_text()); obj['columns']['PSNR']['route'] = 'not-built'
        self.plan.write_text(json.dumps(obj))
        self.run_panel(expected=2)
        with (self.output / 'status.csv').open() as stream:
            states = list(csv.DictReader(stream))
        self.assertEqual([r['status'] for r in states if r['column'] == 'PSNR'], ['unavailable'] * 2)

    def test_path_escape_rejected(self):
        self.csv.write_text('source,distorted,PSNR\n../aic.csv,a.png,1\n')
        self.call('prepare', '--aic-csv', self.csv, '--images', self.images,
                  '--output', self.root / 'escape.json', expected=1)

    def test_relocated_identical_inputs_resume(self):
        self.run_panel(expected=0)
        self.images.rename(self.root / 'new-images')
        self.run_panel('--images', self.root / 'new-images', expected=0)
        self.assertEqual(len(self.calls()), 2)

    def test_subset_schedule_does_not_change_campaign(self):
        self.run_panel('--only', 'HDR_VDP_2,HDR_VDP_3', expected=2)
        self.assertTrue(all(all(r.startswith('hdrvdp') for r in routes) for routes in self.calls()))
        self.run_panel(expected=0)
        self.assertTrue(all(not any(r.startswith('hdrvdp') for r in routes) for routes in self.calls()[2:]))

    def test_model_asset_drift_refused(self):
        models = self.root / 'models'; models.mkdir()
        (models / 'weights').write_bytes(b'first')
        self.run_panel('--models', models, expected=0)
        (models / 'weights').write_bytes(b'second')
        self.assertIn('identity changed', self.run_panel('--models', models, expected=1).stderr)

    def test_explicit_jnd_fit_and_missing_fit(self):
        self.csv.write_text('source,distorted,PSNR,JND_PSNR\nref.png,a.png,99,99\n')
        fit = self.root / 'fits.json'
        fit.write_text(json.dumps({'mappings': {'PSNR': {'a': 2, 'b': 1, 'c': 2, 'sign': -1}}}))
        plan = self.root / 'fit-plan.json'
        self.call('prepare', '--aic-csv', self.csv, '--images', self.images,
                  '--output', plan, '--jnd-config', fit, expected=0)
        self.plan = plan
        self.run_panel(expected=0)
        with (self.output / 'scores.csv').open() as f:
            row = next(csv.DictReader(f))
        self.assertEqual(float(row['JND_PSNR']), 2 * (0.12345678901234566 - 1) ** 2)
        self.assertNotEqual(row['JND_PSNR'], '99')

    def test_live_unfinished_child_refused(self):
        self.run_panel(expected=0)
        attempt = self.output / 'attempts' / 'unfinished'; attempt.mkdir()
        (attempt / 'child.json').write_text(json.dumps({'pid': os.getpid()}))
        self.assertIn('unfinished child', self.run_panel(expected=1).stderr)
        self.assertEqual(len(self.calls()), 2)

    def test_real_interruption_cleans_child_and_resumes(self):
        self.env['AIC_FAKE_SLEEP'] = '1'
        command = [sys.executable, str(HERE / 'aic_rescore.py'), 'run', '--plan', str(self.plan),
                   '--binary', str(self.binary), '--source-commit', 'fixture-source', '--output', str(self.output)]
        child = subprocess.Popen(command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.monotonic() + 15
        records = []
        while time.monotonic() < deadline:
            records = list((self.output / 'attempts').glob('*/child.json'))
            if records:
                break
            time.sleep(0.02)
        self.assertTrue(records, 'real scoring child did not start')
        pid = json.loads(records[0].read_text())['pid']
        child.send_signal(signal.SIGTERM)
        stdout, stderr = child.communicate(timeout=10)
        self.assertEqual(child.returncode, 130, stdout + stderr)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertEqual(json.loads((records[0].parent / 'exit.json').read_text())['interrupted_by'], 'KeyboardInterrupt')
        self.env.pop('AIC_FAKE_SLEEP')
        self.run_panel(expected=0)
        self.assertEqual(len(self.calls()), 3)

    def test_standard_aic_directories_resolve_and_ambiguity_refuses(self):
        (self.images / 'sources').mkdir(); (self.images / 'distorted').mkdir()
        (self.images / 'ref.png').rename(self.images / 'sources' / 'ref.png')
        for name in ['a.png', 'b.png']:
            (self.images / name).rename(self.images / 'distorted' / name)
        self.run_panel(expected=0)
        (self.images / 'ref.png').write_bytes(b'ref.png')
        self.assertIn('exactly one image', self.run_panel(expected=1).stderr)

    def test_combined_csv_requires_explicit_view(self):
        self.csv.write_text('view,source,distorted,PSNR\ncropped,ref.png,a.png,1\nfullres,ref.png,a.png,2\n')
        newplan = self.root / 'combined-plan.json'
        self.call('prepare', '--aic-csv', self.csv, '--images', self.images,
                  '--output', newplan, expected=1)
        self.call('prepare', '--aic-csv', self.csv, '--images', self.images,
                  '--output', newplan, '--view', 'cropped', expected=0)
        self.assertEqual(len(json.loads(newplan.read_text())['pairs']), 1)

    def test_abnormal_exit_and_device_panic_never_accept_valid_json(self):
        for key in ['AIC_FAKE_EXIT', 'AIC_FAKE_PANIC']:
            self.env[key] = '1'
            self.run_panel(expected=2)
            summary = json.loads((self.output / 'summary.json').read_text())
            self.assertNotIn('scored', summary['counts'])
            self.env.pop(key)
        self.assertFalse(json.loads((self.output / 'summary.json').read_text())['complete'])

    def test_unowned_existing_scores_are_preserved(self):
        self.output.mkdir()
        original = self.output / 'scores.csv'
        original.write_text('precious input CSV\n')
        self.assertIn('unowned', self.run_panel(expected=1).stderr)
        self.assertEqual(original.read_text(), 'precious input CSV\n')
        self.assertFalse((self.root / 'calls').exists())

    def test_atomic_collision_preserves_foreign_staging_file(self):
        path = self.root / 'artifact'
        foreign = self.root / 'artifact.collision.part'
        foreign.write_text('existing artifact')
        with mock.patch.object(mod.uuid, 'uuid4', return_value=type('ID', (), {'hex': 'collision'})()):
            with self.assertRaises(FileExistsError):
                mod.atomic(path, 'new value')
        self.assertEqual(foreign.read_text(), 'existing artifact')


if __name__ == '__main__':
    unittest.main()
