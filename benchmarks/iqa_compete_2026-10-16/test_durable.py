#!/usr/bin/env python3
"""OS-backed crash, deadline, identity and recovery controls; no IQA workload."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import durable

WORKER = r'''#!/usr/bin/env python3
import json,os,signal,struct,sys,time,subprocess
from pathlib import Path
c=json.load(open(sys.argv[1])); behavior=c.get('behavior','ok')
p=Path(sys.argv[1]).parent
with open(Path(c['reference']).parent/'execution.jsonl','a') as f:
 f.write(json.dumps(dict(metric=c['metric_name'],phase=c['timing_phase'],pid=os.getpid()))+'\n');f.flush();os.fsync(f.fileno())
(p/'worker.pid').write_text(str(os.getpid()))
if behavior=='stall':
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 kid=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])
 (p/'grandchild.pid').write_text(str(kid.pid))
 while True: print('still alive',flush=True);time.sleep(.02)
if behavior=='wait':time.sleep(.6)
if behavior=='killed':os.kill(os.getpid(),signal.SIGKILL)
if behavior=='nonzero':print('failure diagnostics',flush=True);sys.exit(7)
r=dict(c,mode=c['only_mode'],status='ok',n=1 if c['timing_phase']=='cold' else 10,
 median_ns=100,input_load_ns=5,score=1.,score_bits=struct.pack('>d',1.).hex())
if c['timing_phase']=='cold':r['cold_elapsed_ns']=100
else:r['round_means_ns']=[100]*10
if behavior=='unsupported':
 r.update(status='unsupported',n=0);r.pop('median_ns');r.pop('cold_elapsed_ns',None)
if behavior=='failed':
 r.update(status='failed',n=0);r.pop('median_ns');r.pop('cold_elapsed_ns',None)
if behavior=='partial':r.update(status='partial',n=2,round_means_ns=[100,100])
if behavior=='wrong':r['metric_name']='wrong'
if behavior=='missing_rounds':r.pop('round_means_ns',None)
if behavior=='bad_bits':r['score_bits']='7ff0000000000000'
if behavior=='bad_score':r['score']=2.
if behavior=='no_row':sys.exit(0)
if behavior=='partial_write':sys.stdout.write('IQA_ROW {"status":');sys.stdout.flush();sys.exit(0)
print('IQA_ROW '+json.dumps(r),flush=True)
if behavior=='duplicate':print('IQA_ROW '+json.dumps(r),flush=True)
'''


class DurableTests(unittest.TestCase):
    def test_runner_hash_is_bound_to_executed_controller(self):
        value = self.freeze('ok', phase='cold')
        runner = self.root / 'durable.py'
        runner.write_bytes(Path(durable.SELF).read_bytes())
        value['metadata']['runner_sha256'] = {'durable.py': durable.file_hash(runner)}
        self.path.write_text(json.dumps(value))
        durable.manifest(self.path)
        runner.write_text('changed controller')
        with self.assertRaisesRegex(durable.Refusal, 'runner content changed'):
            durable.manifest(self.path)
        value['metadata']['runner_sha256']['durable.py'] = durable.file_hash(runner)
        self.path.write_text(json.dumps(value))
        with self.assertRaisesRegex(durable.Refusal, 'executed controller'):
            durable.manifest(self.path)

    def test_worker_identity_is_retained_across_exec(self):
        self.freeze('ok', phase='cold')
        self.finish(self.start())
        attempt = self.running_attempt()
        proof = json.loads((attempt / 'worker-process.json').read_text())
        self.assertEqual(proof['pid'], int((attempt / 'worker.pid').read_text()))
        self.assertEqual(proof['pgid'], json.loads((attempt / 'groups.json').read_text())['payload'])
        self.assertTrue(self.no_live(proof['pid']))
        self.assertGreater(proof['unix_s'], 0)

    def test_model_environment_is_frozen_contained_and_rehashed(self):
        value = self.freeze('ok', phase='cold')
        models = self.root / 'models'
        models.mkdir()
        asset = models / 'fixture.onnx'
        asset.write_bytes(b'frozen graph')
        value['input_sha256']['models/fixture.onnx'] = durable.file_hash(asset)
        value['cells'][0].update(asset_paths=['models/fixture.onnx'], worker_environment={
            'ZENMETRICS_NN_MODEL_DIR': 'models', 'ZENMETRICS_NN_ENGINE': 'rten'})
        self.path.write_text(json.dumps(value))
        _, configs, _ = durable.manifest(self.path)
        config = configs[0]
        self.assertEqual(durable.environment(config, self.root)['ZENMETRICS_NN_MODEL_DIR'], str(models))
        result = self.finish(self.start())
        self.assertEqual(result['cells'][0]['status'], 'ok')
        asset.write_bytes(b'changed graph')
        with self.assertRaisesRegex(durable.Refusal, 'asset changed'):
            durable.verify_assets(config)
        asset.write_bytes(b'frozen graph')
        value['cells'][0]['asset_paths'] = []
        self.path.write_text(json.dumps(value))
        with self.assertRaisesRegex(durable.Refusal, 'active asset hashes'):
            durable.manifest(self.path)
        value['cells'][0]['worker_environment'] = {'LD_PRELOAD': 'unreviewed'}
        self.path.write_text(json.dumps(value))
        with self.assertRaisesRegex(durable.Refusal, 'environment key'):
            durable.manifest(self.path)

    def setUp(self):
        scratch = Path.home() / 'tmp'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='durable-test-', dir=scratch)
        self.root = Path(self.temp.name)
        self.binary = self.root / 'worker'
        self.binary.write_text(WORKER)
        self.binary.chmod(0o755)
        (self.root / 'ref.rgb').write_bytes(b'abc')
        (self.root / 'dis.rgb').write_bytes(b'def')
        self.output = self.root / 'out'
        self.path = self.root / 'manifest.json'
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait()
        self.temp.cleanup()

    def freeze(self, *behaviors, phase='warm'):
        cells = []
        for i, behavior in enumerate(behaviors or ('ok',)):
            cell = dict(metric_name=f'metric{i}', width=1, height=1, threads=1,
                        reference='ref.rgb', distorted='dis.rgb', timing_phase=phase,
                        only_mode='score', behavior=behavior, required_samples=10 if phase=='warm' else 1)
            if phase == 'cold':cell['cold_repeat'] = i
            cells.append(cell)
        value = dict(schema=1, binary='worker', binary_sha256=durable.file_hash(self.binary),
                     input_sha256={p:durable.file_hash(self.root/p) for p in ('ref.rgb','dis.rgb')},
                     metadata={}, cells=cells)
        self.path.write_text(json.dumps(value))
        return value

    def start(self, action='run', extra=()):
        cmd = [sys.executable, durable.SELF, action, str(self.path), str(self.output),
               '--timeout-seconds', '1', '--lock', str(self.root/'heavy.lock'), *extra]
        log = (self.root / f'controller-{len(self.children)}.log').open('wb')
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        self.children.append(child)
        return child

    def finish(self, child, code=0):
        actual = child.wait(timeout=25)
        logs = '\n'.join(p.read_text() for p in self.root.glob('controller-*.log'))
        self.assertEqual(actual, code, logs)
        return json.loads((self.output/'report.json').read_text()) if actual==0 else None

    def until(self, fn):
        end = time.monotonic()+10
        while time.monotonic()<end:
            value = fn()
            if value:return value
            time.sleep(.025)
        self.fail('condition not reached')

    def running_attempt(self):
        paths = list(self.output.glob('cells/*/attempt-*/worker.pid'))
        return paths[-1].parent if paths else None

    def no_live(self, pid):
        # A zombie has already stopped executing; container init may reap later.
        try:
            text = subprocess.check_output(['ps','-p',str(pid),'-o','stat='],text=True).strip()
        except subprocess.CalledProcessError:return True
        return not text or text.startswith('Z')

    def test_cold_and_warm_separate_resume(self):
        value=self.freeze('ok',phase='cold')
        warm=dict(value['cells'][0], timing_phase='warm',required_samples=10)
        warm.pop('cold_repeat');value['cells'].append(warm)
        self.path.write_text(json.dumps(value))
        report=self.finish(self.start())
        self.assertEqual((len(report['cold']),len(report['warm'])),(1,1))
        self.assertEqual(report['cold'][0]['n'],1)
        self.assertEqual(report['warm'][0]['n'],10)
        before=(self.root/'execution.jsonl').read_bytes()
        self.finish(self.start())
        self.assertEqual((self.root/'execution.jsonl').read_bytes(),before)
        self.assertTrue(report['complete'])

    def test_failures_are_capped_and_good_cell_advances(self):
        self.freeze('nonzero','ok')
        report=self.finish(self.start())
        self.assertEqual([c['status'] for c in report['cells']],['failed','ok'])
        order=[json.loads(s)['metric'] for s in (self.root/'execution.jsonl').read_text().splitlines()]
        self.assertEqual(order,['metric0','metric1','metric0'])
        self.assertFalse(report['qualified'])
        self.finish(self.start())
        self.assertEqual(len((self.root/'execution.jsonl').read_text().splitlines()),3)

    def test_rejected_protocols_never_success(self):
        self.freeze('killed','partial_write','partial','wrong','no_row','duplicate','missing_rounds','bad_bits','bad_score','failed')
        report=self.finish(self.start(extra=('--max-attempts','1')))
        self.assertTrue(all(c['status']=='failed' and c['row'] is None for c in report['cells']))

    def test_explicit_unsupported_terminal(self):
        self.freeze('unsupported')
        report=self.finish(self.start())
        self.assertEqual(report['cells'][0]['status'],'unsupported')
        self.assertEqual(len(report['cells'][0]['attempts']),1)
        self.assertEqual(report['warm'],[])

    def test_wall_deadline_ignores_continuing_output_and_kills_grandchild(self):
        self.freeze('stall','ok')
        child=self.start(extra=('--max-attempts','1'))
        attempt=self.until(self.running_attempt)
        pid=int((attempt/'worker.pid').read_text())
        grand=int(self.until(lambda:(attempt/'grandchild.pid').read_text() if (attempt/'grandchild.pid').exists() else None))
        report=self.finish(child)
        self.assertEqual(report['cells'][0]['status'],'failed')
        self.assertEqual(report['cells'][0]['attempts'][0]['reason'],'timeout')
        self.assertEqual(report['cells'][1]['status'],'ok')
        self.assertTrue(self.no_live(pid));self.assertTrue(self.no_live(grand))
        self.assertIn('still alive',(attempt/'worker.log').read_text())

    def crash_control(self, sig):
        self.freeze('stall','ok')
        child=self.start(extra=('--timeout-seconds','8'))
        attempt=self.until(self.running_attempt)
        pid=int((attempt/'worker.pid').read_text())
        child.send_signal(sig)
        self.until(lambda:self.no_live(pid))
        self.until(lambda:(attempt/'result.json').exists())
        if sig==signal.SIGSTOP:
            child.send_signal(signal.SIGCONT)
            report=self.finish(child)
        else:
            child.wait(timeout=10)
            report=self.finish(self.start(extra=('--timeout-seconds','8')))
        self.assertEqual(report['cells'][1]['status'],'ok')
        self.assertEqual(report['cells'][0]['status'],'failed')
        self.assertEqual(len(report['cells'][0]['attempts']),2)
        self.assertTrue(self.no_live(pid))
        order=[json.loads(s)['metric'] for s in (self.root/'execution.jsonl').read_text().splitlines()]
        self.assertEqual(order,['metric0','metric1','metric0'])

    def test_controller_sigkill_resume_no_orphan(self):
        self.crash_control(signal.SIGKILL)

    def test_controller_sigstop_wall_guard_resume(self):
        self.crash_control(signal.SIGSTOP)

    def test_sigterm_clean_stop_then_restart(self):
        self.freeze('stall','ok')
        child=self.start(extra=('--timeout-seconds','8'))
        attempt=self.until(self.running_attempt)
        pid=int((attempt/'worker.pid').read_text())
        child.terminate();self.finish(child,143)
        self.assertTrue(self.no_live(pid))
        self.assertEqual(json.loads((self.output/'report.json').read_text())['cells'][1]['status'],'pending')
        self.finish(self.start(extra=('--timeout-seconds','8')))

    def test_concurrent_controllers_refuse(self):
        self.freeze('wait')
        first=self.start(extra=('--timeout-seconds','5'))
        self.until(self.running_attempt)
        self.finish(self.start(extra=('--timeout-seconds','5')),2)
        self.finish(first)

    def test_frozen_tamper_before_first_run_refused(self):
        self.freeze()
        (self.root/'ref.rgb').write_bytes(b'changed')
        self.finish(self.start(),2)
        self.assertFalse((self.root/'execution.jsonl').exists())

    def test_tamper_resume_preserves_results(self):
        self.freeze();self.finish(self.start())
        retained=(self.output/'report.json').read_bytes()
        (self.root/'dis.rgb').write_bytes(b'changed')
        self.finish(self.start(),2)
        self.assertEqual((self.output/'report.json').read_bytes(),retained)

    def test_binary_and_manifest_tamper_refused(self):
        value=self.freeze();self.finish(self.start())
        self.binary.write_text(WORKER+'\n# changed\n')
        self.finish(self.start(),2)
        self.binary.write_text(WORKER)
        value['metadata']['changed']=True;self.path.write_text(json.dumps(value))
        self.finish(self.start(),2)

    def test_log_and_result_tamper_refused(self):
        self.freeze();self.finish(self.start())
        log=next(self.output.glob('cells/*/attempt-*/worker.log'))
        old=log.read_bytes();log.write_bytes(old+b'changed')
        self.finish(self.start(),2)
        log.write_bytes(old)
        result=log.parent/'result.json'
        obj=json.loads(result.read_text());obj['record']['status']='unsupported';result.write_text(json.dumps(obj))
        self.finish(self.start(),2)

    def test_finished_output_adopted_after_result_midwrite(self):
        self.freeze();self.finish(self.start())
        result=next(self.output.glob('cells/*/attempt-*/result.json'))
        result.rename(result.with_name('.result.json.interrupted'))
        before=(self.root/'execution.jsonl').read_bytes()
        report=self.finish(self.start())
        self.assertEqual((self.root/'execution.jsonl').read_bytes(),before)
        self.assertEqual(report['cells'][0]['attempts'][0]['reason'],'adopted_finished_output')

    def test_pending_intent_consumes_attempt_then_good_advances(self):
        self.freeze('ok','ok')
        # Initialize via a real successful run, retain its complete config/intent,
        # then model a crash prior to child launch using a separate fresh output.
        self.finish(self.start())
        first=next((self.output/'cells').iterdir())/'attempt-0001'
        for name in ('result.json','worker.log','exit.json','payload.json','groups.json'):
            (first/name).unlink(missing_ok=True)
        self.finish(self.start())
        report=json.loads((self.output/'report.json').read_text())
        self.assertEqual(len(report['cells'][0]['attempts']),2)

    def test_heartbeat_continues_while_waiting_admission(self):
        self.freeze()
        with (self.root/'heavy.lock').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX)
            child=self.start()
            beat=self.output/'controller-heartbeat'
            self.until(beat.exists)
            initial=beat.stat().st_mtime_ns
            self.until(lambda:beat.stat().st_mtime_ns>initial)
            self.assertFalse((self.root/'execution.jsonl').exists())
            fcntl.flock(guard,fcntl.LOCK_UN)
        self.finish(child)

    def test_sigterm_while_waiting_lock_is_restartable(self):
        self.freeze()
        with (self.root/'heavy.lock').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX)
            child=self.start()
            self.until(lambda:(self.output/'controller-heartbeat').exists())
            child.terminate();self.finish(child,143)
        report=self.finish(self.start())
        self.assertEqual(report['cells'][0]['status'],'ok')

    def test_saved_failed_cleanup_blocks_live_group_without_signaling_it(self):
        self.freeze();self.finish(self.start())
        result=next(self.output.glob('cells/*/attempt-*/result.json'))
        record=durable.read_result(result)
        record.update(status='failed',row=None,cleanup_verified=False)
        durable.save_result(result,record)
        unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        self.children.append(unrelated)
        durable.atomic(result.parent/'groups.json',dict(anchor=unrelated.pid))
        child=self.start()
        self.assertEqual(child.wait(timeout=10),1)
        self.assertIsNone(unrelated.poll(),'recovery must never signal a remembered/reused group ID')
        unrelated.terminate();unrelated.wait(timeout=5)
        report=self.finish(self.start())
        self.assertEqual(report['cells'][0]['status'],'ok')
        self.assertEqual(len(report['cells'][0]['attempts']),2)

    def test_watchdog_sigkill_cleans_nested_groups_then_resume(self):
        self.freeze('stall','ok')
        child=self.start(extra=('--timeout-seconds','8'))
        attempt=self.until(self.running_attempt)
        groups=json.loads((attempt/'groups.json').read_text())
        worker=int((attempt/'worker.pid').read_text())
        supervisor=json.loads((attempt/'watchdog.json').read_text())['pid']
        os.kill(supervisor,signal.SIGKILL)
        self.assertEqual(child.wait(timeout=15),1)
        self.until(lambda:self.no_live(worker) and self.no_live(groups['anchor']))
        report=self.finish(self.start(extra=('--timeout-seconds','8')))
        self.assertEqual(report['cells'][1]['status'],'ok')
        self.assertEqual(len(report['cells'][0]['attempts']),2)

    def test_escape_paths_refused(self):
        value=self.freeze();value['binary']='../worker';self.path.write_text(json.dumps(value))
        self.finish(self.start(),2)
        value=self.freeze();(self.root/'escape').symlink_to('/usr/bin/python3')
        value['binary']='escape';self.path.write_text(json.dumps(value))
        self.finish(self.start(),2)

    def test_nested_session_resource_wrapper_containment(self):
        self.freeze('stall','ok')
        wrapper=self.root/'wrapper'
        wrapper.write_text('#!/usr/bin/env python3\nimport subprocess,sys\na=sys.argv[sys.argv.index("--")+1:]\np=subprocess.Popen(a,start_new_session=True)\nsys.exit(p.wait())\n')
        wrapper.chmod(0o755)
        child=self.start(extra=('--run-heavy',str(wrapper),'--max-attempts','1'))
        attempt=self.until(self.running_attempt);pid=int((attempt/'worker.pid').read_text())
        report=self.finish(child)
        self.assertTrue(self.no_live(pid))
        self.assertEqual(report['cells'][1]['status'],'ok')


if __name__=='__main__':unittest.main(verbosity=2)
