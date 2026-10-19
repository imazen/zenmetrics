"""End-to-end negative controls for quarantine failures and importable bytecode."""
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import bytecode_inventory
import containment
import panel
from test_repairs import fixture

HERE=Path(__file__).resolve().parent


def locks_held(root):
    for name in ('lilith-heavy.lock','lilith-wsl-heavy.lock'):
        with (root/'tmp'/name).open('a') as fd:
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            return False
    return True


def until(predicate,seconds=8):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if predicate():return
        time.sleep(.02)
    raise AssertionError('owned control deadline exceeded')


class QuarantineFailures(unittest.TestCase):
    def test_actual_acquire_holds_both_locks_on_transient_and_persistent_faults(self):
        for kind in ('pidfd','refresh','write','send'):
            for persistent in (False,True):
                with self.subTest(kind=kind,persistent=persistent),tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);(root/'fault').write_text('injected only in owned worker')
                    containment.subreaper()
                    with (root/'controller.log').open('w') as log:
                        worker=subprocess.Popen([sys.executable,str(HERE/'quarantine_control.py'),str(root),kind],stdout=log,stderr=subprocess.STDOUT,env=bytecode_inventory.environment())
                        worker_fd=os.pidfd_open(worker.pid)
                        try:
                            until(lambda:(root/'fault-observed').exists() or worker.poll() is not None)
                            self.assertIsNone(worker.poll(),(root/'controller.log').read_text())
                            self.assertTrue(locks_held(root))
                            if persistent:
                                # A cancellation request must not turn failed cleanup
                                # into ordinary panel lock release.
                                signal.pidfd_send_signal(worker_fd,signal.SIGTERM)
                                end=time.monotonic()+.4
                                while time.monotonic()<end:
                                    self.assertIsNone(worker.poll())
                                    self.assertTrue(locks_held(root));time.sleep(.03)
                            receipts=list((root/'output').glob('*/owned.containment.json'))
                            self.assertEqual(len(receipts),1)
                            if kind!='write':
                                until(lambda:json.loads(receipts[0].read_text())['status']=='containment-blocked')
                            else:
                                until(lambda:Path(str(receipts[0])+'.errors.jsonl').stat().st_size>0)
                            (root/'fault').unlink()  # Explicitly allow recovery.
                            self.assertEqual(worker.wait(timeout=8),0,(root/'controller.log').read_text())
                            result=json.loads(receipts[0].read_text())
                            self.assertEqual(result['status'],'empty');self.assertEqual(result['remaining'],{})
                            self.assertTrue(result['cleanup_failures'])
                            self.assertFalse(locks_held(root))
                            child=json.loads((root/'child.json').read_text())
                            self.assertFalse(Path('/proc/'+str(child['pid'])).exists())
                            block=json.loads((root/'done.json').read_text())[0]
                            self.assertEqual(block['status'],'rejected')
                        finally:
                            (root/'fault').unlink(missing_ok=True)
                            if worker.poll() is None:
                                signal.pidfd_send_signal(worker_fd,signal.SIGTERM)
                                try:worker.wait(timeout=8)
                                except subprocess.TimeoutExpired:
                                    signal.pidfd_send_signal(worker_fd,signal.SIGKILL);worker.wait(timeout=3)
                            os.close(worker_fd)
                            # Bounded backstop owns ONLY this fixture's child.
                            if (root/'child.json').exists():
                                child=json.loads((root/'child.json').read_text());pid=child['pid']
                                observed=containment.processes().get(pid)
                                if observed and observed['start']==child['start']:
                                    fd=os.pidfd_open(pid)
                                    try:
                                        if containment.processes().get(pid,{}).get('start')==child['start']:signal.pidfd_send_signal(fd,signal.SIGKILL)
                                    finally:os.close(fd)
                                    try:os.waitpid(pid,0)
                                    except ChildProcessError:pass

    def test_pidfd_facility_failure_precedes_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with (root/'log').open('w') as log,patch.object(containment.os,'pidfd_open',side_effect=OSError(errno.ENOSYS,'unavailable')),patch.object(containment.subprocess,'Popen') as launch:
                with self.assertRaises(OSError):containment.execute(['never-launch'],log,os.environ.copy(),root/'receipt')
                launch.assert_not_called()

    def test_preexisting_child_is_refused_and_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
            fd=os.pidfd_open(child.pid)
            try:
                with (root/'log').open('w') as log:
                    with self.assertRaises(RuntimeError):containment.execute(['never-launch'],log,os.environ.copy(),root/'receipt')
                self.assertIsNone(child.poll())
            finally:signal.pidfd_send_signal(fd,signal.SIGKILL);child.wait();os.close(fd)


class BytecodeBinding(unittest.TestCase):
    def import_value(self,root):
        return subprocess.check_output([sys.executable,'-B','-c','import specimen;print(specimen.VALUE)'],cwd=root,env=bytecode_inventory.environment(),text=True).strip()

    def write_cache(self,source,value):
        stamp=source.stat();original=source.read_bytes()
        source.write_text('VALUE='+str(value)+'\n')
        os.utime(source,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
        py_compile.compile(str(source),doraise=True,invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
        source.write_bytes(original);os.utime(source,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))

    def test_fresh_interpreter_added_and_mutated_cache_with_unchanged_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'specimen.py';source.write_text('VALUE=1\n')
            original=hashlib.sha256(source.read_bytes()).hexdigest()
            m=fixture(root)
            self.assertEqual(self.import_value(root),'1');panel.verify(m)
            self.write_cache(source,2)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),original)
            self.assertEqual(self.import_value(root),'2') # -B does not forbid reads.
            with self.assertRaises(ValueError):panel.verify(m) # Added cache.
            m['bytecode_binding']=bytecode_inventory.capture([],[str(root)])
            panel.verify(m)
            self.write_cache(source,3)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),original)
            self.assertEqual(self.import_value(root),'3')
            with self.assertRaises(ValueError):panel.verify(m) # Mutated cache.

    def test_sourceless_bytecode_is_in_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'specimen.py';source.write_text('VALUE=4\n')
            py_compile.compile(str(source),cfile=str(root/'specimen.pyc'),doraise=True)
            source.unlink();binding=bytecode_inventory.capture([],[str(root)])
            self.assertEqual(self.import_value(root),'4')
            self.assertIn(str(root/'specimen.pyc'),[f['path'] for f in binding['inventory']['files']])
            (root/'specimen.pyc').write_bytes(b'changed')
            with self.assertRaises(ValueError):bytecode_inventory.verify(binding)
