"""Real-lock controls for bounded holds and inherited-child ownership."""
import fcntl,json,os,pathlib,subprocess,sys,tempfile,unittest
from unittest.mock import patch
from batch import BatchLease,yield_after_release
P=pathlib.Path
class Batches(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=P.home()/'tmp');self.addCleanup(self.temp.cleanup)
        self.root=P(self.temp.name);self.path=self.root/'heavy.lock'
        self.lease=BatchLease(self.path,1100,self.root/'events.jsonl');self.addCleanup(self.lease.close)
    def test_multiple_cells_share_one_bounded_hold(self):
        scope,fresh=self.lease.prepare(1100);fd=self.lease.fd
        self.assertTrue(fresh);self.assertTrue(1<=scope<=1100)
        scope,fresh=self.lease.prepare(1100)
        self.assertFalse(fresh);self.assertEqual(self.lease.fd,fd)
        self.lease.close()
        rows=[json.loads(s) for s in (self.root/'events.jsonl').read_text().splitlines()]
        self.assertEqual([r['event'] for r in rows],['acquired','parent_reference_closed'])
        self.assertEqual(rows[-1]['cells_submitted'],2)
    def test_low_remaining_budget_starts_a_fresh_hold(self):
        self.lease.prepare(1100);self.lease.started-=1100
        scope,fresh=self.lease.prepare(1100)
        self.assertTrue(fresh);self.assertTrue(1000<=scope<=1100)
    def test_child_keeps_the_lock_when_parent_reference_closes(self):
        self.lease.prepare(1100)
        child=subprocess.Popen([sys.executable,'-c','import sys; print("ready",flush=True); sys.stdin.buffer.read(1)'],pass_fds=(self.lease.fd,),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=False)
        try:
            self.assertEqual(child.stdout.readline(),b'ready\n');self.lease.close()
            fd=os.open(self.path,os.O_RDWR)
            try:
                with self.assertRaises(BlockingIOError):fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                child.stdin.write(b'x');child.stdin.flush();child.wait(timeout=5)
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:os.close(fd)
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=5)
            child.stdin.close();child.stdout.close()
    def test_event_write_failure_still_closes_parent_fd(self):
        self.lease.prepare(1100);fd=self.lease.fd
        self.lease.event=lambda kind:(_ for _ in ()).throw(OSError('fixture write failure'))
        with self.assertRaisesRegex(OSError,'fixture'):self.lease.close()
        with self.assertRaises(OSError):os.fstat(fd)
    def test_oversized_holds_are_refused(self):
        for seconds in [0,1101]:
            with self.assertRaises(ValueError):BatchLease(self.path,seconds,self.root/'events.jsonl')
    def test_contended_release_yields_after_unlock(self):
        path=self.root/'i265-heavy.lock';lease=BatchLease(path,1100,self.root/'fair-events.jsonl')
        lease.prepare(1100)
        def sleeping(seconds):
            self.assertEqual(seconds,45)
            fd=os.open(path,os.O_RDWR)
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:os.close(fd)
        with patch('batch.subprocess.run',return_value=subprocess.CompletedProcess([],0,'11\n12\n','')) as probe,patch('batch.time.sleep',side_effect=sleeping) as sleep:
            lease.close();sleep.assert_called_once_with(45)
            probe.assert_called_once_with(['pgrep','-x','flock'],capture_output=True,text=True)
    def test_no_yield_without_multiple_flock_processes(self):
        for stdout,code in [('',1),('11\n',0)]:
            with patch('batch.subprocess.run',return_value=subprocess.CompletedProcess([],code,stdout,'')),patch('batch.time.sleep') as sleep:
                yield_after_release(self.root/'i265-heavy.lock');sleep.assert_not_called()
    def test_other_host_locks_do_not_probe(self):
        with patch('batch.subprocess.run') as probe:
            yield_after_release(self.path);probe.assert_not_called()
if __name__=='__main__':unittest.main()
