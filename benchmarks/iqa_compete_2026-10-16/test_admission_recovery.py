"""Real OS guards protect live/other reservations while preserving our dead one."""
import fcntl,json,os,pathlib,shutil,tempfile,time,unittest
from unittest.mock import patch
import admission_recovery
from admission_recovery import recover
P=pathlib.Path
class Recovery(unittest.TestCase):
 def setUp(self):
  scratch=P.home()/'tmp';scratch.mkdir(exist_ok=True)
  self.root=P(tempfile.mkdtemp(prefix='iqa-admission-test-',dir=scratch));self.addCleanup(shutil.rmtree,self.root)
  self.cache=self.root/'cache';self.cache.mkdir();self.archive=self.root/'archive'
  self.proof=dict(pid=1073741824,pgid=1073741824,unix_s=int(time.time()))
  self.record=dict(sequence=1,pid=self.proof['pid'],start=self.proof['unix_s'],activity='benchmark',running=True)
  self.ticket=self.cache/'1.json';self.guard=self.cache/'1.guard';self.guard.touch();self.write()
 def write(self):self.ticket.write_text(json.dumps(self.record))
 def run_recovery(self):return recover(self.proof,self.archive,self.cache)
 def test_dead_owner_is_archived_verbatim(self):
  original=self.ticket.read_bytes();rows=self.run_recovery()
  self.assertEqual(len(rows),1);self.assertFalse(self.ticket.exists())
  self.assertEqual(P(rows[0]['archive']).read_bytes(),original)
  self.assertEqual(P(rows[0]['preserved']).read_bytes(),original)
  self.assertTrue(self.guard.exists())
 def test_live_owner_is_untouched(self):
  self.proof['pid']=os.getpid();self.record['pid']=os.getpid();self.write()
  self.assertEqual(self.run_recovery(),[]);self.assertTrue(self.ticket.exists())
 def test_live_process_group_is_untouched(self):
  self.proof['pgid']=os.getpgrp()
  self.assertEqual(self.run_recovery(),[]);self.assertTrue(self.ticket.exists())
 def test_other_owner_is_untouched(self):
  self.record['pid']-=1;self.write()
  self.assertEqual(self.run_recovery(),[]);self.assertTrue(self.ticket.exists())
 def test_locked_guard_is_untouched(self):
  with self.guard.open('a+') as guard:
   fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
   self.assertEqual(self.run_recovery(),[]);self.assertTrue(self.ticket.exists())
 def test_old_pid_record_is_untouched(self):
  self.record['start']-=1000;self.write()
  self.assertEqual(self.run_recovery(),[]);self.assertTrue(self.ticket.exists())
 def test_invalid_process_proof_is_refused(self):
  self.proof['pid']=True
  with self.assertRaisesRegex(ValueError,'proof'):self.run_recovery()
  self.assertTrue(self.ticket.exists())
class DarwinRecovery(Recovery):
 def setUp(self):
  super().setUp()
  platform_patch=patch.object(admission_recovery.sys,'platform','darwin')
  platform_patch.start();self.addCleanup(platform_patch.stop)

if __name__=='__main__':unittest.main()
