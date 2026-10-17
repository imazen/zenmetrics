import unittest
import os,subprocess,sys,json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run
from contention import interval
class Accounting(unittest.TestCase):
 def test_reused_pid_excluded(self):
  self.assertEqual(interval((0,{(1,2):{'ticks':0}}),(1,{(1,3):{'ticks':200}}),100),[])
 def test_two_cores(self):
  row={'pid':1,'ticks':0};last=dict(row,ticks=400)
  self.assertEqual(interval((0,{(1,2):row}),(2,{(1,2):last}),100)[0]['cores'],2)
 def test_backward_ticks_never_negative(self):
  self.assertEqual(interval((0,{(1,2):{'ticks':5}}),(1,{(1,2):{'ticks':0}}),100)[0]['cores'],0)
class LockInheritance(unittest.TestCase):
 def test_supervised_child_keeps_inherited_lock_descriptor(self):
  with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
   root=Path(directory);fd=os.open(root/'lock',os.O_CREAT|os.O_RDWR,0o600)
   try:
    command=[sys.executable,str(Path(__file__).with_name('contention.py')),str(root/'receipt.json'),str(root/'log'),
     '--',sys.executable,'-c',"import os;os.fstat(int(os.environ['IQA_HEAVY_LOCK_FD']))"]
    result=subprocess.run(command,env=dict(os.environ,IQA_HEAVY_LOCK_FD=str(fd)),pass_fds=(fd,),capture_output=True,text=True)
    self.assertEqual(result.returncode,0,result.stderr)
    self.assertEqual(json.loads((root/'receipt.json').read_text())['exit_code'],0)
   finally:os.close(fd)

class Retry(unittest.TestCase):
 def check(self,peaks):
  with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as root:
   args=SimpleNamespace(root=Path(root));rows=[]
   for i,peak in enumerate(peaks):
    rows.append([dict(attempt=str(i),timing_quality='contaminated' if peak>1 else 'clean',contention=dict(max_competing_interval_cores=peak))])
   with patch.object(run,'_run_cell',side_effect=rows) as execute:
    chosen=run.run_cell(args,dict(metric_name='example',size_name='cid22'),{})
   return chosen,execute.call_count
 def test_legacy_rows_preserve_finite_json(self):
  import json
  with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as root:
   args=SimpleNamespace(root=Path(root))
   with patch.object(run,'_run_cell',return_value=[dict(attempt='legacy')]):
    selected=run.run_cell(args,dict(metric_name='example',size_name='cid22'),{})
   json.dumps(selected,allow_nan=False)
   self.assertIsNone(selected[0]['contention_attempts'][0]['max_competing_interval_cores'])
 def test_stops_on_clean_attempt(self):
  chosen,count=self.check([3,.5,2]);self.assertEqual(count,2);self.assertEqual(chosen[0]['attempt'],'1')
 def test_three_tries_keep_cleanest_contaminated(self):
  chosen,count=self.check([3,2,4]);self.assertEqual(count,3);self.assertEqual(chosen[0]['attempt'],'1');self.assertEqual(chosen[0]['timing_quality'],'contaminated')
if __name__=='__main__':unittest.main()
