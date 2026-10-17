import unittest
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
class Retry(unittest.TestCase):
 def check(self,peaks):
  with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as root:
   args=SimpleNamespace(root=Path(root));rows=[]
   for i,peak in enumerate(peaks):
    rows.append([dict(attempt=str(i),timing_quality='contaminated' if peak>1 else 'clean',contention=dict(max_competing_interval_cores=peak))])
   with patch.object(run,'_run_cell',side_effect=rows) as execute:
    chosen=run.run_cell(args,dict(metric_name='example',size_name='cid22'),{})
   return chosen,execute.call_count
 def test_stops_on_clean_attempt(self):
  chosen,count=self.check([3,.5,2]);self.assertEqual(count,2);self.assertEqual(chosen[0]['attempt'],'1')
 def test_three_tries_keep_cleanest_contaminated(self):
  chosen,count=self.check([3,2,4]);self.assertEqual(count,3);self.assertEqual(chosen[0]['attempt'],'1');self.assertEqual(chosen[0]['timing_quality'],'contaminated')
if __name__=='__main__':unittest.main()
