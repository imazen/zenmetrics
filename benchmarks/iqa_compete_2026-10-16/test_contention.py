import unittest
from contention import interval
class Accounting(unittest.TestCase):
 def test_reused_pid_excluded(self):
  self.assertEqual(interval((0,{(1,2):{'ticks':0}}),(1,{(1,3):{'ticks':200}}),100),[])
 def test_two_cores(self):
  row={'pid':1,'ticks':0};last=dict(row,ticks=400)
  self.assertEqual(interval((0,{(1,2):row}),(2,{(1,2):last}),100)[0]['cores'],2)
 def test_backward_ticks_never_negative(self):
  self.assertEqual(interval((0,{(1,2):{'ticks':5}}),(1,{(1,2):{'ticks':0}}),100)[0]['cores'],0)
if __name__=='__main__':unittest.main()
