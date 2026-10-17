"""A score startup failure cannot create or hide a frozen map capability."""
import copy,unittest
from run import normalize_nomap
class Capabilities(unittest.TestCase):
 def row(self,metric='ckdn',impl='zenmetrics',mode='score+map',n=0,status='skipped:timeout_1100s'):
  return dict(metric_name=metric,impl=impl,mode=mode,n=n,status=status)
 def test_learned_and_nr_maps_remain_unsupported_on_score_failure(self):
  for metric in ['ckdn','niqe']:
   row=self.row(metric=metric);normalize_nomap(row)
   self.assertEqual(row['status'],'unsupported');self.assertEqual(row['source_status'],'skipped:timeout_1100s')
 def test_score_failure_is_preserved(self):
  row=self.row(mode='score');original=copy.deepcopy(row);normalize_nomap(row);self.assertEqual(row,original)
 def test_competitor_maps_are_not_reclassified(self):
  row=self.row(impl='upstream',n=10,status='ok');original=copy.deepcopy(row);normalize_nomap(row);self.assertEqual(row,original)
 def test_supported_classical_map_failure_is_preserved(self):
  row=self.row(metric='cvvdp');original=copy.deepcopy(row);normalize_nomap(row);self.assertEqual(row,original)
 def test_unexpected_native_maps_require_investigation(self):
  with self.assertRaisesRegex(ValueError,'unexpectedly'):normalize_nomap(self.row(n=10,status='ok'))
if __name__=='__main__':unittest.main()
