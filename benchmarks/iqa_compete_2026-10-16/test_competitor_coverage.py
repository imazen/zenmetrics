import unittest
from competitor_coverage import audit

class Competitors(unittest.TestCase):
    def setUp(self):
        self.pairs=[dict(size_name='cid22',width=512,height=512)]
        self.row=dict(host='fixture',os='fixture',impl='scikit-image',metric_name='psnr',
            device='cpu',size_name='cid22',threads=1,mode='score',width=512,height=512,
            pair_sha256='fixture',status='ok',n=10,score=30.,ref_score=30.)
    def check(self,rows):return audit(rows,self.pairs,['cid22'],[1],['scikit-image'])
    def test_missing_is_explicit(self):
        result=self.check([self.row]);self.assertEqual(result['expected_cells'],4)
        self.assertEqual(len(result['missing_cells']),3);self.assertFalse(result['complete'])
    def test_sampling_and_reference_are_separate_gates(self):
        for changes in [dict(n=2,timed_seconds=4),dict(ref_score=None)]:
            self.assertEqual(len(self.check([self.row|changes])['pending_qualification']),1)
        self.assertFalse(self.check([self.row|dict(n=3,timed_seconds=3)])['pending_qualification'])
    def test_failed_adapter_and_missing_reference_are_pending(self):
        for status in ['skipped:process_exit_1','skipped:missing_same-pair_Rust_reference']:
            self.assertEqual(len(self.check([self.row|dict(status=status)])['pending_qualification']),1)
    def test_refusal_is_not_a_timing(self):
        r=self.check([self.row|dict(status='skipped:timeout_1100s',n=0,score=None)])
        self.assertEqual(r['status_counts'],{'skipped':1});self.assertFalse(r['pending_qualification'])
    def test_duplicates_and_other_testbeds_refuse(self):
        for rows in [[self.row,self.row],[self.row,self.row|dict(os='another')]]:
            with self.assertRaises(ValueError):self.check(rows)
    def test_wrong_input_refuses(self):
        with self.assertRaises(ValueError):self.check([self.row|dict(width=511)])
        with self.assertRaises(ValueError):self.check([self.row,self.row|dict(metric_name='ssimc',pair_sha256='another')])
if __name__=='__main__':unittest.main()
