"""Exercise unchanged numeric gates and exact-pair comparison of saved rows."""
import copy,unittest
from parity import qualify_rows, reproduced_residual

class Parity(unittest.TestCase):
    def setUp(self):
        self.row=dict(host='fixture',os='Linux-fixture',metric_name='butteraugli-gpu',size_name='cid22',threads=1,
                      pair_sha256='a',status='ok',score=1.2,n=10,median_ns=10.,
                      comparison_status='unavailable:no qualified same-pair CPU reference')
        self.ref=dict(score=1.2,impl_binary_sha='rust',pair_sha256='a',threads=1)
        self.references={('fixture','Linux-fixture','butteraugli','cid22',1,'a'):self.ref}
    def test_matching_reference_clears_unavailable_status(self):
        self.assertFalse(qualify_rows([self.row],self.references,[],{}))
        self.assertEqual(self.row['comparison_status'],'same-pair CPU reference available')
        self.assertEqual(self.row['reference_pair_sha256'],'a')
        self.assertEqual(self.row['parity_abs_bound'],1.2*.002)
    def test_existing_bound_keeps_real_residual_mismatched(self):
        self.row['score']=1.21
        self.assertTrue(qualify_rows([self.row],self.references,[],{}))
        self.assertEqual(self.row['status'],'variant_mismatch')
    def test_other_pixels_do_not_supply_a_reference(self):
        self.row['pair_sha256']='b'
        self.assertFalse(qualify_rows([self.row],self.references,[],{}))
        self.assertIsNone(self.row['ref_score'])
    def test_comparison_keeps_measurement(self):
        original=copy.deepcopy(self.row)
        qualify_rows([self.row],self.references,[],{})
        for key in ['score','n','median_ns']:self.assertEqual(self.row[key],original[key])
    def test_saved_repeat_must_reproduce_the_current_residual(self):
        self.row.update(mode='score',score=1.21,mismatch_first_attempt=[dict(mode='score',score=1.2)])
        qualify_rows([self.row],self.references,[],{})
        self.assertFalse(reproduced_residual(self.row))
        self.row['mismatch_first_attempt'][0]['score']=1.21
        self.assertTrue(reproduced_residual(self.row))
    def test_bit_variation_remains_a_mismatch_at_zero_delta(self):
        self.row['gpu_call_bit_variation_count']=1
        self.assertTrue(qualify_rows([self.row],self.references,[],{}))
        self.assertEqual(self.row['abs_delta'],0.)
        self.assertEqual(self.row['status'],'variant_mismatch')

class ApprovedAlternativeTolerance(unittest.TestCase):
    def test_pyiqa_gmsd_bound_is_package_scoped(self):
        for package, delta, mismatch in [('pyiqa', 1.1e-8, False), ('pyiqa', 2.1e-8, True), ('piq', 1.1e-8, True)]:
            row=dict(host='fixture',os='Linux-fixture',metric_name='gmsd',size_name='1mp',threads=1,
                     pair_sha256='a',status='ok',score=delta,n=1)
            refs={('fixture','Linux-fixture','gmsd','1mp',1,'a'):dict(score=0.,impl_binary_sha='rust',pair_sha256='a',threads=1)}
            self.assertEqual(qualify_rows([row],refs,[],{},package),mismatch)

if __name__=='__main__':unittest.main()
