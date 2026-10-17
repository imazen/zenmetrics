"""Sampling/identity regressions for bounded-chunk aggregation."""
import copy,statistics,unittest
from chunks import combine

def piece(values,score=1.):
    return dict(metric_name='fixture',mode='score',size_name='fixture',threads=1,device='cpu',pair_sha256='pair',impl_binary_sha='binary',round_means_ns=values,n=len(values),score=score,load_before=[0.],load_after=[0.],zenbench_arm_index=0,zenbench_samples=[{'compensated_ns':[v],'iterations':1} for v in values],status='partial',partial_reason='time budget',resource=dict(exit_code=0,seconds=1,peak_rss_gib=.1,min_available_mib=10000,peak_load=0.))
class Aggregation(unittest.TestCase):
    def test_six_rounds_cannot_qualify(self):
        self.assertEqual(combine([piece(list(range(1,7)))])['status'],'partial')
    def test_disjoint_chunks_produce_ten_actual_rounds(self):
        a,b=piece(list(range(1,7))),piece(list(range(7,11)))
        result=combine([a,b]);self.assertEqual(result['status'],'ok');self.assertEqual(result['n'],10)
        self.assertEqual(result['median_ns'],statistics.median(range(1,11)))
        self.assertEqual(result['mean_ns'],statistics.mean(range(1,11)))
        self.assertEqual(len(result['zenbench_samples']),10);self.assertEqual(result['p10_ns'],1.9)
        self.assertNotIn('partial_reason',result)
    def test_different_inputs_cannot_merge(self):
        a,b=piece([1.]*6),piece([1.]*4);b['pair_sha256']='other'
        with self.assertRaisesRegex(ValueError,'identity'):combine([a,b])
    def test_different_runtime_or_model_cannot_merge(self):
        for key in ['engine','model_dir','ort_shared_library_sha256','ort_disable_telemetry']:
            a,b=piece([1.]*6),piece([1.]*4);a[key]='first';b[key]='second'
            with self.assertRaisesRegex(ValueError,'identity'):combine([a,b])
    def test_score_bits_and_map_score_disagreement_are_preserved(self):
        self.assertEqual(combine([piece([1.]*6),piece([1.]*4,1.0000001)])['status'],'variant_mismatch')
        a,b=piece([1.]*6),piece([1.]*4);b['map_score_matches_scalar_bits']=False
        self.assertEqual(combine([a,b])['status'],'variant_mismatch')
    def test_sample_count_must_match_recorded_values(self):
        a=piece([1.]*6);a['n']=7
        with self.assertRaisesRegex(ValueError,'count'):combine([a])
if __name__=='__main__':unittest.main()
