import unittest
from publish_saved_results import public_row,key

class Projection(unittest.TestCase):
    def test_no_scores_paths_or_unrequested_fields(self):
        row=dict(metric_name='zensim',score=12.3,score_bits='x',reference='/private/input',weights_consent='secret',qualified_timing=True)
        out=public_row(row,dict(capture=dict(captured_utc='now')),'mac','ordinary')
        for field in ('score','score_bits','reference','weights_consent'):self.assertNotIn(field,out)
        self.assertEqual(out['metric_name'],'zensim')

    def test_binary_and_input_versions_are_separate_series(self):
        a=dict(host='mac',metric_name='metric',impl_binary_sha='one',reference_sha256='ref')
        b=dict(a,impl_binary_sha='two');c=dict(a,reference_sha256='other')
        self.assertNotEqual(key(a),key(b));self.assertNotEqual(key(a),key(c))
