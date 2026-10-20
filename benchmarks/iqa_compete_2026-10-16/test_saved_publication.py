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

    def test_private_nn_aggregates_keep_hashes_and_failed_numerical_gate(self):
        from publish_saved_results import public_provenance
        meta=dict(inputs={'/private/model/graph.onnx':'abc'},capture=dict(captured_utc='now'))
        out=public_provenance(meta,'private-nn-controls')
        self.assertEqual(out['inputs'],dict(content_sha256=['abc']))
        row=public_row(dict(metric_name='dmm-pyiqa',score=1.0),meta,'mac','private-nn-controls')
        self.assertEqual(row['numerical_qualification'],'failed_native_gate')
        self.assertNotIn('score',row)

class Rounds(unittest.TestCase):
    def test_rounds_are_saved_means_and_not_invented_iterations(self):
        from publish_saved_results import round_rows
        public=dict(host='mac',timing_phase='warm',status='ok')
        raw=dict(round_means_ns=[20,30],zenbench_rounds=[dict(iterations=5,elapsed_ns=[105],compensated_ns=[100],gate_clean=True),dict(iterations=4,elapsed_ns=[125],compensated_ns=[120],gate_clean=False)])
        rows=list(round_rows(public,raw))
        self.assertEqual([r['time_ns'] for r in rows],[20,30])
        self.assertEqual([r['gate_clean'] for r in rows],[True,False])
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['iterations'],5)

    def test_no_warm_round_imputation(self):
        from publish_saved_results import round_rows
        self.assertEqual(list(round_rows(dict(timing_phase='warm',status='ok',median_ns=25),{})),[])
