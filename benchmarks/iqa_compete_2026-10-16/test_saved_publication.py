import unittest
from publish_saved_results import public_row,key

class Projection(unittest.TestCase):
    def test_insufficient_warm_samples_remain_unqualified_without_imputation(self):
        meta=dict(capture=dict(captured_utc='now'))
        row=dict(metric_name='dmm-pyiqa',status='failed',reason='insufficient warm samples',qualified_timing=False)
        out=public_row(row,meta,'mac','private-nn-production')
        self.assertEqual(out['failure_class'],'insufficient_warm_samples')
        self.assertFalse(out['qualified_timing'])
        self.assertFalse(out['preferred'])
        self.assertIsNone(out['n'])
        self.assertIsNone(out['median_ns'])
        row['reason']='insufficient memory'
        self.assertEqual(public_row(row,meta,'mac','private-nn-production')['failure_class'],
                         'worker_failure_unclassified')

    def test_explicit_guard_evidence_has_a_resource_failure_class(self):
        row=dict(metric_name='topiq_fr',status='failed',reason='worker/wrapper exited 137',
                 resource_guard_stop='monitored_memory_guard',qualified_timing=False)
        meta=dict(capture=dict(captured_utc='now'))
        out=public_row(row,meta,'mac','private-nn-production')
        self.assertEqual(out['failure_class'],'resource_guard_monitored_memory')
        row['resource_guard_stop']=None
        self.assertEqual(public_row(row,meta,'mac','private-nn-production')['failure_class'],
                         'worker_failure_unclassified')

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

    def test_rerun_cells_count_once_with_their_best_state(self):
        from publish_saved_results import cell_coverage,in_scope
        a,b,c,d=('a',),('b',),('c',),('d',)
        got=cell_coverage([(a,'pending'),(a,'ok'),(b,'failed'),(b,'refused_by_design'),(c,'pending'),(d,'failed'),(d,'pending')])
        self.assertEqual(got,dict(planned=4,ok=1,refused_by_design=1,failed=1,pending=1))
        base=dict(only_mode='score',width=512,height=512,threads=8,timing_phase='warm',coverage_first=False,
                  sample_target_ns=100_000_000,required_samples=10,cold_repeat=0)
        self.assertTrue(in_scope(base))
        for change in (dict(only_mode='map'),dict(width=7680),dict(threads=4),dict(required_samples=3),
                       dict(timing_phase='cold',required_samples=1,cold_repeat=3)):
            self.assertFalse(in_scope({**base,**change}),change)
