"""References cannot cross pixel identities or retain stale startup snapshots."""
import json, pathlib, tempfile, unittest
from references import load_references, same_pair_reference

class References(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(dir=pathlib.Path.home()/'tmp')
        self.addCleanup(self.directory.cleanup)
        self.path=pathlib.Path(self.directory.name)/'cpu.jsonl'
        self.cpu=dict(host='fixture',os='Linux-fixture',impl='zenmetrics',device='cpu',mode='score',status='ok',
                      metric_name='ssim2',size_name='cid22',threads=1,pair_sha256='pair-a',score=91.,n=10)
        self.path.write_text(json.dumps(self.cpu)+'\n')
        self.gpu=self.cpu|dict(metric_name='ssim2-gpu',device='cuda')
    def test_same_pair_matches(self):
        self.assertEqual(same_pair_reference(load_references([self.path]),self.gpu),self.cpu)
    def test_changed_pixels_do_not_match(self):
        self.assertIsNone(same_pair_reference(load_references([self.path]),self.gpu|dict(pair_sha256='pair-b')))
    def test_other_thread_budget_does_not_match(self):
        self.assertIsNone(same_pair_reference(load_references([self.path]),self.gpu|dict(threads=20)))
    def test_competitor_cannot_be_a_rust_reference(self):
        self.path.write_text(json.dumps(self.cpu|dict(impl='piq'))+'\n')
        self.assertIsNone(same_pair_reference(load_references([self.path]),self.gpu))
    def test_other_host_or_os_does_not_match(self):
        for field in ['host','os']:
            self.assertIsNone(same_pair_reference(load_references([self.path]),self.gpu|{field:'another'}))
    def test_newly_appended_reference_is_seen(self):
        before=load_references([self.path])
        with self.path.open('a') as f:f.write(json.dumps(self.cpu|dict(score=92.))+'\n')
        self.assertEqual(same_pair_reference(before,self.gpu)['score'],91.)
        self.assertEqual(same_pair_reference(load_references([self.path]),self.gpu)['score'],92.)
    def warmup(self,**changes):
        return self.cpu|dict(status='skipped:timeout_1100s',n=0,score=None,median_ns=None,
            initialization_diagnostics=[dict(completed=True,score_bits='4056c00000000000')],
            log_path='retained.log',log_sha256='retained-sha')|changes
    def test_completed_warmup_reference_does_not_promote_timing(self):
        row=self.warmup();self.path.write_text(json.dumps(row)+'\n')
        ref=same_pair_reference(load_references([self.path]),self.gpu)
        self.assertEqual(ref['score'],91.)
        self.assertEqual(ref['reference_kind'],'completed_untimed_warmup')
        self.assertEqual(ref['status'],'skipped:timeout_1100s')
        self.assertEqual(ref['n'],0);self.assertIsNone(ref['median_ns'])
        self.assertIsNone(row['score'])
    def test_invalid_warmup_is_not_a_reference(self):
        for diagnostics in [[],[dict(completed=False,score_bits='4056c00000000000')],
                [dict(completed=True,score_bits='7ff8000000000000')],
                [dict(completed=True,score_bits='invalid')]]:
            self.path.write_text(json.dumps(self.warmup(initialization_diagnostics=diagnostics))+'\n')
            self.assertIsNone(same_pair_reference(load_references([self.path]),self.gpu))
    def test_unrelated_failure_or_missing_log_cannot_supply_warmup(self):
        for changes in [dict(status='skipped:process_exit_1'),dict(log_sha256=None)]:
            self.path.write_text(json.dumps(self.warmup(**changes))+'\n')
            self.assertIsNone(same_pair_reference(load_references([self.path]),self.gpu))
    def test_warmup_does_not_replace_timed_reference(self):
        with self.path.open('a') as f:f.write(json.dumps(self.warmup())+'\n')
        self.assertEqual(same_pair_reference(load_references([self.path]),self.gpu),self.cpu)
    def test_ok_without_sufficient_rounds_is_not_a_timed_reference(self):
        for count in [0,1,9]:
            self.path.write_text(json.dumps(self.cpu|dict(n=count))+'\n')
            self.assertIsNone(same_pair_reference(load_references([self.path]),self.gpu))
    def test_nonfinite_ok_score_is_not_a_timed_reference(self):
        for score in [None,float('inf'),float('nan')]:
            self.path.write_text(json.dumps(self.cpu|dict(score=score))+'\n')
            self.assertIsNone(same_pair_reference(load_references([self.path]),self.gpu))
if __name__=='__main__':unittest.main()
