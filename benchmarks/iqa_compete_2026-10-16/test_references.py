"""References cannot cross pixel identities or retain stale startup snapshots."""
import json, pathlib, tempfile, unittest
from references import load_references, same_pair_reference

class References(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(dir=pathlib.Path.home()/'tmp')
        self.addCleanup(self.directory.cleanup)
        self.path=pathlib.Path(self.directory.name)/'cpu.jsonl'
        self.cpu=dict(host='fixture',os='Linux-fixture',impl='zenmetrics',device='cpu',mode='score',status='ok',
                      metric_name='ssim2',size_name='cid22',threads=1,pair_sha256='pair-a',score=91.)
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
if __name__=='__main__':unittest.main()
