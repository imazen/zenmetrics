"""The completion gate refuses missing, diagnostic and unpaired observations."""
import json,pathlib,subprocess,sys,tempfile,unittest
P=pathlib.Path
class Coverage(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=P.home()/'tmp');self.addCleanup(self.temp.cleanup);self.root=P(self.temp.name)
        (self.root/'inventory.json').write_text(json.dumps([dict(metric_name='fixture',gpu=False)]))
        (self.root/'pairs.json').write_text(json.dumps([dict(size_name='fixture',width=2,height=2)]))
        row=dict(host='fixture',os='Linux-fixture',impl='zenmetrics',metric_name='fixture',device='cpu',size_name='fixture',threads=1,width=2,height=2,pair_sha256='pixels')
        self.options=[]
        self.rows=[row|dict(mode='score',status='ok',n=10),row|dict(mode='score+map',status='unsupported',notes='No map API')]
    def run_gate(self):
        raw=self.root/'raw.jsonl';raw.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        result=subprocess.run([sys.executable,str(P(__file__).with_name('coverage.py')),str(raw),str(self.root/'inventory.json'),str(self.root/'pairs.json'),str(self.root/'receipt.json'),'--sizes','fixture','--threads','1','--require-complete',*self.options],capture_output=True,text=True)
        return result,json.loads((self.root/'receipt.json').read_text())
    def test_complete_requested_cpu_routes(self):
        result,receipt=self.run_gate();self.assertEqual(result.returncode,0,result.stderr);self.assertTrue(receipt['complete'])
    def test_missing_cell_fails(self):
        self.rows.pop();result,receipt=self.run_gate();self.assertEqual(result.returncode,1);self.assertEqual(len(receipt['missing_cells']),1)
    def test_incomplete_sampling_fails(self):
        self.rows[0]['n']=9;result,receipt=self.run_gate();self.assertEqual(result.returncode,1);self.assertEqual(len(receipt['pending_qualification']),1)
    def test_diagnostic_timing_fails(self):
        self.rows[0]['status']='skipped:diagnostic_unverified_ort_thread_budget';result,receipt=self.run_gate();self.assertEqual(result.returncode,1)
    def test_unpaired_gpu_timing_fails(self):
        (self.root/'inventory.json').write_text(json.dumps([dict(metric_name='fixture',gpu=True)]))
        self.options=['--gpu-runtime','cuda']
        for row in self.rows:row['device']='cuda'
        result,receipt=self.run_gate();self.assertEqual(result.returncode,1);self.assertEqual(receipt['pending_qualification'][0]['reason'],'same-pair CPU comparison missing')
    def test_unexpected_process_failure_fails(self):
        self.rows[0]['status']='skipped:process_exit_101';result,receipt=self.run_gate();self.assertEqual(result.returncode,1)
    def test_existing_native_map_cannot_be_marked_unsupported(self):
        (self.root/'inventory.json').write_text(json.dumps([dict(metric_name='fsim-y',gpu=False)]))
        for row in self.rows:row['metric_name']='fsim-y'
        result,receipt=self.run_gate();self.assertEqual(result.returncode,1)
        self.assertEqual(receipt['pending_qualification'][0]['reason'],'supported native-map route lacks an adapter measurement')
if __name__=='__main__':unittest.main()
