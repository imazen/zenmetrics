"""The benchmark patch covers every builder and cannot edit the source checkout."""
import pathlib,subprocess,sys,tempfile,unittest
P=pathlib.Path
class SnapshotPatch(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(dir=P.home()/'tmp');self.addCleanup(self.directory.cleanup)
        self.root=P(self.directory.name);self.source=self.root/'source';self.snapshot=self.root/'snapshot'
        self.relative=P('crates/zenmetrics-nn/src/engine.rs')
        original=P(__file__).resolve().parents[2]/self.relative
        self.bytes=original.read_bytes()
        for root in [self.source,self.snapshot]:
            target=root/self.relative;target.parent.mkdir(parents=True);target.write_bytes(self.bytes)
        self.script=P(__file__).with_name('patch_ort_threads.py')
    def run_patch(self,snapshot):
        return subprocess.run([sys.executable,str(self.script),str(snapshot),str(self.source),str(self.root/'receipt.json')],capture_output=True,text=True)
    def test_only_inserted_thread_configuration_changes(self):
        result=self.run_patch(self.snapshot);self.assertEqual(result.returncode,0,result.stderr)
        patched=(self.snapshot/self.relative).read_text()
        self.assertEqual(patched.count('IQA_ORT_THREADS'),3)
        additions='''            // BENCHMARK SNAPSHOT ONLY: explicitly configure the cell budget.
            let budget = super::workers();
            println!("IQA_ORT_THREADS {budget}");
'''
        stripped=patched.replace(additions,'').replace('                .and_then(|b| b.with_intra_threads(budget).map_err(Into::into))\n','').replace('                .and_then(|b| b.with_inter_threads(1).map_err(Into::into))\n','')
        stripped=stripped.replace('            .and_then(|b| b.with_intra_threads(budget).map_err(Into::into))\n','').replace('            .and_then(|b| b.with_inter_threads(1).map_err(Into::into))\n','')
        stripped=stripped.replace('path: &Path, budget: usize','path: &Path').replace('cuda_session(environment()?, path, budget)?','cuda_session(environment()?, path)?')
        stripped=stripped.replace('ep: Ep, budget: usize','ep: Ep').replace('gpu_session(environment()?, path, ep, budget)?','gpu_session(environment()?, path, ep)?')
        for name in ['cuda_session','webgpu_session']:stripped=stripped.replace(f'{name}(env, path, budget)',f'{name}(env, path)')
        self.assertEqual(stripped.encode(),self.bytes)
        self.assertEqual((self.source/self.relative).read_bytes(),self.bytes)
    def test_source_checkout_is_refused(self):
        result=self.run_patch(self.source);self.assertNotEqual(result.returncode,0)
        self.assertIn('aliases the source',result.stderr)
        self.assertEqual((self.source/self.relative).read_bytes(),self.bytes)
    def test_changed_builder_inventory_is_refused(self):
        target=self.snapshot/self.relative;text=target.read_text()
        before='let session = gpu_session(environment()?, path, ep)?;' if 'fn gpu_session(' in text else ('let session = cuda_session(environment()?, path)?;' if 'fn cuda_session(' in text else 'let session = Session::builder(env)')
        self.assertIn(before,text)
        target.write_text(text.replace(before,'let session = another_builder(env)',1))
        result=self.run_patch(self.snapshot);self.assertNotEqual(result.returncode,0)
        self.assertIn('Every ORT',result.stderr)
if __name__=='__main__':unittest.main()
