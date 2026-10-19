import copy
import tempfile
from pathlib import Path
import unittest
import workload
import panel
import patch_nr_result
from test_repairs import fixture,native_row,native_text
import test_panel

class SecondResult(unittest.TestCase):
    def test_schema_routes_actual_reference_not_ref_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=fixture(Path(tmp));m['arms']['ours']['protocol']='iqa-row-nr-v1'
            r=native_row(m);r.update(ref_score=999,nr_result_schema='paired-nr-v1',nr_scores=dict(brisque=1,brisque_ref=2),nr_observed_calls=11,nr_results_stable=True)
            p=workload.parse(native_text(m,r),'ours',m,True)
            self.assertEqual((p['score'],p['reference_score']),(1,2))
            for key,value in [('nr_result_schema','wrong'),('nr_results_stable',False),('nr_observed_calls',0),('nr_scores',{'brisque':1}),('nr_scores',{'brisque':2,'brisque_ref':2}),('nr_scores',{'brisque':1,'brisque_ref':float('nan')})]:
                bad=copy.deepcopy(r);bad[key]=value
                with self.assertRaises(ValueError):workload.parse(native_text(m,bad),'ours',m,True)
    def test_reference_parity_required_under_same_bound(self):
        q=test_panel.Qualification();q.setUp()
        self.assertEqual(panel.qualify_block(q.b,q.m),2)
        for value in (None,float('nan'),3):
            b=copy.deepcopy(q.b);b['arms'][0]['reference_score']=value
            with self.assertRaises(ValueError):panel.qualify_block(b,q.m)
    def test_overlay_preserves_call_count_and_first_distorted_field(self):
        source=(Path(__file__).resolve().parents[2]/'benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete.rs').read_text()
        adapted=patch_nr_result.patched(source)
        self.assertEqual(source.count('run_metric_display('),adapted.count('run_metric_display('))
        self.assertIn('row["ref_score"] = json!(first);',adapted)
        self.assertIn('nr_observer::record(&v)',adapted)
        with self.assertRaises(ValueError):patch_nr_result.patched(adapted)

class TraceReadiness(unittest.TestCase):
    def test_old_trace_never_implies_attribution(self):
        import gate_trace
        text='[zenbench gate] foreign pid=12 name="owned fixture" cpu_pct=12.5\n[zenbench gate] clean=false reason=Some(HeavyProcesses(4)) state=SystemState { heavy_process_count: 4 }\n'
        p=gate_trace.parse(text)
        self.assertEqual(p['foreign'][0]['pid'],12)
        self.assertFalse(p['attribution_ready'])
        self.assertEqual(len(p['gates']),1)
        with self.assertRaises(ValueError):gate_trace.parse('[zenbench gate] foreign malformed')
    def test_attribution_requires_exact_identity_time_count_and_real_receipt(self):
        import gate_trace
        p=dict(pid=12,start_ticks=40,pgid=12,sid=12,cgroup='/fixture',ownership_basis='outside owned tree',counted=True,owned=False,cpu_ticks_before=10,cpu_ticks_after=11,interval_ns=100)
        o=dict(snapshot_id='a',sample_monotonic_ns=100,checked_monotonic_ns=120,observation_age_ns=20,processes=[p],heavy_process_count=1)
        v=dict(schema=1,synthetic=False,observations=[o],**{k:'a'*64 for k in ('binary_sha256','config_sha256','source_sha256','trace_sha256','outer_receipt_sha256')})
        self.assertTrue(gate_trace.validate_attribution(v)) # Invented schema fixture only.
        for key,value in [('observation_age_ns',21),('heavy_process_count',4)]:
            bad=copy.deepcopy(v);bad['observations'][0][key]=value
            with self.assertRaises(ValueError):gate_trace.validate_attribution(bad)
        for key in ('start_ticks','cgroup','cpu_ticks_before'):
            bad=copy.deepcopy(v);del bad['observations'][0]['processes'][0][key]
            with self.assertRaises(ValueError):gate_trace.validate_attribution(bad)
        v['synthetic']=True
        with self.assertRaises(ValueError):gate_trace.validate_attribution(v)

class RustRouting(unittest.TestCase):
    def test_actual_patched_make_call_with_synthetic_cli(self):
        import subprocess
        source=(Path(__file__).resolve().parents[2]/'benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete.rs').read_text()
        adapted=patch_nr_result.patched(source)
        function=adapted[adapted.index('fn make_call('):adapted.index('\nfn main()')]
        stubs='''
mod nr_observer;
struct Value;
struct Rgb8Image;
#[derive(Clone,Copy,PartialEq)] enum MetricKind { Brisque }
enum GpuRuntime { Cpu }
fn parse_display(_: &str) -> Result<(), String> { Ok(()) }
fn run_metric_display(_: MetricKind, _: &Rgb8Image, _: &Rgb8Image, _: GpuRuntime, _: Option<&()>) -> Result<Vec<(&'static str,f64)>,String> { Ok(vec![("brisque",7.0),("brisque_ref",19.0)]) }
mod maps {
    pub type Call = Box<dyn FnMut() -> Result<f64,String>>;
    pub fn native(_: &str, _: &super::Rgb8Image, _: &super::Rgb8Image) -> Option<Call> { None }
    pub fn cvvdp_score(_: &super::Rgb8Image, _: &super::Rgb8Image, _: bool) -> Result<Call,String> { Err("synthetic only".into()) }
}
#[test] fn route() {
    static IMAGE: Rgb8Image = Rgb8Image;
    let mut call=make_call(&Value,"brisque",Some(MetricKind::Brisque),&IMAGE,&IMAGE,false).unwrap().unwrap();
    assert_eq!(call().unwrap(),7.0);
    assert_eq!(nr_observer::result().unwrap(),(7.0,19.0,1));
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'nr_observer.rs').write_bytes(Path(__file__).with_name('nr_observer.rs').read_bytes())
            (root/'test.rs').write_text(stubs+function)
            subprocess.run(['rustc','--edition=2024','--test',str(root/'test.rs'),'-o',str(root/'control')],check=True,capture_output=True)
            subprocess.run([str(root/'control')],check=True,capture_output=True)


class TraceRetention(unittest.TestCase):
    def test_command_retains_exact_raw_and_refuses_overwrite(self):
        import subprocess,sys,json,hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=b'[zenbench gate] clean=false reason=Some(HeavyProcesses(4)) state=fixture\n'
            log=root/'input.log';log.write_bytes(raw);out=root/'retained'
            cmd=[sys.executable,str(Path(__file__).with_name('gate_trace.py')),str(log),str(out)]
            subprocess.run(cmd,check=True,capture_output=True)
            self.assertEqual((out/'gate.log').read_bytes(),raw)
            parsed=json.loads((out/'parsed.json').read_text())
            self.assertEqual(parsed['raw_sha256'],hashlib.sha256(raw).hexdigest())
            self.assertFalse(parsed['attribution_ready'])
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            self.assertEqual((out/'gate.log').read_bytes(),raw)
