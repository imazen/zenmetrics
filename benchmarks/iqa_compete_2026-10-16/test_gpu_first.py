"""Inert CUDA first-boundary controls; no CUDA or scorer subprocess is invoked."""
import copy,contextlib,io,json,os,pathlib,sys,tempfile,unittest
from unittest.mock import patch
import compare_gpu_first_verified as gpu
import compare_classical_first_verified as cpu
import native_gpu_first_cell as native
from compare_canonical import sha
P=pathlib.Path
class Controls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='gpu-first-controls-');self.addCleanup(self.tmp.cleanup);self.root=P(self.tmp.name)
 def fixture(self,metric='cvvdp-gpu'):
  root=self.root/metric;root.mkdir();files=[]
  for name in ['native.txt','canonical.txt','pixels.rgb','extra.txt']:
   p=root/name;p.write_text('inert source');files.append(p)
  n,c,image,extra=files;assets={str(p):sha(p) for p in files};original=dict(id='inert',metric=metric,width=64,height=64,threads=1,reference=str(image),distorted=str(image),gpu_runtime='cuda',canonical=dict(label='inert',command=[str(c),'{samples}']),absolute_bound=None,relative_bound=0,matched_scope=True,scope='inert')
  source=root/'source.json';source.write_text(json.dumps(dict(binary=str(n),rust_source='inert',bindings=assets,cases=[original])))
  review=root/'review.json';review.write_text(json.dumps(dict(routes=[dict(metric=metric,classification='first_evaluation_candidate',canonical_emits_cold=True,canonical_cold_is_first_pair_evaluation=True,adapter_pair_evaluations_before_first_timed_observation=0,canonical_command=original['canonical']['command'])])))
  case=dict(copy.deepcopy(original),source_plan=str(source),source_case_id='inert',binary=str(n),rust_source='inert',first_call_scope=dict(matched=True,boundary=gpu.PHASE,construction='inert',parameter_io='inert',runtime='inert',teardown='inert',evidence=['inert']))
  required=[source,review,P(sys.executable),P('/usr/bin/time'),P('/usr/bin/timeout'),P('/usr/bin/flock'),P.home()/'work/claudehints/scripts/run-heavy',*[P(gpu.__file__).parent/x for x in gpu.HELPERS]];bindings=dict(assets,**{str(p):sha(p) for p in required});plan=root/'plan.json';plan.write_text(json.dumps(dict(phase=gpu.PHASE,seconds=5,review=str(review),bindings=bindings,cases=[case])))
  return dict(root=root,plan=plan,case=case,source=source,extra=extra)
 def test_all_five_explicit_cuda_names_accepted(self):
  for metric in ['butteraugli-gpu','ssim2-gpu','dssim-gpu','iwssim-gpu','cvvdp-gpu']:
   f=self.fixture(metric);plan,routes=gpu.verify(f['plan'],sha(f['plan']));self.assertFalse(gpu.scope_reasons(plan['cases'][0],routes[metric]))
 def test_existing_cpu_controller_keeps_gpu_refusal(self):
  f=self.fixture()
  with self.assertRaisesRegex(ValueError,'only reviewed classical CPU'):cpu.verify(f['plan'],sha(f['plan']))
 def test_cpu_and_disabled_gpu_names_refused(self):
  for metric in ['cvvdp','zensim-gpu']:
   f=self.fixture(metric)
   with self.assertRaisesRegex(ValueError,'only reviewed explicit CUDA'):gpu.verify(f['plan'],sha(f['plan']))
 def test_missing_or_non_cuda_runtime_refused(self):
  for number,value in enumerate([None,'cpu','wgpu']):
   f=self.fixture(['cvvdp-gpu','iwssim-gpu','dssim-gpu'][number]);plan=json.loads(f['plan'].read_text());plan['cases'][0]['gpu_runtime']=value;f['plan'].write_text(json.dumps(plan))
   with self.assertRaisesRegex(ValueError,'only reviewed explicit CUDA'):gpu.verify(f['plan'],sha(f['plan']))
 def test_every_binding_rehashed_on_every_invocation(self):
  f=self.fixture();plan=json.loads(f['plan'].read_text());expected=[str(f['plan']),*plan['bindings']]
  for unused in range(2):
   seen=[]
   def record(path):seen.append(str(path));return sha(path)
   with patch.object(gpu,'sha',side_effect=record):gpu.verify(f['plan'],sha(f['plan']))
   self.assertEqual(seen,expected)
 def test_native_adapter_forwards_and_observes_cuda(self):
  f=self.fixture();out=f['root']/'native-output';captured=[]
  def execute(command,env,directory,role,seconds):
   config=json.loads(P(command[1]).read_text());captured.append(config);row=dict(config,n=1,status='ok',cold_elapsed_ns=1,score=1)
   (directory/(role+'-stdout.log')).write_text('IQA_ROW '+json.dumps(row)+'\n');return 0,command
  with patch.object(native,'execute',side_effect=execute),contextlib.redirect_stdout(io.StringIO()):native.run(str(f['root']/'native.txt'),f['case'],out,5)
  self.assertEqual(captured[0]['gpu_runtime'],'cuda');self.assertEqual(captured[0]['timing_phase'],'cold')
 def test_native_adapter_refuses_unstated_device_without_execution(self):
  f=self.fixture();case=dict(f['case']);case.pop('gpu_runtime')
  with patch.object(native,'execute') as execute,self.assertRaisesRegex(ValueError,'explicit CUDA'):native.run(str(f['root']/'native.txt'),case,f['root']/'out',5)
  execute.assert_not_called()
 def test_controller_keeps_cuda_device_visible(self):
  f=self.fixture();seen=[]
  def execute(command,env,directory,role,seconds):
   seen.append(env);(directory/(role+'-stdout.log')).write_text('COLD 1 1\n');return 0,command
  with patch.object(gpu,'execute',side_effect=execute):gpu.run_arm(f['case'],'canonical',f['root']/'canonical-output',5)
  self.assertEqual(seen[0]['CUDA_VISIBLE_DEVICES'],'0')
 def test_tamper_after_first_arm_prevents_second_execution(self):
  f=self.fixture();calls=[]
  def arm(case,role,directory,seconds):calls.append(role);f['extra'].write_text('changed');return dict(ns=1,score=1)
  with patch.object(gpu,'run_arm',side_effect=arm):result=gpu.block(f['plan'],sha(f['plan']),'inert',0,f['root']/'block')
  self.assertEqual(calls,['canonical']);self.assertIn('admission_or_binding',result['errors']);self.assertIs(result['passed'],False)
if __name__=='__main__':unittest.main()
