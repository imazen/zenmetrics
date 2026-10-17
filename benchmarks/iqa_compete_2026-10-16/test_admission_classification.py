"""Launch admission refusals remain pending; executed metric failures do not."""
import pathlib,shutil,tempfile,unittest
from run import admission_reason,prior_admission_reason
P=pathlib.Path
class Classification(unittest.TestCase):
 def setUp(self):
  scratch=P.home()/'tmp';scratch.mkdir(exist_ok=True)
  self.root=P(tempfile.mkdtemp(prefix='iqa-admission-classify-',dir=scratch));self.addCleanup(shutil.rmtree,self.root)
 def test_wrapper_memory_refusal_is_admission(self):
  self.assertEqual(admission_reason('run-heavy: only 1914 MiB available — refusing to launch (free memory first)\n'),'host_memory')
 def test_load_refusal_is_admission(self):
  self.assertEqual(admission_reason('Error: "busy load [24.28, 6.39, 3.98]"'),'host_load')
 def test_orphaned_state_is_not_a_metric_outcome(self):
  self.assertEqual(admission_reason('benchmark admission failed: orphaned running reservation 746'),'zenbench_orphaned_reservation')
 def test_executed_metric_error_is_not_admission(self):
  self.assertIsNone(admission_reason('Error: invalid input geometry'))
 def test_saved_refusal_can_resume_from_its_log(self):
  path=self.root/'attempt.log';path.write_text('run-heavy: only 1882 MiB available — refusing to launch (free memory first)')
  self.assertEqual(prior_admission_reason(dict(status='skipped:process_exit_3',log_path=str(path))),'host_memory')
 def test_legacy_log_pointer_is_preserved(self):
  path=self.root/'legacy.log';path.write_text('run-heavy: only 1882 MiB available — refusing to launch')
  self.assertEqual(prior_admission_reason(dict(status='skipped:process_exit_3',notes='No completed timing; full log '+str(path))),'host_memory')
 def test_success_does_not_need_refusal_inspection(self):
  self.assertIsNone(prior_admission_reason(dict(status='ok')))
 def test_missing_refusal_evidence_is_an_error(self):
  with self.assertRaises(FileNotFoundError):prior_admission_reason(dict(status='skipped:process_exit_3',log_path=str(self.root/'missing')))
if __name__=='__main__':unittest.main()
