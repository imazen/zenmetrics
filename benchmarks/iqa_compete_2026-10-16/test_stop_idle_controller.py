"""Priority gates preserve completed results held by a sleeping controller."""
import os,pathlib,signal,subprocess,sys,tempfile,time,unittest
P=pathlib.Path
HELPER=P(__file__).with_name('stop_idle_controller.py').resolve()
class IdleControls(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=P.home()/'tmp');self.addCleanup(self.temp.cleanup)
  self.root=P(self.temp.name)
 def launch(self,text,name='run.py'):
  script=self.root/name;script.write_text(text)
  process=subprocess.Popen([sys.executable,str(script),str(self.root)])
  def cleanup():
   if process.poll() is None:
    os.kill(process.pid,signal.SIGCONT);process.terminate();process.wait(timeout=5)
  self.addCleanup(cleanup)
  return script,process
 def invoke(self,script,process):
  return subprocess.run([sys.executable,str(HELPER),str(process.pid),str(script),str(self.root),str(self.root/'proof.json'),'--cooldown-waiter'],capture_output=True,text=True)
 def wait_sleep(self,process):
  for _ in range(300):
   if P(f'/proc/{process.pid}/wchan').read_text().strip()=='hrtimer_nanosleep':return
   time.sleep(.01)
  self.fail('Fixture did not enter its sleep')
 def test_cooldown_suspension_retains_result_until_continue(self):
  script,process=self.launch('import pathlib,signal,sys,time\nresult="retained"\nresumed=[False]\nsignal.signal(signal.SIGCONT,lambda *_:resumed.__setitem__(0,True))\nwhile not resumed[0]:time.sleep(.05)\npathlib.Path(sys.argv[1],"finished").write_text(result)\n')
  self.wait_sleep(process);result=self.invoke(script,process)
  self.assertEqual(result.returncode,0,result.stderr)
  self.assertIn('\nState:\tT',P(f'/proc/{process.pid}/status').read_text())
  self.assertIsNone(process.poll());self.assertFalse((self.root/'finished').exists())
  os.kill(process.pid,signal.SIGCONT);self.assertEqual(process.wait(timeout=5),0)
  self.assertEqual((self.root/'finished').read_text(),'retained')
 def test_active_child_is_preserved(self):
  script,process=self.launch('import subprocess,sys\np=subprocess.Popen([sys.executable,"-c","import time;time.sleep(1)"])\np.wait()\n')
  for _ in range(300):
   if P(f'/proc/{process.pid}/task/{process.pid}/children').read_text().strip():break
   time.sleep(.01)
  result=self.invoke(script,process);self.assertNotEqual(result.returncode,0)
  self.assertFalse((self.root/'proof.json').exists())
  self.assertEqual(process.wait(timeout=5),0)
 def test_unrecognized_sleeping_program_is_preserved(self):
  script,process=self.launch('import time;time.sleep(1)\n',name='worker.py')
  self.wait_sleep(process);result=self.invoke(script,process)
  self.assertNotEqual(result.returncode,0);self.assertFalse((self.root/'proof.json').exists())
  self.assertNotIn('\nState:\tT',P(f'/proc/{process.pid}/status').read_text())
if __name__=='__main__':unittest.main()
