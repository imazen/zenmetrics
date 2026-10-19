"""Regression controls for the independent review; no scorer is executed."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import containment
import panel
import runtime_closure
import seal_pilot
import workload
import bytecode_inventory

HERE = Path(__file__).resolve().parent


def fixture(root):
    def file(name, content='fixture'):
        path=root/name;path.write_text(content);return str(path)
    reference,distorted=file('reference.rgb'),file('distorted.rgb')
    work=dict(metric_name='brisque',width=512,height=512,threads=1,mode='score',timing_phase='warm',reference=reference,distorted=distorted)
    config=dict(work,only_mode='score',locked_process_accounting=True)
    binary=file('baseline');cell=file('cell.json',json.dumps(config));weights=file('weights')
    native=runtime_closure.closure([],runtime_closure.loader_environment())
    paths=[binary,cell,weights,sys.executable,'/usr/bin/taskset',str(HERE/'containment.py'),str(HERE/'brisque_oracle.py')]
    contention=file('contention.py');heavy=file('heavy');paths += [contention,heavy] + [str(HERE/name) for name in ('panel.py','workload.py','runtime_closure.py','bytecode_inventory.py','nr_validation.py')]
    return dict(schema=2,metric='brisque',shape=[512,512,1,'score'],scope=dict(ours='paired-NR',canonical='paired-NR'),
        workload=work,native_closure=native,bytecode_binding=bytecode_inventory.capture([],[str(root)]),artifacts=[seal_pilot.record(p) for p in paths],
        inputs=[seal_pilot.record(p) for p in [reference,distorted]],parity_abs_bound=.001,cpu=0,
        resources_script='unused-legacy-helper',controller_python=sys.executable,taskset='/usr/bin/taskset',containment_script=str(HERE/'containment.py'),contention_script=contention,run_heavy=heavy,
        arms=dict(ours=dict(build_identity='native-build',argv=[binary,cell],protocol='iqa-row',sample_count=10),
                  canonical=dict(build_identity='oracle-build',argv=[sys.executable,str(HERE/'brisque_oracle.py'),reference,distorted,weights],protocol='canonical-v3',adapter=workload.ADAPTER,pyiqa_version='test',version='pyiqa-test'+workload.VERSION_SUFFIX,sample_count=10)))


def native_row(m):
    return dict(m['workload'],declared_columns=['brisque','brisque_ref'],headline_declared_column='brisque',status='ok',only_mode='score',locked_process_accounting=True,n=10,score=1,
        zenbench_samples=[dict(execution_order=[0],elapsed_ns=[20],iterations=1,gate_clean=True) for _ in range(10)])


def native_text(m,row=None):
    return 'IQA_PROCESS {"pid":100,"pgid":100}\nIQA_ROW '+json.dumps(native_row(m) if row is None else row)+'\n'


def canonical_text(m):
    spec=m['arms']['canonical'];w=m['workload']
    obs=dict(w,adapter=workload.ADAPTER,adapter_sha256=panel.digest(spec['argv'][1]),weights_sha256=panel.digest(spec['argv'][4]),input_sha256={k:panel.digest(w[k]) for k in ('reference','distorted')},sample_count=10,timing_boundary=workload.BOUNDARY)
    lines=['IQA_PROCESS {"pid":101,"pgid":101}','WORKLOAD '+json.dumps(obs),'VERSION '+spec['version']]
    lines += ['SAMPLE '+json.dumps(dict(sample_id=i,ns=10,score=1,reference_score=2,nr_observed_calls=i+1,nr_results_stable=True,timing_boundary=workload.BOUNDARY)) for i in range(1,11)]
    return '\n'.join(lines)+'\n'


class ParserRepairs(unittest.TestCase):
    def test_actual_run_arm_parser_rejects_wrong_native_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=fixture(root)
            for key,value in [('metric_name','butteraugli'),('width',64),('height',64),('threads',8),('mode','score+map'),('only_mode','score+map'),('timing_phase','cold'),('reference','wrong'),('distorted','wrong'),('headline_declared_column','brisque_ref'),('rayon_threads_observed',8),('input_sha256',{}),('impl_binary_sha','wrong')]:
                with self.subTest(key=key):
                    out=root/key;out.mkdir();row=native_row(m);row[key]=value
                    def emit(command,output,env,receipt):
                        (out/'ours.contention.json').write_text('{"contaminated":false}')
                        output.write(native_text(m,row));return 0
                    with patch.object(containment,'execute',side_effect=emit), patch.object(panel.subprocess,'run',side_effect=lambda command,**kw: subprocess.CompletedProcess(command,emit(command,kw['stdout'],kw.get('env'),None))):
                        with self.assertRaises(ValueError):panel.run_arm('ours',m,'block',out)
            result=workload.parse(native_text(m),'ours',m,True)
            self.assertTrue(result['observed_validated'])
            self.assertIsNone(result['reference_score'])

    def test_native_counts_arm_order_process_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=fixture(Path(tmp))
            bad=[]
            r=native_row(m);r['n']=9;bad.append(native_text(m,r))
            r=native_row(m);r['zenbench_samples'].pop();bad.append(native_text(m,r))
            r=native_row(m);r['zenbench_samples'][0]['execution_order']=[1];bad.append(native_text(m,r))
            bad += [native_text(m)*2,native_text(m).replace('"pid":100','"pid":-1')]
            for text in bad:
                with self.assertRaises(ValueError):workload.parse(text,'ours',m,True)

    def test_canonical_identity_sequence_and_both_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=fixture(Path(tmp));text=canonical_text(m)
            parsed=workload.parse(text,'canonical',m,True)
            self.assertEqual(parsed['reference_score'],2)
            cases=[text.replace('"sample_id": 2','"sample_id": 1'),text.replace('"threads": 1','"threads": 8'),
                text.replace('VERSION pyiqa-test','VERSION other'),text+'VERSION other\n',
                '\n'.join(text.splitlines()[:-1]),text.replace('"reference_score": 2','"reference_score": NaN'),
                text.replace('IQA_PROCESS','MISSING_PROCESS')]
            for bad in cases:
                with self.assertRaises(ValueError):workload.parse(bad,'canonical',m,True)

    def test_required_path_config_and_adapter_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=fixture(Path(tmp));panel.verify(m)
            for key in ('contention_script','containment_script','controller_python','run_heavy','taskset'):
                bad=copy.deepcopy(m);bad['artifacts']=[r for r in bad['artifacts'] if r['path']!=bad[key]]
                with self.assertRaises(ValueError):panel.verify(bad)
            for key in ('ours','canonical'):
                for path in m['arms'][key]['argv'][:2]:
                    bad=copy.deepcopy(m);bad['artifacts']=[r for r in bad['artifacts'] if r['path']!=path]
                    with self.assertRaises(ValueError):panel.verify(bad)
            bad=copy.deepcopy(m);bad['arms']['canonical']['adapter']='vmaf'
            with self.assertRaises(ValueError):panel.verify(bad)
            bad=copy.deepcopy(m);bad['metric']='vsi'
            with self.assertRaises(ValueError):panel.verify(bad)
            config=Path(m['arms']['ours']['argv'][1]);value=json.loads(config.read_text());value['threads']=8;config.write_text(json.dumps(value))
            for r in m['artifacts']:
                if r['path']==str(config):r['sha256']=panel.digest(config)
            with self.assertRaises(ValueError):panel.verify(m)


class ReceiptRepairs(unittest.TestCase):
    def exercise(self,error,post=False):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);(home/'tmp').mkdir();m=fixture(home)
            calls=[]
            def run(name,manifest,bid,out):
                calls.append(name)
                if name=='canonical' and not post:raise error
                arm=dict(arm=name,block_id=bid,build_identity=m['arms'][name]['build_identity'],input_identity=panel.identity(m['inputs']),scope=m['scope'][name],exit_code=0,contaminated=False,
                         **workload.parse(native_text(m) if name=='ours' else canonical_text(m),name,m,True))
                if name=='canonical':self.assertTrue((out/'ours.arm.json').exists())
                return arm
            checks=[None,error] if post else [None]
            with patch.object(panel.Path,'home',return_value=home),patch.object(panel,'verify',side_effect=checks),patch.object(panel,'run_arm',side_effect=run):
                blocks=panel.acquire(m,home/'out',4)
            b=blocks[0];path=home/'out'/b['block_id']
            self.assertTrue((path/'block.json').exists());self.assertTrue((home/'out'/'summary.json').exists())
            self.assertEqual(b['status'],'rejected');self.assertEqual(b['error']['type'],type(error).__name__)
            self.assertEqual(len(blocks),1);self.assertEqual(panel.summarize(blocks,m)['accepted_blocks'],0)
            self.assertEqual(len(b['arms']),2 if post else 1)
            self.assertEqual(b['arm_states']['canonical'],'completed' if post else 'failed')
    def test_parse_launch_interruption_and_deleted_artifact_receipts(self):
        for exc in [json.JSONDecodeError('malformed','x',0),FileNotFoundError('launch absent'),KeyboardInterrupt('owned interrupt')]:
            with self.subTest(kind=type(exc).__name__):self.exercise(exc)
        self.exercise(FileNotFoundError('deleted artifact'),post=True)

    def test_stop_between_completed_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);(home/'tmp').mkdir();m=fixture(home)
            def arm(name,manifest,bid,out):
                value=dict(arm=name,block_id=bid,build_identity=m['arms'][name]['build_identity'],input_identity=panel.identity(m['inputs']),scope=m['scope'][name],exit_code=0,contaminated=False,
                           **workload.parse(native_text(m) if name=='ours' else canonical_text(m),name,m,True))
                if name=='canonical':(out.parent/'STOP').write_text('boundary cancellation')
                return value
            with patch.object(panel.Path,'home',return_value=home),patch.object(panel,'run_arm',side_effect=arm):
                blocks=panel.acquire(m,home/'out',4)
            self.assertEqual(len(blocks),1)
            self.assertEqual(json.loads((home/'out'/'cancelled.json').read_text())['completed_blocks'],1)

    def test_initial_missing_artifact_retains_not_started_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);(home/'tmp').mkdir();m=fixture(home)
            Path(m['arms']['ours']['argv'][0]).unlink()
            with patch.object(panel.Path,'home',return_value=home):blocks=panel.acquire(m,home/'out',1)
            self.assertEqual(blocks[0]['arm_states'],dict(ours='not-started',canonical='not-started'))
            self.assertEqual(blocks[0]['error']['type'],'FileNotFoundError')


class RuntimeRepairs(unittest.TestCase):
    def test_transitive_dependency_mutation_and_shadow_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[root/n for n in ('extension','direct','libstdc++.so','replacement')]
            for p in paths:p.write_text(p.name)
            edges={str(paths[0]):[str(paths[1])],str(paths[1]):[str(paths[2])],str(paths[2]):[],str(paths[3]):[]}
            def inspect(path,loader):return dict(dependencies=edges[path],dynamic_search=['RUNPATH='+str(root)])
            binding=runtime_closure.closure([paths[0]],runtime_closure.loader_environment(),inspect)
            self.assertIn(str(paths[2]),[r['path'] for r in binding['files']])
            runtime_closure.verify(binding,inspect)
            paths[2].write_text('changed')
            with self.assertRaises(ValueError):runtime_closure.verify(binding,inspect)
            paths[2].write_text(paths[2].name);edges[str(paths[1])]=[str(paths[3])]
            with self.assertRaises(ValueError):runtime_closure.verify(binding,inspect)

    def test_loader_environment_change(self):
        binding=runtime_closure.closure([],runtime_closure.loader_environment())
        with patch.dict(os.environ,{'LD_LIBRARY_PATH':'/owned/new/path'}):
            with self.assertRaises(ValueError):runtime_closure.verify(binding)

class SealerLayout(unittest.TestCase):
    def test_committed_directory_with_tiny_fake_environment(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);build=root/'build';source=build/'work/zenmetrics'
            (build/'target/release').mkdir(parents=True)
            shutil.copy2('/usr/bin/true',build/'target/release/iqa-compete')
            (source/'ci').mkdir(parents=True)
            (source/'Cargo.lock').write_text('fake lock\n');(source/'ci/sibling-pins.tsv').write_text('fake pins\n')
            old=source/'benchmarks/iqa_compete_2026-10-16';old.mkdir(parents=True)
            (old/'contention.py').write_text('# fake supervisor, never executed\n')
            for name in ['build-resource.log','rustc.txt','cargo.txt','features.txt','metadata.json','build-command.txt','model-preload.meta.json','luma-maps.meta.json']:
                (build/name).write_text('{}')
            env=root/'env';(env/'bin').mkdir(parents=True);(env/'lib').mkdir();stdlib=root/'stdlib';stdlib.mkdir();(stdlib/'fake.py').write_text('# fake stdlib\n')
            python=env/'bin/python'
            python.write_text('#!'+sys.executable+'\nimport sys,json\nif "stdlib" in sys.argv[-1]: print(json.dumps({"version":"fake", "paths":[], "cache_prefix":None, "stdlib":'+repr(str(stdlib))+'}))\nelse: print(json.dumps([["pyiqa","test"]]))\n')
            python.chmod(0o755)
            reference=root/'r.rgb';distorted=root/'d.rgb';weights=root/'weights'
            reference.write_bytes(bytes(512*512*3));distorted.write_bytes(bytes(512*512*3));weights.write_text('fake')
            output=root/'sealed'
            argv=[str(HERE/'seal_pilot.py'),str(build),str(output),'--python',str(python),'--reference',str(reference),'--distorted',str(distorted),'--weights',str(weights),'--revision','synthetic']
            with patch.object(sys,'argv',argv):seal_pilot.main()
            m=json.loads((output/'manifest.json').read_text());panel.verify(m)
            self.assertIn(str(HERE/'build_baseline.sh'),[r['path'] for r in m['artifacts']])
            self.assertEqual(m['arms']['ours']['build_identity'],panel.identity(json.loads((output/'baseline.json').read_text())))
            # Fake source layout and true executable are fixtures, never scored.
            self.assertEqual(m['workload']['threads'],1)


class OwnedContainment(unittest.TestCase):
    def test_timeout_reaps_term_ignoring_descendant_and_escaped_session(self):
        for escape in (False,True):
            with self.subTest(escape=escape),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);pidfile=root/'pid';script=root/'owned.py'
                script.write_text('import os,signal,time,sys\npid=os.fork()\nif pid==0:\n '+('os.setsid()\n ' if escape else '')+'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n open(sys.argv[1],"w").write(str(os.getpid()))\n time.sleep(30)\nelse:\n time.sleep(30)\n')
                with (root/'log').open('w') as log:
                    code=containment.execute([sys.executable,str(script),str(pidfile)],log,os.environ.copy(),root/'receipt.json',timeout=.3,grace=.2)
                pid=int(pidfile.read_text());receipt=json.loads((root/'receipt.json').read_text())
                self.assertEqual(code,124);self.assertEqual(receipt['status'],'empty');self.assertEqual(receipt['remaining'],{})
                self.assertFalse(Path(f'/proc/{pid}').exists())
                self.assertTrue(any(e['pid']==pid and e['signal']==signal.SIGKILL for e in receipt['signals']))

    def test_interruption_cleans_children_before_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=root/'interrupt.py';pidfile=root/'pid'
            script.write_text('import os,signal,time,sys\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nopen(sys.argv[1],"w").write(str(os.getpid()))\nos.kill(os.getppid(),signal.SIGTERM)\ntime.sleep(30)\n')
            with (root/'log').open('w') as log:
                with self.assertRaises(KeyboardInterrupt):
                    containment.execute([sys.executable,str(script),str(pidfile)],log,os.environ.copy(),root/'receipt.json',timeout=2,grace=.2)
            self.assertFalse(Path('/proc/'+pidfile.read_text()).exists())
            self.assertEqual(json.loads((root/'receipt.json').read_text())['status'],'empty')

    def test_direct_child_exit_does_not_leave_descendant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=root/'exits.py';pidfile=root/'pid'
            script.write_text('import os,signal,time,sys\npid=os.fork()\nif pid==0:\n signal.signal(signal.SIGTERM,signal.SIG_IGN)\n open(sys.argv[1],"w").write(str(os.getpid()))\n time.sleep(30)\nelse:\n time.sleep(.1)\n')
            with (root/'log').open('w') as log:
                containment.execute([sys.executable,str(script),str(pidfile)],log,os.environ.copy(),root/'receipt.json',timeout=2,grace=.2)
            self.assertFalse(Path('/proc/'+pidfile.read_text()).exists())
            self.assertEqual(json.loads((root/'receipt.json').read_text())['remaining'],{})

class ExecutionBoundary(unittest.TestCase):
    def test_valid_emitted_rows_and_forged_process_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=fixture(root)
            for name,text,pid in [('ours',native_text(m),100),('canonical',canonical_text(m),101)]:
                for forged in (False,True):
                    out=root/(name+str(forged));out.mkdir()
                    def emit(command,output,env,receipt):
                        output.write(text)
                        (out/(name+'.contention.json')).write_text('{"contaminated":false}')
                        containment.write(receipt,dict(status='empty',remaining={},owned=[dict(pid=pid+(1 if forged else 0),pgid=pid,start=123,exe=str(Path(m['arms'][name]['argv'][0]).resolve()))]))
                        return 0
                    with patch.object(containment,'execute',side_effect=emit):
                        if forged:
                            with self.assertRaises(ValueError):panel.run_arm(name,m,'block',out)
                        else:
                            result=panel.run_arm(name,m,'block',out)
                            self.assertTrue(result['observed_validated'])
                            self.assertEqual(result['observed_process']['start'],123)
                    self.assertTrue((out/(name+'.execution.json')).exists())

    def test_real_parser_failure_retains_raw_exit_and_terminal_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);(home/'tmp').mkdir();m=fixture(home)
            def emit(command,output,env,receipt):
                output.write('IQA_PROCESS {"pid":100,"pgid":100}\nIQA_ROW {malformed\n')
                (receipt.parent/'ours.contention.json').write_text('{"contaminated":false}')
                containment.write(receipt,dict(status='empty',remaining={},owned=[]));return 17
            with patch.object(panel.Path,'home',return_value=home),patch.object(containment,'execute',side_effect=emit):
                blocks=panel.acquire(m,home/'out',1)
            block=blocks[0];out=home/'out'/block['block_id']
            self.assertEqual(block['error']['type'],'JSONDecodeError')
            self.assertEqual(block['arm_states'],dict(ours='failed',canonical='not-started'))
            self.assertEqual(json.loads((out/'ours.execution.json').read_text())['exit_code'],17)
            self.assertIn('ours.log',[f['path'] for f in block['attempt_files']])

    def test_owned_timeout_is_empty_while_both_locks_still_held(self):
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);(home/'tmp').mkdir();m=fixture(home)
            pidfile=home/'pid';script=home/'child.py'
            script.write_text('import os,signal,time,sys\npid=os.fork()\nif pid==0:\n signal.signal(signal.SIGTERM,signal.SIG_IGN)\n open(sys.argv[1],"w").write(str(os.getpid()))\n time.sleep(30)\nelse:\n time.sleep(30)\n')
            def arm(name,manifest,bid,out):
                with (out/'owned.log').open('w') as log:
                    code=containment.execute([sys.executable,str(script),str(pidfile)],log,os.environ.copy(),out/'owned.containment.json',timeout=.3,grace=.2)
                self.assertEqual(code,124)
                self.assertFalse(Path('/proc/'+pidfile.read_text()).exists())
                for lock in ('lilith-heavy.lock','lilith-wsl-heavy.lock'):
                    with (home/'tmp'/lock).open('a') as fd:
                        with self.assertRaises(BlockingIOError):fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                raise TimeoutError('owned timeout fully contained')
            with patch.object(panel.Path,'home',return_value=home),patch.object(panel,'run_arm',side_effect=arm):
                blocks=panel.acquire(m,home/'out',1)
            self.assertEqual(blocks[0]['error']['type'],'TimeoutError')
            for lock in ('lilith-heavy.lock','lilith-wsl-heavy.lock'):
                with (home/'tmp'/lock).open('a') as fd:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_real_missing_executable_preserves_containment_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with (root/'log').open('w') as log:
                with self.assertRaises(RuntimeError):containment.execute([str(root/'absent')],log,os.environ.copy(),root/'containment.json',timeout=.2,grace=.1)
            result=json.loads((root/'containment.json').read_text())
            self.assertEqual(result['status'],'empty');self.assertEqual(result['remaining'],{})
            self.assertEqual(result['error']['type'],'FileNotFoundError')

class RealElfClosure(unittest.TestCase):
    def test_transitive_elf_mutation_and_loader_search_change(self):
        # Three tiny owned libraries, never loaded as metric/scorer code.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'leaf.c').write_text('int leaf(void) { return 1; }\n')
            (root/'middle.c').write_text('extern int leaf(void); int middle(void) { return leaf(); }\n')
            (root/'root.c').write_text('extern int middle(void); int top(void) { return middle(); }\n')
            for name,dependency in [('leaf',None),('middle','leaf'),('root','middle')]:
                command=['/usr/bin/cc','-shared','-fPIC',str(root/(name+'.c')),'-o',str(root/('lib'+name+'.so')),'-Wl,-rpath,$ORIGIN','-L'+str(root)]
                if dependency:command.append('-l'+dependency)
                subprocess.run(command,check=True,capture_output=True)
            binding=runtime_closure.closure([root/'libroot.so'],runtime_closure.loader_environment())
            self.assertIn(str(root/'libleaf.so'),[r['resolved'] for r in binding['files']])
            runtime_closure.verify(binding)
            with (root/'libleaf.so').open('ab') as f:f.write(b'owned mutation control')
            with self.assertRaises(ValueError):runtime_closure.verify(binding)
