"""Inert protocols/files only. Executors and runtime captures are mocked."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('hdr_numerical',HERE/'hdr_original_grid_numerical.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def native_text(config, value=1.0):
    bits=struct.pack('>d',value).hex()
    emitted={k:val for k,val in config.items() if k!='probe_only'}
    row=dict(emitted,status='ok',mode='score',model_preload_adapter=False,
        rayon_threads_observed=config['threads'],score=value,
        score_bits=bits,self_warmup_score=value,ref_score=value,map_score_matches_scalar_bits=True,
        n=10,round_means_ns=[1]*10,zenbench_samples=[dict(gate_clean=True)]*10)
    return 'IQA_WARMUP '+json.dumps(dict(completed=True,score_bits=bits))+'\nIQA_ROW '+json.dumps(row)+'\n'


def original_text(value=1.0):
    return ('SAMPLE 1 '+repr(value)+'\nSCORE_HEX '+struct.pack('>d',value).hex()+'\n')*5


class FakeWorker:
    def __init__(self, real):
        self.scalar_bits=real.scalar_bits;self.files_under=real.files_under
        self.invariant=real.invariant;self.checked_seal=real.checked_seal

    def context(self, root, expected):
        cohort=v.pinned(root/'cohort.json',expected);v.verify(cohort['bindings'])
        plan=v.pinned(cohort['parent_plan'],v.PARENT_SHA);v.verify(plan['bindings'])
        if list((Path(cohort['parent_plan']).parent/'execution-cwd').iterdir()):
            raise ValueError('nonempty execution cwd')
        if v.load(cohort['case_specs'])!=plan['cases']:
            raise ValueError('original case inventory differs')
        return cohort,plan

    def seal(self, root, expected, preflight_sha):
        cohort,plan=self.context(root,expected)
        preflight=v.pinned(cohort['preflight_receipt'],preflight_sha)
        if preflight['passed'] is not True:
            raise ValueError('preflight must pass')
        v.write(root/'preflight-seal.json',dict(passed=True,controls=9,cohort_sha256=expected,
            parent_plan_sha256=cohort['parent_plan_sha256'],preflight_receipt=cohort['preflight_receipt'],
            preflight_sha256=preflight_sha,artifacts=self.files_under(Path(cohort['preflight_receipt']).parent),
            baseline=FakeExecutor.baseline(plan['octave'])))


class FakeExecutor:
    def __init__(self):
        self.calls=[];self.fail_native=False;self.drift=False

    @staticmethod
    def baseline(binary):
        return dict(files={binary:dict(sha256=v.sha(binary))},dependencies={},environment={})

    def capture(self, plan, bundle, directory, env):
        directory.mkdir()
        result=dict(self.baseline(plan['octave']),environment={'OMP_NUM_THREADS':env['OMP_NUM_THREADS']})
        if self.drift:result['files']={'changed':{}}
        v.write(directory/'snapshot.json',result)
        return result

    def execute(self, command, env, root, name, seconds):
        self.calls.append(command)
        native=command[-1].endswith('-native.json')
        text=native_text(v.load(command[-1])) if native else original_text()
        code=124 if native and self.fail_native else 0
        (root/(name+'-stdout.log')).write_text(text)
        (root/(name+'-stderr.log')).write_text('timeout\n' if code else '')
        (root/(name+'-memory.txt')).write_text('inert synthetic memory receipt\n')
        v.write(root/(name+'-command.json'),['/usr/bin/time','-v','-o',str(root/(name+'-memory.txt')),
            '/usr/bin/timeout','--kill-after=20s',str(seconds)+'s',*command])
        return dict(returncode=code,stdout_sha256=v.sha(root/(name+'-stdout.log')),
                    stderr_sha256=v.sha(root/(name+'-stderr.log')))


class Fixture:
    def __init__(self, base, real_worker):
        self.base=base;self.parent=base/'parent';self.parent.mkdir()
        (self.parent/'execution-cwd').mkdir()
        self.binary=self.file('native',b'INERT native bytes')
        self.octave=self.file('octave',b'INERT octave bytes')
        self.source=self.file('native-source/opaque.json',b'{ deliberately malformed source JSON')
        self.worker=FakeWorker(real_worker);self.executor=FakeExecutor()
        cases=[];inventory=[];bindings={str(self.binary):v.sha(self.binary)}
        body="result=hdrvdp();score=result.Q;\nclear reference distorted result;\n\n"
        body_sha=__import__('hashlib').sha256(body.encode()).hexdigest()
        for metric in ('hdrvdp','hdrvdp3'):
            for pair in range(12):
                reference=self.file(f'pair{pair}-reference.rgb',b'\x01'*3)
                distorted=self.file(f'pair{pair}-distorted.rgb',b'\x02'*3)
                for threads in (1,28):
                    case_id=f'original12-{metric}-{pair}-t{threads}-warm-current40'
                    script=self.file(f'parent/{case_id}.m',('for iteration=1:5\nclock=tic();\n'+body+
                        'elapsed=toc(clock);\nend\n').encode())
                    config=dict(width=1,height=1,threads=threads,reference=str(reference),distorted=str(distorted),
                        metric_name=metric,only_mode='score',timing_phase='warm',probe_only=False,
                        required_samples=5,sample_target_ns=10000000,chunk_budget_ms=240000,locked_process_accounting=True)
                    cfg=self.doc(f'{case_id}-native.json',config)
                    command=[str(self.octave),'--no-gui','--no-window-system','--no-init-file','--no-history','--quiet',str(script)]
                    case=dict(id=case_id,panel='original12',phase='warm',metric=metric,width=1,height=1,
                        threads=threads,samples=5,reference=str(reference),distorted=str(distorted),binary=str(self.binary),
                        native_config=str(cfg),native={'command':[str(self.binary),str(cfg)]},
                        arms={'default':{'command':command}},environment={k:str(threads) for k in v.THREADS},
                        evaluator_body_sha256=body_sha,absolute_bound=0.005 if metric=='hdrvdp' else None,
                        relative_bound=0,matched_scope=False)
                    for path in (reference,distorted,script,cfg):bindings[str(path)]=v.sha(path)
                    cases.append(case)
                    row={k:case[k] for k in ('id','metric','width','height','threads','evaluator_body_sha256','absolute_bound','relative_bound')}
                    row.update(reference=dict(path=str(reference),sha256_from_frozen_binding=v.sha(reference)),
                        distorted=dict(path=str(distorted),sha256_from_frozen_binding=v.sha(distorted)),
                        reference_default_command=command,native_command=case['native']['command'])
                    inventory.append(row)
        old=self.doc('parent/original12-warm.plan.json',dict(exact_original_call_gate='OLD canonical/direct twice',
            native_thread_gate='OLD exact first/warm1/28'))
        bindings[str(old)]=v.sha(old)
        self.plan=self.doc('parent/plan.json',dict(cases=cases,bindings=bindings,headless=True,
            expected_control_order=['default','resident','resident','default'],
            rust_source='f'*40,binary_sha256=v.sha(self.binary),octave=str(self.octave)))
        self.inventory=self.doc('selected48.json',dict(parent_plan=str(self.plan),parent_plan_sha256=v.sha(self.plan),cases=inventory))
        self.preflight=self.doc('preflight/result.json',dict(passed=True))
        proof=dict(source_revision='f'*40,binary_sha256=v.sha(self.binary))
        build=self.doc('build.json',dict(proof,adapters_applied=False,features=['compete','metrics-parallel','zenmetrics-cli/nr']))
        verified=self.doc('verified.json',proof)
        repro=self.doc('repro.json',dict(proof,source_files={'opaque.json':v.sha(self.source)}))
        inert_wrapper=self.file('wrapper',b'inert wrapper, never executed')
        self.admission=dict(wrapper=str(inert_wrapper),locks=[str(base/'lock1'),str(base/'lock2')],memory='12G',
                            python=__import__('sys').executable,
                            tools={str(inert_wrapper):v.sha(inert_wrapper),
                                   __import__('sys').executable:v.sha(__import__('sys').executable),
                                   '/usr/bin/flock':v.sha('/usr/bin/flock')})
        self.request=self.doc('request.json',dict(schema=1,inventory=str(self.inventory),parent_plan=str(self.plan),
            preflight_receipt=str(self.preflight),native_source_root=str(self.source.parent),admission=self.admission,
            native_evidence={name:dict(path=str(path),sha256=v.sha(path))
                             for name,path in [('build',build),('verified',verified),('reproducible',repro)]},
            sources={str(Path(v.__file__).resolve()):v.sha(v.__file__)}))
        self.root=base/'output';self.cases=cases

    def file(self,name,data):
        path=self.base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return path

    def doc(self,name,value):
        path=self.base/name;path.parent.mkdir(parents=True,exist_ok=True);v.write(path,value);return path

    def prepare(self):
        v.prepare(self.request,v.sha(self.request),self.root)
        self.cohort_sha=v.sha(self.root/'cohort.json')

    def audit(self):
        self.prepare();v.audit(self.root,self.cohort_sha);self.setup_sha=v.sha(self.root/'setup.json')

    def cell(self,index=0):
        c=self.cases[index]
        with patch.dict(os.environ,{k:str(c['threads']) for k in v.THREADS}):
            return v.cell(self.root,self.cohort_sha,self.setup_sha,c['id'],self.admission)


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory=HERE/'helpers' if (HERE/'helpers').is_dir() else HERE
        cls.real_worker,_=v.modules(directory)

    def setUp(self):
        scratch=Path(os.environ.get('TMPDIR',str(Path.home()/'tmp/hdr-numerical-tests')))
        scratch.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=scratch);self.addCleanup(self.temp.cleanup)
        self.f=Fixture(Path(self.temp.name),self.real_worker)
        self.patches=patch.multiple(v,HELPERS={},INVENTORY_SHA=v.sha(self.f.inventory),PARENT_SHA=v.sha(self.f.plan),
            PREFLIGHT_SHA=v.sha(self.f.preflight),BINARY_SHA=v.sha(self.f.binary),SOURCE='f'*40)
        self.patches.start();self.addCleanup(self.patches.stop)
        self.module_patch=patch.object(v,'modules',return_value=(self.f.worker,self.f.executor))
        self.module_patch.start();self.addCleanup(self.module_patch.stop)

    def config(self):
        return v.load(self.f.cases[0]['native_config'])

    def mutate_native(self, mutate):
        config=self.config();text=native_text(config);row=json.loads(text.split('IQA_ROW ')[1]);mutate(row)
        return text.split('IQA_ROW ')[0]+'IQA_ROW '+json.dumps(row)+'\n'

    def test_complete48_and96_inert_children(self):
        self.f.audit()
        for i in range(48):self.assertEqual(self.f.cell(i),0)
        self.assertEqual(len(self.f.executor.calls),96)
        self.assertEqual(v.finish(self.f.root,self.f.cohort_sha,self.f.setup_sha),0)
        summary=v.load(self.f.root/'result.json')
        self.assertEqual(summary['complete_observations'],48)
        self.assertEqual(summary['v2_bound_pass_count'],24)
        self.assertEqual(summary['v3_unbounded_observations'],24)
        self.assertEqual(summary['old_gates']['native_thread_gate']['status'],'incomplete')
        self.assertIsNone(summary['ratio']);self.assertFalse(summary['matched_scope'])

    def test_opaque_malformed_source_json_is_not_parsed(self):
        self.f.prepare()  # Malformed native-source/opaque.json is a hashed asset.

    def test_source_config_input_and_preflight_drift(self):
        for path in (self.f.source,Path(self.f.cases[0]['native_config']),
                     Path(self.f.cases[0]['reference']),self.f.preflight):
            with self.subTest(path=path):
                old=path.read_bytes();path.write_bytes(old+b'changed')
                with self.assertRaises(v.Drift):self.f.prepare()
                path.write_bytes(old)

    def test_evaluator_body_drift(self):
        path=Path(self.f.cases[0]['arms']['default']['command'][-1]);path.write_text('changed body')
        with self.assertRaises(v.Drift):self.f.prepare()

    def test_output_directory_reuse(self):
        self.f.prepare()
        with self.assertRaises(FileExistsError):self.f.prepare()

    def test_original_decimal_hex_count_nonfinite_and_cold_rejected(self):
        examples=[original_text()+'SAMPLE 1 1\n',original_text().replace('SCORE_HEX 3ff0000000000000','SCORE_HEX 0000000000000000',1),
                  original_text().replace('SAMPLE 1 1.0','SAMPLE 1 nan',1),original_text().replace('SAMPLE','COLD',1),
                  original_text().replace('SAMPLE 1 1.0','SAMPLE malformed',1),original_text().replace('SAMPLE 1','SAMPLE 0',1)]
        for text in examples:
            with self.subTest(text=text),self.assertRaises((v.Protocol,ValueError)):
                v.parse_default(text,self.f.worker)

    def test_native_duplicates_nonfinite_malformed_hex_counts_partial_and_identity_rejected(self):
        config=self.config();good=native_text(config)
        examples=[good+good,good.replace('"score": 1.0','"score": NaN'),good.replace('"score": 1.0','"score": 1e999'),
            good.replace('"score": 1.0','"score": 1.0, "score": 1.0'),good+'IQA_PARTIAL {}\n','IQA_ROW {bad}\n']
        for mutate in [lambda r:r.update(score_bits=None),lambda r:r.update(score_bits='0'*16),
                       lambda r:r.update(n=5),lambda r:r.update(round_means_ns=[1]*5),
                       lambda r:r.update(status='partial'),lambda r:r.update(threads=28),
                       lambda r:r.update(mode='score+map'),lambda r:r.update(timing_phase='cold'),
                       lambda r:r.update(probe_only=True),lambda r:r.update(model_preload_adapter=True),
                       lambda r:r.update(zenbench_samples=[{'gate_clean':None}]*10),
                       lambda r:r.update(round_means_ns=[0]*10)]:
            examples.append(self.mutate_native(mutate))
        for text in examples:
            with self.subTest(text=text),self.assertRaises(v.Protocol):v.parse_native(text,config)

    def test_native_effective10_despite_declared5_and_flags_retained(self):
        text=self.mutate_native(lambda r:r.update(zenbench_samples=[{'gate_clean':False}]*10))
        result=v.parse_native(text,self.config())
        self.assertEqual(result['raw_row']['required_samples'],5)
        self.assertEqual(result['resource_gate_clean'],[False]*10)

    def test_v2_bound_and_v3_null_ratio(self):
        arms={'native':{'complete':True,'score':1.006},'default':{'complete':True,'score':1.0}}
        self.assertFalse(v.compare(self.f.cases[0],arms)['bound_pass'])
        result=v.compare(next(c for c in self.f.cases if c['metric']=='hdrvdp3'),arms)
        self.assertIsNone(result['bound_pass']);self.assertIsNone(result['absolute_bound'])
        self.assertIsNone(result['ratio']);self.assertFalse(result['native_parity_qualified'])

    def test_timeout_retains_both_children_and_ordinary_failure(self):
        self.f.audit();self.f.executor.fail_native=True
        self.assertEqual(self.f.cell(),1);self.assertEqual(len(self.f.executor.calls),2)
        row=v.load(self.f.root/'cells'/self.f.cases[0]['id']/'result.json')
        self.assertFalse(row['fatal']);self.assertEqual(row['arms']['native']['returncode'],124)
        self.assertIsNone(row['absolute_difference'])

    def test_runtime_drift_stops_before_scoring(self):
        self.f.audit();self.f.executor.drift=True
        self.assertEqual(self.f.cell(),2);self.assertEqual(self.f.executor.calls,[])

    def test_altered_seal_and_output_log_refused(self):
        self.f.audit();self.assertEqual(self.f.cell(),0)
        setup=v.load(self.f.root/'setup.json')
        directory=self.f.root/'cells'/self.f.cases[0]['id']
        (directory/'default/control-stdout.log').write_text(original_text(2.0))
        with self.assertRaises(v.Drift):
            v.audit_cell(self.f.root,self.f.cases[0],0,self.f.cohort_sha,self.f.setup_sha,self.f.worker,setup)
        (self.f.root/'preflight-seal.json').write_text('{}')
        # The pinned worker raises ValueError here; require its exact rejection,
        # while the earlier altered-log check exercises our own Drift subtype.
        with self.assertRaisesRegex(ValueError,'^preflight seal differs from explicit queue argument$'):
            v.checked_setup(self.f.root,v.load(self.f.root/'cohort.json'),self.f.cohort_sha,self.f.setup_sha,self.f.worker)

    def test_missing_evidence_never_gracefully_skipped(self):
        self.f.audit()
        with self.assertRaises(FileNotFoundError):v.finish(self.f.root,self.f.cohort_sha,self.f.setup_sha)

    def test_override_and_admission_mismatch(self):
        with patch.dict(os.environ,{'OCTAVE_PATH':'/unbound'}),self.assertRaises(v.Drift):
            v.environment(1,self.f.base)
        self.f.audit()
        with self.assertRaises(v.Drift):
            v.cell(self.f.root,self.f.cohort_sha,self.f.setup_sha,self.f.cases[0]['id'],{})

    def test_alternating_order(self):
        self.assertEqual(v.order(0),['default','native'])
        self.assertEqual(v.order(1),['native','default'])

    def queue_args(self):
        return SimpleNamespace(root=self.f.root,cohort_sha=self.f.cohort_sha,setup_sha='inert-seal',
            case_id=None,heavy_wrapper=Path(self.f.admission['wrapper']),
            lock=[Path(p) for p in self.f.admission['locks']])

    def test_queue_continues_after_ordinary_failure(self):
        self.f.prepare();codes=[1]+[0]*47
        def child(*args,**kwargs):
            index=48-len(codes);case=self.f.cases[index];code=codes.pop(0)
            directory=self.f.root/'cells'/case['id'];directory.mkdir(parents=True)
            v.write(directory/'result.json',dict(id=case['id'],fatal=False,cohort_sha256=self.f.cohort_sha,
                setup_sha256='inert-seal',arms={'default':{},'native':{}},observation_complete=code==0,
                bound_pass=None,ratio=None,matched_scope=False,qualified_timing_cells=0))
            return SimpleNamespace(pid=123,wait=lambda:code)
        with patch.object(v.subprocess,'Popen',side_effect=child) as mocked:
            self.assertEqual(v.queue(self.queue_args()),1)
            self.assertEqual(mocked.call_count,48)

    def test_queue_stops_on_fatal_or_incomplete_worker(self):
        self.f.prepare()
        with patch.object(v.subprocess,'Popen',return_value=SimpleNamespace(pid=123,wait=lambda:2)) as mocked:
            self.assertEqual(v.queue(self.queue_args()),2)
            self.assertEqual(mocked.call_count,1)

    def test_admission_refusal_is_not_an_ordinary_child_failure(self):
        self.f.prepare()
        with patch.object(v.subprocess,'Popen',return_value=SimpleNamespace(pid=123,wait=lambda:1)) as mocked:
            self.assertEqual(v.queue(self.queue_args()),2)
            self.assertEqual(mocked.call_count,1)

    def test_probe_config_false_required_and_optional_false_row(self):
        config=self.config();text=native_text(config)
        self.assertNotIn('"probe_only"',text)
        v.parse_native(text,config)
        v.parse_native(self.mutate_native(lambda r:r.update(probe_only=False)),config)
        for value in (None,True,0):
            with self.subTest(value=value),self.assertRaises(v.Protocol):
                v.parse_native(text,dict(config,probe_only=value))
        with self.assertRaises(v.Protocol):v.parse_native(text+'IQA_PROBE {}\n',config)

    def test_retained_production_row_shapes(self):
        # Caller supplies frozen real logs. Missing fixtures fail, never skip.
        directory=Path(os.environ['HDR_NUMERICAL_RETAINED_ROWS'])
        records=v.load(directory/'manifest.json')['rows']
        expected={
            'hdrvdp-t1':'4bc384d8cb5468bd27f7fc195ea5afba6bf112903e3127b9be8232f4213c768a',
            'hdrvdp-t8':'bf6204c716e6d3c78c59a8b343f32af55289a2f8c0c2531c0418752f8e0ab2ea',
            'hdrvdp3-t1':'e7c27de777e8b3e105403cd42f77fbeb226e4cea86bbb9c2054ddb67976af195',
            'hdrvdp3-t8':'082a19097c70732f62d7fc3b37db7d0cf7d85ee3d1b0eee6dcb5ae8b62ab1110'}
        self.assertEqual({r['id'] for r in records},set(expected))
        self.assertEqual(len(records),4)
        for record in records:
            with self.subTest(case=record['id']):
                stdout=directory/record['stdout'];config_path=directory/record['config']
                self.assertEqual(v.sha(stdout),expected[record['id']])
                self.assertEqual(v.sha(config_path),record['config_sha256'])
                config=v.load(config_path)
                self.assertNotIn('probe_only',config)
                text=stdout.read_text()
                row=next(v.strict(s[8:]) for s in text.splitlines() if s.startswith('IQA_ROW '))
                self.assertNotIn('probe_only',row)
                self.assertEqual(set(config)-set(row),set())
                result=v.parse_native(text,dict(config,probe_only=False))
                self.assertEqual(result['score_bits'],row['score_bits'])


if __name__=='__main__':unittest.main()
