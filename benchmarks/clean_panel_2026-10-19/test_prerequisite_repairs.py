import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import gate_trace
import fixed_look as f
import nr_validation
import native_contract
import observer_cohort


def attribution():
    p=dict(pid=12,start_ticks=40,pgid=12,sid=12,cgroup='/fixture',ownership_basis='outside owned tree',counted=True,owned=False,cpu_ticks_before=10,cpu_ticks_after=11,interval_ns=100)
    s=dict(snapshot_id='refresh1',sample_monotonic_ns=100,processes=[p],heavy_process_count=1)
    checks=[dict(check_id='c'+str(t),snapshot_id='refresh1',snapshot_sha256=gate_trace.snapshot_hash(s),checked_monotonic_ns=t,observation_age_ns=t-100,heavy_process_count=1,counted_identities=[[12,40]]) for t in (120,130)]
    return dict(schema=2,synthetic=False,snapshots=[s],checks=checks,**{k+'_sha256':'a'*64 for k in ('binary','config','source','trace','outer_receipt')})


class CachedAttribution(unittest.TestCase):
    def test_cached_refresh_one_to_many_counterexample(self):
        r=gate_trace.validate_attribution(attribution())
        self.assertEqual((r['snapshot_count'],r['check_count']),(1,2));self.assertFalse(r['attribution_ready'])
    def test_conflicts_missing_join_and_identity_age_count(self):
        for mutation in ('duplicate-check','snapshot-redefinition','missing-join','wrong-hash','wrong-age','wrong-count','wrong-process'):
            v=attribution()
            if mutation=='duplicate-check':v['checks'].append(copy.deepcopy(v['checks'][0]))
            elif mutation=='snapshot-redefinition':
                other=copy.deepcopy(v['snapshots'][0]);other['sample_monotonic_ns']=99;v['snapshots'].append(other)
            elif mutation=='missing-join':v['checks'][1]['snapshot_id']='unknown'
            elif mutation=='wrong-hash':v['checks'][1]['snapshot_sha256']='b'*64
            elif mutation=='wrong-age':v['checks'][1]['observation_age_ns']=20
            elif mutation=='wrong-count':v['checks'][1]['heavy_process_count']=4
            else:v['checks'][1]['counted_identities']=[[12,41]]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):gate_trace.validate_attribution(v)
    def test_actual_artifact_bytes_and_emitted_join_not_causality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);v=attribution();paths={}
            for key in ('binary','config','source','trace','outer_receipt'):
                paths[key]=root/key;paths[key].write_text('owned synthetic '+key)
            events=[dict(kind='snapshot',value=s) for s in v['snapshots']]+[dict(kind='check',value=c) for c in v['checks']]
            paths['trace'].write_text('\n'.join(json.dumps(e) for e in events))
            for key,path in paths.items():v[key+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            r=gate_trace.verify_artifact_join(v,paths);self.assertTrue(r['emitted_events_verified']);self.assertFalse(r['attribution_ready'])
            paths['binary'].write_text('changed')
            with self.assertRaises(ValueError):gate_trace.verify_artifact_join(v,paths)
            v['binary_sha256']=hashlib.sha256(paths['binary'].read_bytes()).hexdigest()
            paths['trace'].write_text(json.dumps(events[0]))
            v['trace_sha256']=hashlib.sha256(paths['trace'].read_bytes()).hexdigest()
            with self.assertRaises(ValueError):gate_trace.verify_artifact_join(v,paths)


class ObserverBoundary(unittest.TestCase):
    def test_canonical_score_validation_and_state_between_clocks(self):
        events=[];base=nr_validation.Observer()
        class Observed:
            def record(self,values):
                events.append('validation');return base.record(values)
        def score():events.append('metric');return [('brisque',7.),('brisque_ref',19.)]
        def clock():events.append('clock');return 10 if events==['clock'] else 30
        ns,values=nr_validation.timed_pair(score,Observed(),clock)
        self.assertEqual(events,['clock','metric','validation','clock'])
        self.assertEqual((ns,values),(20,(7.,19.)))
        self.assertEqual(base.result(),(7.,19.,1))
        base.record([('brisque',7.),('brisque_ref',20.)])
        with self.assertRaises(ValueError):base.result()
    def test_invalid_columns_nonfinite_and_signed_zero(self):
        for values in ([],[('brisque_ref',1),('brisque',2)],[('brisque',1),('brisque_ref',float('nan'))]):
            with self.assertRaises(ValueError):nr_validation.Observer().record(values)
        o=nr_validation.Observer();o.record([('brisque',1),('brisque_ref',0.)]);o.record([('brisque',1),('brisque_ref',-0.)])
        with self.assertRaises(ValueError):o.result()
    def test_source_stage_finish_seal_refuses_old_binary_and_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'work/zenmetrics'/observer_cohort.DRIVER;source.parent.mkdir(parents=True)
            original=Path(__file__).resolve().parents[2]/observer_cohort.DRIVER
            source.write_bytes(original.read_bytes());observer_cohort.stage(root)
            binary=root/'target/release/iqa-compete';binary.parent.mkdir(parents=True);binary.write_bytes(b'historical baseline')
            with self.assertRaises(ValueError):observer_cohort.finish(root)
            for name in ('build-command.txt','features.txt','metadata.json','rustc.txt','cargo.txt'):(root/name).write_text('synthetic fixture')
            binary.write_bytes(b'synthetic paired-nr-v2 paired-nr-timed-validation-v1')
            observer_cohort.finish(root);observer_cohort.validate(root)
            (root/'features.txt').write_text('changed')
            with self.assertRaises(ValueError):observer_cohort.validate(root)


def fixtures(n=144):
    return [dict(block_id='b'+str(i),status='accepted',times=dict(ours=[100]*10,canonical=[100]*10),duration_ns=1,cleanup_complete=True) for i in range(n)]


def controller(rows=None,stop=None,book=None):
    return f.SyntheticController(f.policy(),f.SyntheticRunner(fixtures() if rows is None else rows,stop),f.SyntheticClock(),book or f.StudyBook(),enable_synthetic=True)


class FixedLook(unittest.TestCase):
    def test_disabled_and_no_live_callbacks_policy_change(self):
        with self.assertRaises(ValueError):f.SyntheticController(f.policy(),f.SyntheticRunner([]),f.SyntheticClock(),f.StudyBook())
        with self.assertRaises(ValueError):f.SyntheticController(f.policy(),lambda:None,f.SyntheticClock(),f.StudyBook(),enable_synthetic=True)
        p=f.policy();p['accepted_total']=4
        with self.assertRaises(ValueError):f.SyntheticController(p,f.SyntheticRunner([]),f.SyntheticClock(),f.StudyBook(),enable_synthetic=True)
    def test_exact_ranks_and_ties_and_next_narrower_tail(self):
        from fractions import Fraction
        from math import comb
        r=f.rank_interval(list(range(1,121)))
        self.assertEqual((r['k'],r['upper_rank'],r['lo'],r['hi'],r['estimate']),(49,72,49,72,60.5))
        self.assertLessEqual(Fraction(2*sum(comb(120,j) for j in range(49)),2**120),Fraction(1,20))
        self.assertGreater(Fraction(2*sum(comb(120,j) for j in range(50)),2**120),Fraction(1,20))
        self.assertEqual(f.rank_interval([1.]*120)['lo'],1.)
        for values in ([1.]*119,[1.]*121,[0.]+[1.]*119,[float('nan')]+[1.]*119):
            with self.assertRaises(ValueError):f.rank_interval(values)
    def test_interval_semantics(self):
        self.assertFalse(f.interval_decision(.48,.5,.52)['precision_pass'])
        self.assertTrue(f.interval_decision(1.95,2.,2.05)['precision_pass'])
        self.assertTrue(f.interval_decision(1.08,1.09,1.1)['threshold_crossing'])
        self.assertEqual(f.interval_decision(1.09,1.1,1.11)['conclusion'],'above-1.08')
        self.assertEqual(f.interval_decision(.99,1.,1.01)['conclusion'],'below-1.08')
    def test_complete_balanced_fixed_look_only_once(self):
        c=controller();r=c.run('study1')
        self.assertEqual(r['accepted'],dict(AB=60,BA=60));self.assertEqual(len(r['attempts']),120)
        self.assertEqual(len(r['not_attempted']),24);self.assertFalse(r['qualified'])
        self.assertEqual(r['fixed_look']['interval']['k'],49)
        self.assertEqual(r['fixed_look']['diagnostics']['p_order'],1.)
        with self.assertRaises(ValueError):c.run('study2')
        with self.assertRaises(ValueError):controller(book=c.book).run('study1')
        self.assertEqual(len(c.book.records),1)
    def test_full_study_reporting_and_admission_wait_deadline(self):
        book=f.StudyBook();rows=fixtures();rows[0]['admission_wait_ns']=f.policy()['deadline_ns']
        incomplete=controller(rows,book=book).run('queued')
        self.assertEqual(incomplete['reason'],'deadline-during-admission')
        self.assertEqual(incomplete['attempts'],[]);self.assertNotIn('fixed_look',incomplete)
        controller(book=book).run('complete')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'studies.json';book.write_full_report(path)
            saved=json.loads(path.read_text());self.assertEqual(len(saved['studies']),2)
            self.assertEqual(saved['study_ids'],['complete','queued'])
            self.assertNotIn('fixed_look',saved['studies'][0])
            with self.assertRaises(FileExistsError):book.write_full_report(path)
    def test_attempt_budget_and_rejection_boundary(self):
        rows=fixtures()
        for row in rows[:24]:row['status']='resource-rejected'
        r=controller(rows).run('at-cap')
        self.assertEqual(r['accepted'],dict(AB=60,BA=60));self.assertEqual(r['resource_rejected'],dict(AB=12,BA=12));self.assertEqual(len(r['attempts']),144)
        rows=fixtures()
        for row in rows[:25]:row['status']='resource-rejected'
        r=controller(rows).run('over-cap');self.assertEqual(r['reason'],'rejection-cap');self.assertNotIn('fixed_look',r)
    def test_stop_deadline_cleanup_and_unknown_containment(self):
        r=controller(stop=0).run('before');self.assertEqual(len(r['attempts']),0);self.assertNotIn('fixed_look',r)
        rows=fixtures();rows[0]['duration_ns']=10
        r=controller(rows,stop=5).run('inflight');self.assertEqual(r['reason'],'STOP-inflight');self.assertNotIn('fixed_look',r)
        self.assertEqual([e['kind'] for e in r['events']],['synthetic-locks-held','synthetic-launch','synthetic-cleanup','synthetic-locks-released'])
        rows=fixtures();rows[0]['duration_ns']=f.policy()['deadline_ns']
        r=controller(rows).run('deadline');self.assertEqual(r['reason'],'deadline-inflight');self.assertNotIn('fixed_look',r)
        rows=fixtures();rows[0]['cleanup_complete']=False
        r=controller(rows).run('unknown');self.assertEqual(r['reason'],'containment-unknown');self.assertNotIn('fixed_look',r)
        self.assertNotIn('synthetic-locks-released',[e['kind'] for e in r['events']])
    def test_duplicate_zero_missing_wrong_order_fatal_not_resource_retry(self):
        variants=[]
        rows=fixtures();rows[1]['block_id']=rows[0]['block_id'];variants.append(rows)
        rows=fixtures();rows[0]['times']['ours'][0]=0;variants.append(rows)
        rows=fixtures();rows[0]['times']['canonical']=[];variants.append(rows)
        rows=fixtures();rows[0]['order']='invalid';variants.append(rows)
        rows=fixtures();rows[0]['status']='parity-failure';variants.append(rows)
        for i,rows in enumerate(variants):
            r=controller(rows).run('bad'+str(i));self.assertNotIn('fixed_look',r);self.assertLessEqual(len(r['attempts']),2)
    def test_schedule_fixed_pairs_and_diagnostic_tie_tail(self):
        s=f.schedule(f.policy());self.assertEqual((s.count('AB'),s.count('BA')),(72,72))
        for i in range(0,144,2):self.assertEqual(set(s[i:i+2]),{'AB','BA'})
        self.assertEqual(s[:8],['BA','AB','BA','AB','BA','AB','AB','BA'])
        d=f.diagnostic_stats([1.]*120,['AB','BA']*60,permutations=9)
        self.assertEqual((d['p_order'],d['p_lag']),(1.,1.))
        d=f.diagnostic_stats([1.]*60+[2.]*60,['AB']*60+['BA']*60,permutations=99)
        self.assertLessEqual(d['p_order'],.025);self.assertTrue(d['diagnostic_refusal']);self.assertFalse(d['iid_established'])


class NativeProposal(unittest.TestCase):
    def test_separate_native_graph_and_identity_no_scope_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def file(name):p=root/name;p.write_bytes(name.encode());return dict(path=str(p),role=name)
            exe=file('binary');lib=file('transitive-native-lib');source=file('shim.cc')
            builds=[file(n) for n in ('source_revision','features','overlays','toolchain','build_command','binary_mapping')]
            inputs=[file(n) for n in ('reference','distorted')];control=[file('controller')]
            for metric in ('butteraugli-original','butteraugli','ssim2'):
                p=native_contract.proposal(metric,executable=exe['path'],source_receipts=[source],build_receipts=builds,inputs=inputs,controller_receipts=control,loader={'LD_LIBRARY_PATH':None})
                self.assertTrue(native_contract.verify_files(p));self.assertFalse(p['scope_qualified']);self.assertIsNone(p['python_scorer_environment'])
                closure=native_contract.synthetic_closure(p,lambda path,env:dict(dependencies=[lib['path']] if path==exe['path'] else [],dynamic_search=[]))
                self.assertEqual(len(closure['files']),2)
                bad=copy.deepcopy(p);bad['scope_qualified']=True
                with self.assertRaises(ValueError):native_contract.verify_files(bad)
            Path(source['path']).write_text('changed')
            with self.assertRaises(ValueError):native_contract.verify_files(p)
            with self.assertRaises(ValueError):native_contract.proposal('new-cell',executable=exe['path'],source_receipts=[source],build_receipts=builds,inputs=inputs,controller_receipts=control,loader={})


class AdmissionBinding(unittest.TestCase):
    def test_kernel_lock_and_owned_launch_receipt_without_lock_lifetime_change(self):
        import fcntl
        import admission_receipt
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[root/'first.lock',root/'second.lock']
            with paths[0].open('a') as a,paths[1].open('a') as b:
                with self.assertRaises(ValueError):admission_receipt.Held([(a,paths[0]),(b,paths[1])],root/'unheld')
                for fd in (a,b):fcntl.flock(fd,fcntl.LOCK_EX)
                held=admission_receipt.Held([(a,paths[0]),(b,paths[1])],root/'held')
                with self.assertRaises(ValueError):held.bind_launch(['foreign'],os.getpid())
                args=[sys.executable,'-c','import time;time.sleep(10)']
                child=subprocess.Popen(args)
                try:
                    r=held.bind_launch(args,child.pid);self.assertTrue(r['ownership_verified']);self.assertEqual(r['owner']['pid'],os.getpid())
                    with self.assertRaises(ValueError):held.bind_launch(args,child.pid)
                    probe='import fcntl,sys;f=open(sys.argv[1],"a");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
                    for path in paths:self.assertNotEqual(subprocess.run([sys.executable,'-c',probe,str(path)],capture_output=True).returncode,0)
                finally:child.terminate();child.wait(timeout=5)
            for path in paths:
                with path.open('a') as fd:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
