"""Read-only capture of sealed results published after an in-progress report."""
import json
from pathlib import Path
import tempfile
import unittest
import durable
from export_chart_snapshot import export, resource_evidence


class ChartExport(unittest.TestCase):
    def test_explicit_memory_guard_stop_does_not_impute_peak_rss(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            log=Path(directory)/'worker.log'
            log.write_text('run-heavy: done rc=137 19s | peak-RSS unavailableGiB | min-avail 10844MiB\n'
                           'run-heavy: monitored memory guard stopped the process group; BSD peak RSS may be unavailable\n')
            result=resource_evidence(log,durable.file_hash(log),'mac')
            self.assertEqual(result['resource_guard_stop'],'monitored_memory_guard')
            self.assertIsNone(result['peak_rss_gib_rounded'])
            self.assertIsNone(result['os_time_max_rss_bytes'])
            self.assertIsNone(resource_evidence(log,'wrong-hash','mac')['resource_guard_stop'])
            log.write_text('run-heavy: done rc=137 19s | peak-RSS unavailableGiB\n')
            self.assertIsNone(resource_evidence(log,durable.file_hash(log),'mac')['resource_guard_stop'])

    def test_resource_receipt_binds_hash_and_preserves_units(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            log = Path(directory)/'worker.log'
            log.write_text('  104857600 maximum resident set size\nrun-heavy: done rc=0 peak-RSS 0.10 GiB\n')
            measured = resource_evidence(log, durable.file_hash(log), 'mac')
            self.assertEqual(measured['os_time_max_rss_bytes'], 104857600)
            self.assertEqual(measured['peak_rss_gib_rounded'], 0.10)
            self.assertIsNone(resource_evidence(log, durable.file_hash(log), 'i265')['os_time_max_rss_bytes'])
            self.assertIsNone(resource_evidence(log, 'wrong-hash', 'mac')['peak_rss_gib_rounded'])

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path.home()/'tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/'results'/'phase'
        self.attempt=self.source/'cells'/'000000-fixture'/'attempt-0001'
        self.attempt.mkdir(parents=True)
        self.config=dict(metric_name='fixture',durable_cell_id='000000-fixture',timing_phase='warm',
                         only_mode='score',cold_repeat=0,threads=8,width=512,height=512)
        row=dict(self.config,mode='score',status='ok',score=1.,score_bits='3ff0000000000000',n=10,
                 median_ns=10,round_means_ns=[10]*10,zenbench_samples=[dict(gate_clean=True) for _ in range(10)])
        (self.attempt/'config.json').write_text(json.dumps(self.config))
        (self.attempt/'intent.json').write_text('{}')
        (self.attempt/'worker.log').write_text('IQA_ROW '+json.dumps(row)+'\n')
        self.record=dict(identity='identity',cell_id='000000-fixture',attempt=1,status='ok',row=row,
                         cleanup_verified=True,**{k:durable.file_hash(self.attempt/n) for k,n in
                         [('config_sha256','config.json'),('log_sha256','worker.log'),('intent_sha256','intent.json')]})
        durable.save_result(self.attempt/'result.json',self.record)
        self.report=dict(identity='identity',complete=False,qualified=False,
                         cells=[dict(config=self.config,status='pending',attempts=[])])
        (self.source/'report.json').write_text(json.dumps(self.report))
        (self.source/'identity.json').write_text(json.dumps(dict(policy=dict(max_attempts=2))))

    def capture(self,name,sealed=True):
        out=self.root/name
        export(self.source.parent,out,'fixture',sealed)
        return out,[json.loads(l) for l in (out/'observations.jsonl').read_text().splitlines()]

    def test_sealed_overlay_is_qualified_without_modifying_report(self):
        original=(self.source/'report.json').read_bytes()
        out,rows=self.capture('new')
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['evidence_origin'],'sealed-result-overlay')
        self.assertTrue(rows[0]['qualified_timing'])
        self.assertEqual(rows[0]['cell_status'],'ok')
        self.assertEqual(rows[0]['report_cell_status'],'pending')
        self.assertEqual(json.loads((out/'snapshot.json').read_text())['schema'],2)
        self.assertEqual((self.source/'report.json').read_bytes(),original)
        self.assertEqual(self.capture('old',False)[1],[])

    def test_native_worker_engine_survives_export_without_nn_environment(self):
        self.record['row']['engine']='rust'
        (self.attempt/'worker.log').write_text('IQA_ROW '+json.dumps(self.record['row'])+'\n')
        self.record['log_sha256']=durable.file_hash(self.attempt/'worker.log')
        durable.save_result(self.attempt/'result.json',self.record)
        self.assertEqual(self.capture('native-engine')[1][0]['engine'],'rust')

    def test_reported_attempt_is_not_duplicated(self):
        self.report['cells'][0].update(status='ok',attempts=[self.record])
        (self.source/'report.json').write_text(json.dumps(self.report))
        rows=self.capture('new')[1]
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['evidence_origin'],'report')

    def test_changed_log_or_bad_seal_refuses(self):
        (self.attempt/'worker.log').write_text('changed')
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            self.capture('bad-log')
        packet=json.loads((self.attempt/'result.json').read_text());packet['record']['status']='failed'
        (self.attempt/'result.json').write_text(json.dumps(packet))
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            self.capture('bad-seal')

    def test_valid_seal_from_another_experiment_refuses(self):
        self.record['identity']='another-experiment'
        durable.save_result(self.attempt/'result.json',self.record)
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            self.capture('wrong-experiment')

    def test_flagged_round_is_kept_but_not_qualified(self):
        self.record['row']['zenbench_samples'][0]['gate_clean']=False
        (self.attempt/'worker.log').write_text('IQA_ROW '+json.dumps(self.record['row'])+'\n')
        self.record['log_sha256']=durable.file_hash(self.attempt/'worker.log')
        durable.save_result(self.attempt/'result.json',self.record)
        row=self.capture('flagged')[1][0]
        self.assertFalse(row['qualified_timing'])
        self.assertEqual(row['warm_rounds_flagged'],1)
        self.assertEqual(row['warm_rounds_clean'],9)

    def test_explicit_rejection_diagnostic_never_qualifies_invalid_overlay(self):
        (self.attempt/'worker.log').write_text('changed')
        original=(self.source/'report.json').read_bytes()
        out=self.root/'rejected-diagnostic'
        export(self.source.parent,out,'fixture',True,True)
        self.assertEqual((out/'observations.jsonl').read_text(),'')
        snapshot=json.loads((out/'snapshot.json').read_text())
        self.assertEqual(snapshot['qualified_attempt_rows'],0)
        self.assertEqual(snapshot['rejected_overlay_cells'],1)
        rejected=json.loads((out/'sources.json').read_text())[0]['rejected_overlay_cells'][0]
        self.assertIn('evidence changed',rejected['error'])
        self.assertIn('cells/000000-fixture/attempt-0001/worker.log',rejected['artifacts'])
        self.assertEqual((self.source/'report.json').read_bytes(),original)
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            self.capture('default-still-refuses')

    def test_missing_final_result_is_not_an_observation(self):
        (self.attempt/'result.json').unlink()
        self.assertEqual(self.capture('pending')[1],[])


if __name__ == '__main__': unittest.main()
