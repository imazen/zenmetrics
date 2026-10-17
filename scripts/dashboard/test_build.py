"""Rejection controls for publication contracts, using isolated disk scratch."""
import json
from pathlib import Path
import tempfile
import unittest
from build import adapt, number, table, catalog


class Contracts(unittest.TestCase):
    def test_competition_metadata_registers_without_timing_rows(self):
        (Path.home()/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            p=Path(d)/'run.meta'
            metadata=dict(identity={'commit':'abc'},build_commit='abc',command=['just','bench-iqa'],host='i265',hardware={},toolchains={},competitors={})
            p.write_text('commit = abc\n'+json.dumps(metadata))
            self.assertEqual(list(adapt(p,{'adapter':'competition-meta'})),[])
            metadata['build_commit']='different';p.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'commit differs'):
                list(adapt(p,{'adapter':'competition-meta'}))

    def test_unknown_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows.tsv';p.write_text('ours\toracle\tnew_column\n1\t2\t3\n')
            with self.assertRaisesRegex(ValueError,'schema mismatch'):list(table(p,['ours','oracle']))

    def test_short_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows.tsv';p.write_text('ours\toracle\n1\n')
            with self.assertRaisesRegex(ValueError,'malformed'):list(table(p,['ours','oracle']))

    def test_nonfinite_refused(self):
        for value in ['NaN','Infinity','-inf']:
            with self.assertRaises(ValueError):number(value)

    def test_competition_preserves_refusal_and_dimensions(self):
        row=dict(host='test',metric_name='metric',impl='reference',device='cpu',threads=2,
                 mode='score+map',width=3840,height=2160,status='unsupported',median_ns=None)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows.json';p.write_text(json.dumps([row]))
            result=list(adapt(p,dict(adapter='competition',columns=list(row))))[0]
            self.assertIsNone(result['ms']);self.assertEqual(result['status'],'unsupported')
            self.assertEqual(result['width'],3840)
            row['unexpected']=1;p.write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError,'unknown competition'):list(adapt(p,dict(adapter='competition',columns=[k for k in row if k!='unexpected'])))

    def test_quality_scatter_uses_measured_values(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'points.tsv'
            p.write_text('metric\tdataset\tpair\tmos\tscore\nmetric\tpanel\timage1\t3.5\t0.25\n')
            result=list(adapt(p,dict(adapter='quality-points',columns=['metric','dataset','pair','mos','score'])))[0]
            self.assertEqual(result['mos'],3.5)
            self.assertEqual(result['score'],0.25)
            self.assertNotIn('srocc',result)

    def test_published_and_oracle_remain_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'aic.tsv'
            cols=['key','ours','oracle','oracle_error','oracle_jnd_error','published','published_error','published_jnd_error']
            p.write_text('\t'.join(cols)+'\npair-a\t2\t2.000001\t0.000001\t0.00001\t3\t1\t0.5\n')
            rows=list(adapt(p,dict(adapter='parity',metric='m',columns=cols,reproduction='reproduction unconfirmed',oracle_precision='7 digits')))
            self.assertEqual([r['series'] for r in rows],['oracle','published'])
            self.assertEqual(rows[1]['abs_delta'],1)
            self.assertEqual(rows[1]['jnd_delta'],0.5)
            self.assertEqual(rows[0]['pair'],'pair-a')

    def test_card_joins_each_output_column(self):
        rows={r['metric']:r for r in catalog()}
        for name in ['lpips-alex','lpips-vgg','stlpips-vgg','wadiqam-fr-live']:
            self.assertNotEqual(rows[name]['license'],'no evidence')
            self.assertIn('card.toml',rows[name]['evidence'])
        self.assertEqual(rows['cvvdp']['needs_display'],'true')
        self.assertEqual(rows['cvvdp-gpu']['requires_gpu'],'true')

    def test_shared_hygiene_rejects_network_shape(self):
        from build import scan_output
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'index.html';p.write_text('.'.join(['172','20','1','9']))
            with self.assertRaisesRegex(ValueError,'hygiene'):scan_output(Path(d))

    def test_exact_hex_residual(self):
        import struct
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'hex.tsv';cols=['key','metric','common_score_bits','common_native_residual_bits','common_jnd_residual_bits']
            p.write_text('\t'.join(cols)+'\npair\tm\t'+struct.pack('>f',2.5).hex()+'\t'+struct.pack('>d',0.125).hex()+'\t'+struct.pack('>d',0.5).hex()+'\n')
            row=list(adapt(p,dict(adapter='hex-residual',columns=cols)))[0]
            self.assertEqual(row['ours'],2.5);self.assertEqual(row['abs_delta'],0.125);self.assertEqual(row['jnd_delta'],0.5)

    def test_new_upload_requires_explicit_classification(self):
        from build import validate_inventory
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'benchmarks/dashboard';folder.mkdir(parents=True)
            path=folder/'classification.tsv';path.write_text('path\tclassification\treason\n')
            (root/'benchmarks/new.csv').write_text('metric,ms\nm,1\n')
            with self.assertRaisesRegex(ValueError,'unclassified'):validate_inventory(root,set())
            path.write_text('path\tclassification\treason\nbenchmarks/new.csv\tignored\tControlled nonmeasurement fixture\n')
            self.assertEqual(validate_inventory(root,set()),{'benchmarks/new.csv'})

    def test_registry_host_labels_are_cpu_derived(self):
        from build import valid_host
        for name in ['i265','r7900x','r5900xt','r5600g','mac','dev','tower']:
            self.assertTrue(valid_host(name))
        for name in ['node-7','r123desk','arbitrary-label']:
            self.assertFalse(valid_host(name))

    def test_json_metadata_preserves_host_and_calculation_identity(self):
        from build import provenance
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'benchmarks/quality';folder.mkdir(parents=True)
            source=folder/'rows.tsv';source.write_text('metric\tvalue\nm\t1\n')
            source.with_suffix('.meta.json').write_text(json.dumps(dict(host='i265',build_commit='a'*40,command='calculate',generated_date_utc='2026-10-17')))
            with patch('build.ROOT',root):result=provenance(source,{})
            self.assertEqual(result['host'],'i265')
            self.assertEqual(result['date'],'2026-10-17')
            self.assertEqual(result['commit'],'a'*40)
            self.assertEqual(result['command'],'calculate')

    def test_nonfinite_json_is_rejected_before_projection(self):
        from adapters import strict_json
        for token in ['NaN','Infinity','-Infinity']:
            with self.assertRaisesRegex(ValueError,'non-finite JSON'):
                strict_json('{"diagnostic":'+token+'}')

    def test_catalog_names_unique(self):
        rows=list(catalog());self.assertGreater(len(rows),100)
        self.assertEqual(len(rows),len({r['metric'] for r in rows}))


if __name__=='__main__':unittest.main()


class NewWebgpuReceipts(unittest.TestCase):
    def test_explicit_nn_backend_and_refused_goldens(self):
        with tempfile.TemporaryDirectory() as scratch:
            path=Path(scratch)/'timing.tsv'
            columns=['host','engine','rten_threads','metric','mode','size','n','median_ms','min_ms','score']
            path.write_text('\t'.join(columns)+'\ni265\tort-webgpu\tdefault\tlpips_vgg\tcompute\t256\t50\t8.468\t7.832\t0.3\n')
            rows=list(adapt(path,dict(adapter='nn-speed',columns=columns,backend='WebGPU',host='i265')))
            self.assertEqual(rows[0]['backend'],'WebGPU')
            self.assertEqual(rows[0]['threads'],'default')
            columns=['commit','engine','model','stem','size','got','want','abs_err','jnd_err','status']
            path.write_text('\t'.join(columns)+'\n107edec1\tort-webgpu\tafine\t-\t-\t-\t-\t-\t-\tLOADFAIL\n')
            rows=list(adapt(path,dict(adapter='nn-goldens',columns=columns,host='i265')))
            self.assertEqual(rows[0]['status'],'failed')
            self.assertIsNone(rows[0]['ours'])
            self.assertIsNone(rows[0]['abs_delta'])
