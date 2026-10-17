"""Rejection controls for publication contracts, using isolated disk scratch."""
import json
from pathlib import Path
import tempfile
import unittest
from build import adapt, number, table, catalog


class Contracts(unittest.TestCase):
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

    def test_catalog_names_unique(self):
        rows=list(catalog());self.assertGreater(len(rows),100)
        self.assertEqual(len(rows),len({r['metric'] for r in rows}))


if __name__=='__main__':unittest.main()
