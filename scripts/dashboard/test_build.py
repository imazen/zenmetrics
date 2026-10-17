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

    def test_catalog_names_unique(self):
        rows=list(catalog());self.assertGreater(len(rows),100)
        self.assertEqual(len(rows),len({r['metric'] for r in rows}))


if __name__=='__main__':unittest.main()
