import json
from pathlib import Path
import tempfile
import unittest

from compare_canonical import sha
from prepare_ordered_canonical import prepare


class OrderedContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.asset=self.root/'rgb';self.asset.write_bytes(b'rgb')
        self.binary=self.root/'worker';self.inventory(['a','b'])
        self.order=self.root/'order.tsv';self.order.write_text('metric\thistorical_ms\n'+'b\t0.5\na\t1.0\n')
        self.source=self.root/'source.json'
        cases=[dict(metric=name,id=name+str(t),width=1,height=1,threads=t,reference=str(self.asset),distorted=str(self.asset),
                    matched_scope=True,absolute_bound=0.004,relative_bound=0.0001,scope='resident inputs',canonical={'command':['original',name]})
               for name in ['a','b'] for t in [8,1]]
        self.source.write_text(json.dumps({'bindings':{str(self.asset):sha(self.asset)},'cases':cases}))
        self.manifest=self.root/'manifest.json'
        self.manifest.write_text(json.dumps({'binary':str(self.binary),'rust_source':'pinned-source',
            'order':str(self.order),'metric_plans':{name:str(self.source) for name in ['a','b']}}))
        self.output=self.root/'output.json'

    def inventory(self,names):
        rows=[{'metric_name':name,'compiled':True} for name in names]
        self.binary.write_text('#!/usr/bin/env python3\nimport json\nprint(json.dumps('+repr(rows)+'))\n');self.binary.chmod(0o755)

    def test_order_threads_and_numerical_contract_are_preserved(self):
        self.assertEqual(prepare(self.manifest,self.output,[1,3]),(2,4))
        value=json.loads(self.output.read_text());self.assertEqual([c['metric'] for c in value['cases']],['b','b','a','a'])
        self.assertEqual([c['threads'] for c in value['cases']],[1,3,1,3])
        for case in value['cases']:
            self.assertEqual((case['absolute_bound'],case['relative_bound'],case['scope']),(0.004,0.0001,'resident inputs'))
        self.assertEqual(value['bindings'][str(self.source)],sha(self.source))
        self.assertEqual(value['bindings'][str(self.binary)],sha(self.binary))

    def test_changed_reference_asset_is_refused(self):
        self.asset.write_bytes(b'changed')
        with self.assertRaises(ValueError):prepare(self.manifest,self.output,[1])
        self.assertFalse(self.output.exists())

    def test_malformed_scope_order_and_threads_are_refused(self):
        for text in ['metric\thistorical_ms\na\t1\na\t2\n','metric\thistorical_ms\na\t2\nb\t1\n',
                     'metric\thistorical_ms\na\t1\n','metric\thistorical_ms\na\tNaN\nb\t2\n']:
            with self.subTest(text=text):
                self.order.write_text(text)
                with self.assertRaises(ValueError):prepare(self.manifest,self.output,[1])
        self.order.write_text('metric\thistorical_ms\nb\t1\na\t2\n')
        for counts in [[],[0],[1,1]]:
            with self.subTest(counts=counts),self.assertRaises(ValueError):prepare(self.manifest,self.output,counts)

    def test_missing_compiled_route_or_matched_case_is_refused(self):
        self.inventory(['a'])
        with self.assertRaises(ValueError):prepare(self.manifest,self.output,[1])
        self.inventory(['a','b']);value=json.loads(self.source.read_text())
        for case in value['cases']:case['matched_scope']=False
        self.source.write_text(json.dumps(value))
        with self.assertRaises(ValueError):prepare(self.manifest,self.output,[1])

    def test_existing_output_is_preserved(self):
        self.output.write_text('retained observations')
        with self.assertRaises(FileExistsError):prepare(self.manifest,self.output,[1])
        self.assertEqual(self.output.read_text(),'retained observations')


if __name__=='__main__':unittest.main()
