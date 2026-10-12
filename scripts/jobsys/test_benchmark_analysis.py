"""Synthetic protocol and numerical controls; no benchmark claims."""
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import analyze_benchmark_study as analysis
import build_benchmark_study as study
from test_benchmark_study import StudyTests


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.fixture=StudyTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.fixture.qualify();self.root=self.fixture.root
        self.generated=self.root/'generated';study.write_study(self.fixture.write(),self.generated)
        _,self.cells=analysis.read_cells(self.generated)
        self.packaged=self.root/'packaged';self.packaged.mkdir()
        identities=[]
        for sha,row in self.cells.items():
            if row['status']!='ready': continue
            spec=study.load(self.generated/sha/'bundle-spec.json')
            files={name:dict(sha256=study.file_sha(Path(value['source'])),bytes=Path(value['source']).stat().st_size,
                            executable=value['executable']) for name,value in spec['files'].items()}
            bundle=self.packaged/sha;bundle.mkdir()
            (bundle/'bundle.json').write_bytes(study.canonical(dict(schema=1,target=row['target'],plan=spec['plan'],files=files)))
            identities.append(dict(cell_sha=sha,bundle_sha=study.file_sha(bundle/'bundle.json')))
        (self.packaged/'identities.json').write_bytes(study.canonical(dict(schema=1,study_sha=next(iter(self.cells.values()))['study_sha'],runner_sha='b'*64,cells=identities)))
        self.row=next(row for row in self.cells.values() if row['status']=='ready')
        self.sha=self.row['cell_sha']

    def attempt(self,name,status='complete'):
        root=self.root/name;root.mkdir()
        row=self.row
        driver=dict(schema=1,operations=2,correctness=row['expected_correctness'],memory=[])
        stdout=dict(driver=driver,timing_usable=True,operation_ns=[123,456],workload=dict(row["case"]))
        (root/'driver.json').write_bytes(study.canonical(driver))
        (root/'stdout.txt').write_bytes(study.canonical(stdout)+b'\n')
        identities=study.load(self.packaged/'identities.json')
        bundle=next(b for b in identities['cells'] if b['cell_sha']==self.sha)['bundle_sha']
        packet=dict(schema=1,kind='zenbench_fleet_attempt',attempt=name,experiment=self.sha,
                    bundle_sha256=bundle,machine=row['machine'],domain=row['domain'],replicate=row['replicate'],
                    runner=dict(runner=dict(sha256='b'*64,native_arch_verified=False)),
                    plan=study.load(self.generated/self.sha/'bundle-spec.json')['plan'],
                    artifacts={name:study.file_sha(root/name) for name in ['driver.json','stdout.txt']},
                    status=status,correctness=row['expected_correctness'],wall_seconds=1.5,memory=[],
                    inference='synthetic control only')
        (root/'packet.json').write_bytes(study.canonical(packet))
        return root

    def test_gpu_analysis_refuses_software_wrong_backend_and_wrong_name(self):
        case=dict(backend='wgpu',memory_mode='full',parameter_preset='api-defaults-standard-fhd',expected_adapter='fixture GPU')
        workload=dict(case,device=dict(name='fixture GPU',type='IntegratedGpu'))
        analysis.verify_gpu_workload(case,workload)
        for kind in ('Cpu','Other','VirtualGpu','CudaGpu'):
            workload['device']['type']=kind
            with self.assertRaisesRegex(ValueError,'GPU runtime identity'):
                analysis.verify_gpu_workload(case,workload)
        workload['device']=dict(name='different GPU',type='DiscreteGpu')
        with self.assertRaisesRegex(ValueError,'GPU runtime identity'):
            analysis.verify_gpu_workload(case,workload)
        workload['device']['name']='fixture GPU';workload['backend']='cpu'
        with self.assertRaisesRegex(ValueError,'GPU runtime workload'):
            analysis.verify_gpu_workload(case,workload)

    def test_missing_failed_and_retry_ambiguity_remain_visible(self):
        empty=analysis.ingest(self.generated,self.packaged,[])
        self.assertEqual(sum(r['status']=='missing' for r in empty['audit']),12)
        busy=self.attempt('busy','busy')
        failed=analysis.ingest(self.generated,self.packaged,[busy])
        self.assertEqual(next(r for r in failed['audit'] if r['cell_sha']==self.sha)['status'],'failed')
        one=self.attempt('one');two=self.attempt('two')
        observed=analysis.ingest(self.generated,self.packaged,[one,busy])
        self.assertEqual(len(observed['observations']),3)
        retry=analysis.ingest(self.generated,self.packaged,[one,two,busy])
        self.assertEqual(retry['observations'],[])
        self.assertEqual(next(r for r in retry['audit'] if r['cell_sha']==self.sha)['status'],'ambiguous_retry')

    def test_hash_and_contract_corruption_refused(self):
        one=self.attempt('one')
        (one/'stdout.txt').write_text('changed')
        with self.assertRaisesRegex(ValueError,'artifact'): analysis.ingest(self.generated,self.packaged,[one])
        bundle=self.packaged/self.sha/'bundle.json';bundle.write_text('{}')
        with self.assertRaisesRegex(ValueError,'bundle manifest'): analysis.ingest(self.generated,self.packaged,[])

    def test_curves_do_not_pool_unlike_groups_or_quantities(self):
        value=analysis.ingest(self.generated,self.packaged,[self.attempt('one')])
        self.assertEqual(len(analysis.curves(value)),3)
        self.assertTrue(all(curve['fit'] is None for curve in analysis.curves(value)))
        point=dict(value['observations'][0]);point['group']=dict(point['group'],threads=9)
        value['observations'].append(point)
        self.assertEqual(len(analysis.curves(value)),4)

    def test_svg_and_html_preserve_evidence_labels(self):
        value=analysis.ingest(self.generated,self.packaged,[self.attempt('one')])
        chart=analysis.svg(analysis.curves(value)[0])
        ET.fromstring(chart)
        self.assertIn('no confirmation interval',chart)
        output=self.root/'charts';result=analysis.write(value,output)
        self.assertEqual(result['curves'],3)
        self.assertIn('Declared cell audit',(output/'index.html').read_text())
        with self.assertRaises(FileExistsError): analysis.write(value,output)

    def test_descriptive_fit_reports_residuals_and_needs_three_sizes(self):
        self.assertIsNone(analysis.linear_fit([(1,2),(2,3)]))
        fit=analysis.linear_fit([(1,5),(2,8),(3,11),(4,14)])
        self.assertEqual((fit['intercept'],fit['slope'],fit['rmse']),(2,3,0))
        self.assertEqual(fit['residuals'],[0]*4)
        group=dict(metric='fixture',testbed='fixture',threads=1,reuse='cold')
        curve=dict(identity=dict(group=group,quantity='fixture',scope='synthetic',unit='bytes'),observations=[dict(value=0,pixels=1,width=1,height=1,replicate=0,attempt='fixture')])
        with self.assertRaisesRegex(ValueError,'positive'): analysis.svg(curve,True)
