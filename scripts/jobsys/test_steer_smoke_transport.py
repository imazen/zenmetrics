"""Versioned, non-installable engineering probe transport; invented blobs only."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fit_cell_exec as exe
import harvest_fit_cells as harvest
import pack_fit_program as pack


def sha(b):
    return hashlib.sha256(b).hexdigest()


def archive(path,files):
    with path.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z,tarfile.open(fileobj=z,mode='w') as t:
        for name,b in files.items():
            member=tarfile.TarInfo(name);member.size=len(b);t.addfile(member,io.BytesIO(b))


class SteerSmokeTransport(unittest.TestCase):
    def test_cap_and_roots(self):
        argv=['steer_exact_smoke.py','--wall-cap-sec','600']
        self.assertEqual(exe.registered_wall_cap(argv,'a'*64),600)
        for a in [[],['--wall-cap-sec','0'],['--wall-cap-sec','901'],['--wall-cap-sec','x']]:
            with self.assertRaises((ValueError,IndexError)):
                exe.registered_wall_cap(['steer_exact_smoke.py',*a],'a'*64)
        self.assertEqual(exe.SCRIPTS['steer_exact_smoke.py'][0],harvest.PROGRAM_ROOTS['steer_exact_smoke.py'])
        self.assertEqual(exe.ROOTED['steer_exact_smoke.py'],harvest.ROOTED['steer_exact_smoke.py'])

    def test_control_cap_is_archive_bound_and_smoke_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'bounded_smoke_policy.json').write_text(json.dumps(dict(schema='bounded-fit-smoke-policy-v1',epochs=1,pairs_per_epoch=49999,control_wall_cap_sec=900)))
            a=['v2_lodo_mlp.py','--local-smoke-budget','1:49999']
            self.assertEqual(exe.registered_wall_cap(a,'b'*64,root),900)
            with self.assertRaises(ValueError):exe.registered_wall_cap(a[:-1]+['2:128'],'b'*64,root)
            self.assertIsNone(exe.registered_wall_cap(['v2_lodo_mlp.py'],'b'*64,root))

    def test_source_pins_are_required(self):
        with self.assertRaises(ValueError):pack.bounded_smoke_sources(None,None)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pins.json';p.write_text(json.dumps({'schema':'bounded-smoke-program-v1','files':{}}))
            with self.assertRaises(ValueError):pack.bounded_smoke_sources(p,sha(p.read_bytes()))
            with self.assertRaises(ValueError):pack.bounded_smoke_sources(p,'0'*64)
            # A training contract is data, not a source file in the checkout.
            required=["v2_lodo_mlp.py","v2_confirm_fit.py","v2_common.py","v2_teacher.py","v2_smoke_contract.py","steer_exact_smoke.py"]
            p.write_text(json.dumps(dict(schema='bounded-smoke-program-v1',files={'scripts/rev4_featpot/'+n:'a'*64 for n in required},data={'benchmarks/e33_fit_contract_2026-10-10.json':'b'*64})))
            self.assertEqual(len(pack.bounded_smoke_sources(p,sha(p.read_bytes()))),6)


    def test_timed_control_preserves_fresh_destination(self):
        # Independent child contract: a strict trainer refuses any existing destination.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); program=root/'program'; source=program/'scripts/rev4_featpot'
            source.mkdir(parents=True); fits=root/'fits'; dest=fits/'results/cells/arm/fit'
            dest.parent.mkdir(parents=True)
            (source/'v2_lodo_mlp.py').write_text("""
import sys,json,hashlib
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--dest')+1])
if p.exists():
    raise RuntimeError('strict child requires a fresh destination')
(p/'refit').mkdir(parents=True)
m=p/'refit/model.bin';m.write_bytes(b'invented control')
(p/'result.json').write_text(json.dumps(dict(selected_bake=str(m),selected_bake_sha256=hashlib.sha256(m.read_bytes()).hexdigest())))
""")
            argv=['v2_lodo_mlp.py','--dest',str(dest),'--strict-admission','--local-smoke-budget','1:49999']
            def local_path(value):
                return root/'runtime-tmp' if str(value)=='/scratch/tmp' else Path(value)
            with patch.object(exe,'PROGRAM',program),patch.object(exe,'FIT_ROOT',fits), \
                 patch.object(exe,'registered_wall_cap',return_value=10),patch.object(exe,'Path',side_effect=local_path):
                blob=exe.run_fit_locked({},dict(data_sha='b'*64),'a'*64,dict(effective='v3'),argv,'c'*64,Path('arm/fit'),None,dest)
            with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as t:
                resource=t.extractfile('results/cells/arm/fit/RESOURCE_USAGE.txt').read()
                receipt=json.load(t.extractfile('results/cells/arm/fit/fleet_receipt.json'))
            self.assertIn(b'Maximum resident set size (kbytes)',resource)
            self.assertIn(b'Exit status: 0',resource)
            self.assertEqual(receipt['files']['RESOURCE_USAGE.txt'],sha(resource))

    def test_archive_verifier_and_negative_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp=Path(directory)
            inspector=tmp/'inspect';inspector.write_text('#!/bin/sh\nexit 0\n');inspector.chmod(0o755)
            program=tmp/'program.tar.gz'
            archive(program,{'build_meta.json':json.dumps({'files':{'bin/inspect_qualified_checkpoint':sha(inspector.read_bytes())}}).encode()})
            argv=['steer_exact_smoke.py','--family','a','--model-sha256',sha(b'invented-model')]
            kind={'argv':argv,'argv_sha':sha(json.dumps(argv,separators=(',',':')).encode()),'program_sha':sha(program.read_bytes()),'data_sha':'b'*64}
            def blob(mutate=None):
                probe={'schema':'steer-exact-synthetic-smoke-v1','policy':'all-blocks-v1','model_sha256':sha(b'invented-model'),
                    'cases':[{'width':w,'height':h,'score_and_feature_bits_equal':True,
                        'rectangles':[{'gain_bits':0,'native_gain_bits':0} for _ in range(n)]} for w,h,n in [(65,49,91),(128,96,256)]]}
                if mutate:mutate(probe)
                pb=json.dumps(probe).encode()
                selected=harvest.POT_ROOT/'steer-smoke/cells/probe/refit/control.bin'
                result={'schema':'steer-exact-smoke-cell-v1','execution_contract':'local-smoke','epochs':0,'family':'a',
                        'selected_bake':str(selected),'selected_bake_sha256':sha(b'invented-model'),'probe_sha256':sha(pb)}
                files={'result.json':json.dumps(result).encode(),'exact-probe.json':pb,'refit/control.bin':b'invented-model','RESOURCE_USAGE.txt':b'invented-time'}
                receipt={'schema':'zenfleet-fit-cell-receipt-v1','cell':'probe',**{k:kind[k] for k in ('program_sha','data_sha','argv_sha')},
                         'files':{k:sha(v) for k,v in files.items()},'result_sha':sha(files['result.json']),'selected_bake_sha':sha(b'invented-model')}
                files['fleet_receipt.json']=json.dumps(receipt).encode()
                path=tmp/f'blob-{len(list(tmp.glob("blob-*")))}.tar.gz'
                archive(path,{'steer-smoke/cells/probe/'+k:v for k,v in files.items()});return path
            kw=dict(program_archive=program,checkpoint_inspector=inspector,allow_local_smoke=True)
            valid=harvest.verify_blob(blob(),tmp/'valid','probe',kind,**kw)
            self.assertEqual(valid['execution_contract'],'local-smoke')
            with self.assertRaises(ValueError):harvest.verify_blob(blob(),tmp/'install','probe',kind,**(kw|{'allow_local_smoke':False}))
            for i,mutation in enumerate([lambda p:p['cases'][0]['rectangles'].pop(),
                    lambda p:p['cases'][1]['rectangles'][0].update(native_gain_bits=1),lambda p:p.update(model_sha256='0'*64)]):
                with self.assertRaises(ValueError):harvest.verify_blob(blob(mutation),tmp/f'invalid{i}','probe',kind,**kw)
            with self.assertRaises(ValueError):harvest.verify_blob(blob(),tmp/'archive','probe',kind,**(kw|{'program_archive':inspector}))

if __name__=='__main__':unittest.main()
