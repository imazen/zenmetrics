"""Benchmark protocol controls; real runner integration is an explicit CLI mode."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

path=Path(__file__).with_name('benchmark_exec.py')
spec=importlib.util.spec_from_file_location('benchmark_exec',path)
execmod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(execmod)

class Contract(unittest.TestCase):
    def setUp(self):
        env = dict(os.environ)
        for key in ('ZEN_BLOBS_DIR', 'ZENBENCH_BLOB_URI'):
            env.pop(key, None)
        self.environment = patch.dict(os.environ, env, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_parent_watch_observes_parent_exit(self):
        root=Path(tempfile.mkdtemp(prefix='benchmark-parent-',dir=os.environ.get('TMPDIR',str(Path.home()/'tmp'))))
        child = f"""import runpy,time
from pathlib import Path
Watch=runpy.run_path({str(path.resolve())!r})['ParentWatch']
with Watch() as parent:
    Path({str(root/'ready')!r}).touch()
    deadline=time.monotonic()+5
    while parent.alive() and time.monotonic()<deadline: time.sleep(.02)
    Path({str(root/'result')!r}).write_text('alive' if parent.alive() else 'gone')
"""
        launch = f"""import subprocess,sys,time
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c',{child!r}])
deadline=time.monotonic()+5
while not Path({str(root/'ready')!r}).exists():
    assert p.poll() is None
    assert time.monotonic()<deadline
    time.sleep(.02)
"""
        subprocess.run([sys.executable,'-c',launch],check=True,timeout=10)
        deadline=time.monotonic()+10
        while not (root/'result').exists():
            self.assertLess(time.monotonic(),deadline)
            time.sleep(.02)
        self.assertEqual((root/'result').read_text(),'gone')

    def test_identity_golden_and_replication(self):
        job=json.loads(path.with_name('benchmark_identity_v1.json').read_text())
        kind=execmod.validate_job(job)
        self.assertEqual(execmod.canonical_job_id(kind,job['inputs']),job['job_id'])
        for key,value in [('machine','cpu-b'),('domain','efficiency'),('replicate',1)]:
            changed=dict(kind,**{key:value})
            self.assertNotEqual(execmod.canonical_job_id(changed,job['inputs']),job['job_id'])
        bad=dict(job,job_id='0'*64)
        with self.assertRaises(ValueError):execmod.validate_job(bad)

    def test_materialize_and_corruption(self):
        root=Path(tempfile.mkdtemp(prefix='benchmark-store-',dir=os.environ.get('TMPDIR',str(Path.home()/'tmp'))))
        source=root/'input';source.write_bytes(b'known input')
        store=execmod.Store(root)
        value=store.put(source)
        manifest={'schema':1,'target':'test','plan':{},'files':{'data/input':{'sha256':value,'bytes':11,'executable':False}}}
        p=root/'manifest';p.write_text(json.dumps(manifest))
        sha=store.put(p)
        execmod.materialize(store,sha,root/'bundle')
        self.assertEqual((root/'bundle/data/input').read_bytes(),b'known input')
        (store.root/value).write_bytes(b'wrong input')
        with self.assertRaises(ValueError):store.get(value)

    def test_path_escape_is_refused(self):
        root=Path(tempfile.mkdtemp(prefix='benchmark-path-',dir=os.environ.get('TMPDIR',str(Path.home()/'tmp'))))
        store=execmod.Store(root)
        for index,name in enumerate(['../escape','/escape','C:/escape','dir\\escape']):
            p=root/f'manifest-{index}'
            p.write_text(json.dumps({'schema':1,'files':{name:{'sha256':'a'*64,'bytes':1,'executable':False}}}))
            with self.assertRaises(ValueError):execmod.materialize(store,store.put(p),root/f'bundle-{index}')

if __name__=='__main__':unittest.main()
