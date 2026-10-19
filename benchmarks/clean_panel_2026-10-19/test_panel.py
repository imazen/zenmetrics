import copy
import unittest
import panel


class Qualification(unittest.TestCase):
    def setUp(self):
        self.m = {'inputs':[{'sha256':'input'}], 'arms':{'ours':{'build_identity':'rust','sample_count':1},'canonical':{'build_identity':'oracle','sample_count':1}}, 'scope':{'ours':'resident-rgb-call','canonical':'resident-rgb-call'},'parity_abs_bound':0.001}
        self.b = {'block_id':'a','manifest_sha256':panel.identity(self.m),'arms':[]}
        for name,time in [('ours',20),('canonical',10)]:
            self.b['arms'].append(dict(arm=name,block_id='a',build_identity=self.m['arms'][name]['build_identity'],input_identity=panel.identity(self.m['inputs']),scope=self.m['scope'][name],exit_code=0,contaminated=False,samples=[dict(ns=time,gate_clean=True)],score=1.0))

    def test_actual_block(self):
        self.assertEqual(panel.qualify_block(self.b,self.m),2)

    def test_rejections(self):
        for key,value in [('block_id','foreign'),('build_identity','other'),('input_identity','other'),('scope','compare-only'),('contaminated',True),('exit_code',1),('score',1.1)]:
            with self.subTest(key=key):
                b=copy.deepcopy(self.b);b['arms'][0][key]=value
                with self.assertRaises(ValueError):panel.qualify_block(b,self.m)
        b=copy.deepcopy(self.b);b['arms'][0]['samples'].append(dict(ns=20,gate_clean=False))
        with self.assertRaises(ValueError):panel.qualify_block(b,self.m)

    def test_duplicate_and_missing(self):
        with self.assertRaises(ValueError):panel.summarize([self.b,self.b],self.m)
        self.b['arms'].pop()
        with self.assertRaises(ValueError):panel.qualify_block(self.b,self.m)

    def test_relative_precision(self):
        self.assertFalse(panel.precision(.48,.5,.52)) # absolute halfwidth .02 hides 4%
        self.assertTrue(panel.precision(1.95,2,2.05)) # absolute halfwidth .05, relative 2.5%
        self.assertFalse(panel.precision(2.01,2,2.02))

    def test_no_inner_sample_pairing(self):
        self.m['arms']['ours']['sample_count']=3
        self.m['arms']['canonical']['sample_count']=3
        self.b['manifest_sha256']=panel.identity(self.m)
        self.b['arms'][0]['samples']=[dict(ns=v,gate_clean=True) for v in [2,20,200]]
        self.b['arms'][1]['samples']=[dict(ns=v,gate_clean=True) for v in [100,10,1]]
        self.assertEqual(panel.qualify_block(self.b,self.m),2)

    def test_pilot_never_claims_precision(self):
        self.assertNotIn('estimate',panel.summarize([self.b],self.m))
        self.assertFalse(panel.summarize([self.b],self.m)['qualified'])

    def test_artifact_mutation(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source';path.write_text('frozen')
            manifest=copy.deepcopy(self.m)
            manifest.update(metric='brisque',shape=[512,512,1,'score'],artifacts=[dict(path=str(path),sha256=panel.digest(path))])
            manifest['inputs']=manifest['artifacts']
            for arm in manifest['arms'].values():arm['argv']=['scorer']
            panel.verify(manifest)
            path.write_text('changed')
            with self.assertRaises(ValueError):panel.verify(manifest)

    def test_missing_sample_and_cross_manifest(self):
        b=copy.deepcopy(self.b);b['arms'][0]['samples']=[]
        with self.assertRaises(ValueError):panel.qualify_block(b,self.m)
        b=copy.deepcopy(self.b);b['manifest_sha256']='other'
        with self.assertRaises(ValueError):panel.qualify_block(b,self.m)

    def test_acquisition_holds_both_locks_and_preserves_blocks(self):
        import tempfile
        import subprocess
        import sys
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory);(home/'tmp').mkdir()
            m=copy.deepcopy(self.m)
            def arm(name,manifest,block_id,output):
                probe="import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
                for lock in ['lilith-heavy.lock','lilith-wsl-heavy.lock']:
                    result=subprocess.run([sys.executable,'-c',probe,str(home/'tmp'/lock)],capture_output=True)
                    self.assertNotEqual(result.returncode,0)
                value=copy.deepcopy(next(a for a in self.b['arms'] if a['arm']==name))
                value['block_id']=block_id
                return value
            with patch.object(panel.Path,'home',return_value=home),patch.object(panel,'verify'),patch.object(panel,'run_arm',side_effect=arm):
                blocks=panel.acquire(m,home/'pilot',2)
            self.assertEqual([b['order'] for b in blocks],[['ours','canonical'],['canonical','ours']])
            self.assertNotEqual(blocks[0]['block_id'],blocks[1]['block_id'])
            for b in blocks:self.assertTrue((home/'pilot'/b['block_id']/'block.json').exists())
            with self.assertRaises(FileExistsError):panel.acquire(m,home/'pilot',1)

    def test_post_attempt_change_preserves_rejection(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory);(home/'tmp').mkdir()
            def arm(name,manifest,block_id,output):
                value=copy.deepcopy(next(a for a in self.b['arms'] if a['arm']==name))
                value['block_id']=block_id
                return value
            with patch.object(panel.Path,'home',return_value=home),patch.object(panel,'verify',side_effect=[None,ValueError('source changed')]),patch.object(panel,'run_arm',side_effect=arm):
                blocks=panel.acquire(self.m,home/'pilot',1)
            self.assertEqual(blocks[0]['identity_error'],'source changed')
            result=panel.summarize(blocks,self.m)
            self.assertEqual(result['accepted_blocks'],0)
            self.assertEqual(len(result['rejected']),1)

    def test_fixed_look_estimator_and_crossing(self):
        blocks=[]
        for index,(a,b) in enumerate([(10,1),(8,3),(6,9)]*4):
            block=copy.deepcopy(self.b);block['block_id']=str(index)
            for arm,ns in zip(block['arms'],[a,b]):
                arm['block_id']=str(index);arm['samples'][0]['ns']=ns
            blocks.append(block)
        self.assertEqual(panel.summarize(blocks,self.m)['estimate'],8/3)
        for block in blocks:
            block['arms'][0]['samples'][0]['ns']=108
            block['arms'][1]['samples'][0]['ns']=100
        result=panel.summarize(blocks,self.m)
        self.assertTrue(result['threshold_crossing'])
        self.assertFalse(result['qualified'])

if __name__=='__main__':unittest.main()
