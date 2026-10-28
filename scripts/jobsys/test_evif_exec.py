"""Failure gates and real-worker caller contract for EVIF jobs."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from evif_exec import validate, TEACHER

class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=str(Path.home()/'tmp')); self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'ref.png'
        self.data=b'\x89PNG\r\n\x1a\n'+b'\0'*8+struct.pack('>II',32,32)
        self.path.write_bytes(self.data)
        self.params={'quality':95,'xyb':True}
        digest=lambda b:hashlib.sha256(b).hexdigest()
        self.job=dict(kind=dict(kind='encode',codec='evif:zenjpeg-expert',q=95,knobs=json.dumps(self.params)),inputs=[digest(self.data),digest(json.dumps(self.params,sort_keys=True,separators=(',',':')).encode())],cell=dict(image_path=str(self.path),codec='evif:zenjpeg-expert',q=95,knob_tuple_json=json.dumps(self.params)))
    def test_encode_and_frozen_metric_identity(self):
        self.assertEqual(validate(self.job)[0],'zenjpeg-expert')
        self.job['kind']=dict(kind='metric',metric=TEACHER)
        self.assertEqual(validate(self.job)[1],self.params)
    def test_conflicting_encode_params_and_current_judge_refused(self):
        self.job['kind']['knobs']='{"quality":80}'
        with self.assertRaises(ValueError):validate(self.job)
        self.job['kind']=dict(kind='metric',metric='ssim2')
        with self.assertRaises(ValueError):validate(self.job)
    def test_unbound_source_and_large_source_refused(self):
        self.job['inputs']=[]
        with self.assertRaises(ValueError):validate(self.job)
        self.path.write_bytes(self.data[:16]+struct.pack('>II',1000,1000))
        with self.assertRaises(ValueError):validate(self.job)
if __name__=='__main__':unittest.main()
