"""Model copies preserve the initial inventory and refuse changed artifacts."""
import hashlib,json,pathlib,tempfile,unittest
from freeze_models import freeze
P=pathlib.Path
class Freeze(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=P.home()/'tmp');self.addCleanup(self.temp.cleanup);self.root=P(self.temp.name)
  self.default=self.root/'source';self.default.mkdir();self.weights=self.default/'fixture.rten';self.weights.write_bytes(b'original model')
  self.overrides=self.root/'overrides.json';self.overrides.write_text(json.dumps({'fixture':str(self.default)}))
  info=self.weights.stat();self.record=dict(directory=str(self.default),file=self.weights.name,source=str(self.weights),bytes=info.st_size,mtime_ns=info.st_mtime_ns,sha256=hashlib.sha256(self.weights.read_bytes()).hexdigest())
  self.manifest=self.root/'manifest.json';self.manifest.write_text(json.dumps(dict(artifacts=[self.record])));self.output=self.root/'frozen'
 def test_independent_verified_copy_and_override(self):
  original=self.weights.read_bytes();receipt=freeze(self.default,self.overrides,self.manifest,self.output)
  target=P(receipt['default'])/self.weights.name;self.assertEqual(target.read_bytes(),original)
  self.assertEqual(json.loads((self.output/'models.json').read_text())['fixture'],receipt['default'])
  self.weights.write_bytes(b'changed model');self.assertEqual(target.read_bytes(),original)
  self.assertTrue((self.output/'COMPLETE').exists());self.assertEqual(target.stat().st_mode&0o222,0)
 def test_changed_content_is_not_completed(self):
  self.record['sha256']='0'*64;self.manifest.write_text(json.dumps(dict(artifacts=[self.record])))
  with self.assertRaisesRegex(ValueError,'content differs'):freeze(self.default,self.overrides,self.manifest,self.output)
  self.assertFalse((self.output/'COMPLETE').exists());self.assertTrue((self.output/'PREPARATION_STARTED').exists())
 def test_new_export_requires_inventory_review(self):
  (self.default/'new.onnx').write_bytes(b'new export')
  with self.assertRaisesRegex(ValueError,'artifact set changed'):freeze(self.default,self.overrides,self.manifest,self.output)
  self.assertFalse(self.output.exists())
 def test_changed_link_target_is_not_completed(self):
  replacement=self.root/'replacement.rten';replacement.write_bytes(self.weights.read_bytes())
  self.weights.unlink();self.weights.symlink_to(replacement)
  with self.assertRaisesRegex(ValueError,'link target changed'):freeze(self.default,self.overrides,self.manifest,self.output)
  self.assertFalse((self.output/'COMPLETE').exists())
if __name__=='__main__':unittest.main()
