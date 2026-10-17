import pathlib,tempfile,unittest
from patch_luma_maps import patch,WRAPPERS
P=pathlib.Path

class LumaPatch(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=P.home()/'tmp');self.addCleanup(self.tmp.cleanup)
        self.root=P(self.tmp.name);self.source=self.root/'source';self.snapshot=self.root/'snapshot'
        for crate in WRAPPERS:
            relative=P('crates')/crate/'src/lib.rs'
            for base in [self.source,self.snapshot]:
                path=base/relative;path.parent.mkdir(parents=True);path.write_text('// fixture original\n')
    def test_source_stays_identical_and_snapshot_only_appends(self):
        receipt=patch(self.snapshot,self.source,self.root/'receipt.json')
        self.assertEqual(len(receipt['files']),2)
        for crate in WRAPPERS:
            relative=P('crates')/crate/'src/lib.rs'
            self.assertEqual((self.source/relative).read_text(),'// fixture original\n')
            self.assertEqual((self.snapshot/relative).read_text(),'// fixture original\n'+WRAPPERS[crate])
    def test_source_alias_refuses_without_writing(self):
        with self.assertRaises(ValueError):patch(self.source,self.source,self.root/'receipt.json')
        self.assertEqual((self.source/'crates/fsim/src/lib.rs').read_text(),'// fixture original\n')
    def test_unexpected_prior_adaptation_refuses(self):
        (self.snapshot/'crates/fsim/src/lib.rs').write_text('// changed snapshot\n')
        with self.assertRaises(ValueError):patch(self.snapshot,self.source,self.root/'receipt.json')
        self.assertFalse((self.root/'receipt.json').exists())
if __name__=='__main__':unittest.main()
