"""Same hardware under Windows and Linux must remain separate observations."""
import json,pathlib,subprocess,sys,tempfile,unittest
P=pathlib.Path
class Platforms(unittest.TestCase):
    def test_operating_systems_are_not_folded_together(self):
        with tempfile.TemporaryDirectory(dir=P.home()/'tmp') as directory:
            root=P(directory);raw=root/'raw.jsonl';out=root/'collected.jsonl'
            row=dict(host='fixture',metric_name='fixture',impl='zenmetrics',device='cpu',threads=1,mode='score+map',size_name='fixture',status='unsupported',n=0,notes='Fixture has no map')
            records=[row|dict(os='Windows-fixture',marker='old'),row|dict(os='Linux-fixture',marker='linux',contaminated=True,contamination_classification='contaminated'),row|dict(os='Windows-fixture',marker='new',contaminated=False,contamination_classification='clean')]
            raw.write_text(''.join(json.dumps(r)+'\n' for r in records))
            result=subprocess.run([sys.executable,str(P(__file__).with_name('collect.py')),str(raw),'--output',str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            collected=[json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(collected),2)
            self.assertEqual({r['os']:r['marker'] for r in collected},{'Windows-fixture':'new','Linux-fixture':'linux'})
            self.assertEqual({r['os']:r['timing_quality'] for r in collected},{'Windows-fixture':'clean','Linux-fixture':'contaminated'})
if __name__=='__main__':unittest.main()
