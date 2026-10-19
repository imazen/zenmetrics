import unittest
from prepare_gate_retry import selection, reserve, frozen_controller
from pathlib import Path
import tempfile
import os


def cell(attempts, clean):
    return dict(cell_id=str(attempts)+str(clean), status='ok',
                config=dict(timing_phase='warm', only_mode='score', width=512, height=512, threads=8, sample_target_ns=100_000_000, coverage_first=False), attempts=[{}]*attempts,
                row=dict(n=10, zenbench_samples=[dict(gate_clean=clean)]*10))


class GateRetry(unittest.TestCase):
    def test_maps_and_8k_cannot_enter_retry_scope(self):
        c=cell(1,False)
        c['config']['only_mode']='score+map'
        with self.assertRaisesRegex(ValueError,'scalar non-8K'):
            selection(dict(complete=True,cells=[c]),2)
        c['config']['only_mode']='score';c['config']['width']=7680
        with self.assertRaisesRegex(ValueError,'scalar non-8K'):
            selection(dict(complete=True,cells=[c]),2)

    def test_verifier_uses_frozen_runner_identity(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            bundle=Path(tmp)
            (bundle/'durable.py').write_text('from pathlib import Path\nSELF=str(Path(__file__).resolve())\n')
            self.assertEqual(frozen_controller(bundle).SELF,str(bundle/'durable.py'))

    def test_successor_reservation_cannot_be_reused(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            results=Path(tmp)
            reserve(results,'identity',results/'first')
            original=(results/'gate-retry-reservation.json').read_bytes()
            with self.assertRaises(FileExistsError):
                reserve(results,'identity',results/'second')
            self.assertEqual((results/'gate-retry-reservation.json').read_bytes(),original)

    def test_preserves_caps_and_never_repeats_clean_cells(self):
        report=dict(complete=True, cells=[cell(1,False),cell(2,False),cell(2,True)])
        groups, exhausted=selection(report,2)
        self.assertEqual(dict(groups),{1:[0]})
        self.assertEqual(exhausted,['2False'])
        report['complete']=False
        with self.assertRaisesRegex(ValueError,'terminal'):
            selection(report,2)

    def test_missing_rounds_are_unqualified_and_excess_history_refused(self):
        c=cell(1,True);c['row']['zenbench_samples'].pop()
        self.assertEqual(dict(selection(dict(complete=True,cells=[c]),2)[0]),{1:[0]})
        with self.assertRaisesRegex(ValueError,'exceeds'):
            selection(dict(complete=True,cells=[cell(3,False)]),2)


if __name__=='__main__':unittest.main()
