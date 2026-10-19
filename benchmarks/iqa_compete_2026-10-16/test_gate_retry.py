import unittest
from prepare_gate_retry import selection, reserve
from pathlib import Path
import tempfile
import os


def cell(attempts, clean):
    return dict(cell_id=str(attempts)+str(clean), status='ok',
                config=dict(timing_phase='warm'), attempts=[{}]*attempts,
                row=dict(n=10, zenbench_samples=[dict(gate_clean=clean)]*10))


class GateRetry(unittest.TestCase):
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
