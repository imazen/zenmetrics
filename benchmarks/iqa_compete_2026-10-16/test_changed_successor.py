import unittest
from prepare_changed_successor import select

class Selection(unittest.TestCase):
    def test_no_completed_repeats_or_cap_reset(self):
        cells=[dict(cell_id=str(i),config=dict(metric_name=m),status=s,attempts=[{}]*n)
               for i,(m,s,n) in enumerate([('changed','ok',1),('changed','pending',1),
                                          ('changed','failed',2),('unchanged','pending',0)])]
        groups,retained,excluded,exhausted=select(cells,{'changed'},2)
        self.assertEqual(dict(groups),{1:[1]})
        self.assertEqual(retained,['0'])
        self.assertEqual(excluded,['3'])
        self.assertEqual(exhausted,['2'])

    def test_refuse_cap_overrun(self):
        with self.assertRaises(ValueError):
            select([dict(cell_id='x',config=dict(metric_name='changed'),status='pending',attempts=[1,2,3])],{'changed'},2)
