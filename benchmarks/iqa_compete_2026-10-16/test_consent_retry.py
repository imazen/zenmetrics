"""New authorization applies only to exhausted consent refusals, never other faults."""
import unittest
from prepare_consent_retry import eligible


class ConsentRetry(unittest.TestCase):
    def test_only_exhausted_proven_consent_refusals_are_eligible(self):
        cell=dict(only_mode='score',width=512,height=512)
        records=[dict(record=dict(row=dict(error='requires explicit WeightsConsent')))]*2
        policy=dict(disabled_geometries=[[7680,4320]])
        self.assertTrue(eligible(cell,records,'exhausted',policy))
        for state in ['retained','pending']:
            self.assertFalse(eligible(cell,records,state,policy))
        self.assertFalse(eligible(cell,[],'exhausted',policy))
        self.assertFalse(eligible(dict(cell,only_mode='score+map'),records,'exhausted',policy))
        self.assertFalse(eligible(dict(cell,width=7680,height=4320),records,'exhausted',policy))
        for error in ['patch bound','ORT allocation failed','partial sample count']:
            self.assertFalse(eligible(cell,[dict(record=dict(row=dict(error=error)))],'exhausted',policy))
        self.assertFalse(eligible(cell,[*records,dict(record=dict(row=None))],'exhausted',policy))


if __name__ == '__main__': unittest.main()
