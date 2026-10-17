"""Scientific regeneration controls; run with requirements-quality.txt installed."""
import unittest
import numpy as np
from correlate import measures


class CorrelationControl(unittest.TestCase):
    def test_signed_rank_and_linear_correlations(self):
        # A monotone nonlinear relationship keeps rank correlations at one;
        # raw Pearson must remain below one (no logistic remap was fitted).
        x=np.array([1.,2.,3.,4.,5.]);y=-x*x
        rank,linear,tau=measures(x,y)
        self.assertAlmostEqual(rank,-1.)
        self.assertAlmostEqual(tau,-1.)
        self.assertGreater(linear,-1.)
        self.assertLess(linear,-0.9)


if __name__=='__main__':unittest.main()
