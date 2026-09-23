#!/usr/bin/env python3
"""Regression test: pycvvdp_worker must call `predict(test, reference)`.

pycvvdp's signature is `cvvdp.predict(test_cont, reference_cont, dim_order)`.
The worker passed `(ref, dist)` until 2026-09-22, silently scoring every pair
with reference and test swapped (benchmarks/cvvdp_aic_discrepancy_2026-09-22.md).
CVVDP is not symmetric in its arguments, so the swapped scores were close to the
correct ones but not equal, which is why nothing flagged it.

Hermetic: a recording stand-in replaces pycvvdp, so this needs no torch, no
images and no network. Run standalone
(`python3 scripts/sweep/test_pycvvdp_worker.py`) or under pytest.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pycvvdp_worker as W  # noqa: E402


class RecordingMetric:
    """Mimics pycvvdp's `predict(test_cont, reference_cont, dim_order)`."""

    def __init__(self):
        self.calls = []

    def predict(self, test_cont, reference_cont, dim_order="BCFHW"):
        self.calls.append((test_cont, reference_cont, dim_order))
        return 7.5, None


def test_score_pair_passes_distorted_as_test_and_reference_second():
    ref = np.zeros((4, 5, 3), dtype=np.uint8)
    dist = np.full((4, 5, 3), 9, dtype=np.uint8)
    m = RecordingMetric()
    assert W.score_pair(m, ref, dist) == 7.5
    (test_cont, reference_cont, dim_order), = m.calls
    assert test_cont is dist, "first predict() argument must be the DISTORTED image"
    assert reference_cont is ref, "second predict() argument must be the REFERENCE image"
    assert dim_order == "HWC"


def test_score_pair_rejects_shape_mismatch():
    m = RecordingMetric()
    try:
        W.score_pair(m, np.zeros((4, 5, 3), np.uint8), np.zeros((5, 4, 3), np.uint8))
    except ValueError:
        return
    raise AssertionError("shape mismatch must raise")


def test_default_column_names_the_installed_reference():
    orig = W.installed_pycvvdp_version
    try:
        W.installed_pycvvdp_version = lambda: "0.5.7"
        assert W.default_score_col_name() == "cvvdp_pycvvdp_v057"
        W.installed_pycvvdp_version = lambda: "0.5.4"
        assert W.default_score_col_name() == "cvvdp_pycvvdp_v054"
    finally:
        W.installed_pycvvdp_version = orig


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
