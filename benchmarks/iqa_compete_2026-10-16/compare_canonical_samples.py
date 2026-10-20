#!/usr/bin/env python3
"""Compare every canonical warm score against the unchanged numerical bound.

The first runner remains frozen for retained active plans. This entry point
extends its sample protocol for multithreaded references whose reductions vary
slightly between calls; every value remains in the acquired block record.
"""
import math

import compare_canonical as core

original_parse = core.parse
original_compare = core.compare
canonical_scores = []


def parse(text, protocol):
    global canonical_scores
    if protocol != 'samples':
        return original_parse(text, protocol)
    rows = [line for line in text.splitlines() if line.startswith('SAMPLE ')]
    if not rows:
        raise ValueError('canonical samples missing')
    parsed = [original_parse(line, protocol) for line in rows]
    canonical_scores = [row[0] for row in parsed]
    if not all(math.isfinite(score) for score in canonical_scores):
        raise ValueError('nonfinite canonical warm score')
    return canonical_scores[0], [row[1][0] for row in parsed], {'sample_scores': canonical_scores}


def compare_samples(ours, scores, absolute, relative=0):
    results = [original_compare(ours, score, absolute, relative) for score in scores]
    if not results:
        raise ValueError('empty canonical score set')
    passed = False if any(ok is False for _, ok in results) else None if absolute is None else True
    return max(delta for delta, _ in results), passed


def compare(ours, canonical, absolute, relative=0):
    return compare_samples(ours, canonical_scores or [canonical], absolute, relative)


def main():
    core.parse = parse
    core.compare = compare
    # The existing coordinator launches this entry point for each isolated block.
    core.__file__ = __file__
    core.main()


if __name__ == '__main__':
    main()
