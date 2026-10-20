#!/usr/bin/env python3
"""Run an explicitly single-thread reference under a declared worker budget."""
import json
import os
import sys


def main():
    if len(sys.argv) != 8:
        raise ValueError('binary width height reference distorted worker-budget samples')
    binary, width, height, reference, distorted, budget, samples = sys.argv[1:]
    if min(int(width), int(height), int(budget), int(samples)) < 1:
        raise ValueError('positive geometry, budget and samples required')
    print('CANONICAL ' + json.dumps(dict(requested_worker_budget=int(budget),
          effective_threads=1, qualification='Original single-thread callable; numerical body unchanged')),
          flush=True)
    os.execv(binary, [binary, width, height, reference, distorted, '1', samples])


if __name__ == '__main__':
    main()
