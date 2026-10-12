#!/usr/bin/env python3
"""Require qualified output in each fresh A/B invocation, including A/A."""
import json
from pathlib import Path
import sys


def read(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    if path.stat().st_size > 8*1024*1024: raise ValueError('oversized checker metadata')
    return json.loads(path.read_bytes(),object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def check(baseline,candidate,expected):
    contract=read(expected)
    for directory in [baseline,candidate]:
        result=read(directory/'driver.json')
        if (set(result)!={'schema','operations','correctness','memory'} or result['schema']!=1
                or result['operations']!=contract['operations'] or result['correctness']!=contract['correctness']
                or result['memory']!=[]):
            raise ValueError('pair invocation differs from qualified ordinary-driver output')

if __name__=='__main__':
    try: check(*(Path(p).resolve() for p in sys.argv[1:]))
    except (ValueError,OSError,TypeError) as error: sys.exit(f'pair correctness refused: {error}')
