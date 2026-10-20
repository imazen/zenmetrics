#!/usr/bin/env python3
"""Resolve retained RGB paths before an author-model constructor control."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('python',type=Path);p.add_argument('config',type=Path)
    p.add_argument('metric');p.add_argument('pairs',type=Path)
    p.add_argument('--driver',type=Path,default=Path(__file__).with_name('canonical_author_network_cell.py'))
    a=p.parse_args();pairs=json.loads(a.pairs.read_text())
    if isinstance(pairs,dict):pairs=pairs['pairs']
    pair=next(row for row in pairs if row['size_name']=='cid22')
    paths=[str((a.pairs.parent/pair[role]).resolve(strict=True)) for role in ['reference','distorted']]
    command=[str(a.python.absolute()),'-B',str(a.driver.resolve(strict=True)),
             str(a.config.resolve()),a.metric,str(pair['width']),str(pair['height']),*paths,'1','3','--setup-only']
    os.execv(command[0],command)


if __name__=='__main__':main()
