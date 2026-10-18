#!/usr/bin/env python3
"""Check every deployed extension page and its pinned source identity."""
import argparse
import json
from urllib.request import Request, urlopen
from urllib.parse import urljoin


def verify(url, source):
    def fetch(name):
        request=Request(urljoin(url,name),headers={'User-Agent':'zenmetrics-site-verification/1.0'})
        return urlopen(request,timeout=30)
    with fetch('manifest.json') as r:manifest=json.load(r)
    if manifest['source_commit']!=source:raise ValueError('deployed source commit differs')
    for name in manifest['pages']:
        with fetch(name) as r:
            text=r.read().decode()
            if r.status!=200 or source not in text:raise ValueError('deployed page identity differs: '+name)
        print('HTTP 200, source verified:',urljoin(url,name))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--source',required=True);a=p.parse_args();verify(a.url,a.source)
