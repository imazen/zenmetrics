#!/usr/bin/env python3
"""Check every exported DOI/arXiv identifier with curl, without reference code.

A DOI resolver redirect is resolution success; publisher access may require a
subscription. arXiv abstract URLs must return success directly. No PDFs, code
or model checkpoints are downloaded.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from model_cards import cards


def verify(url):
    result=subprocess.run(['curl','--silent','--show-error','--retry','2','--retry-delay','1',
                           '--max-time','20','--output','/dev/null','--write-out','%{http_code}',url],
                          capture_output=True,text=True)
    code=int(result.stdout) if result.stdout.isdigit() else 0
    good=result.returncode==0 and 200<=code<400
    return dict(url=url,http_status=code,passed=good,details=result.stderr.strip())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    urls=sorted({c['citation']['url'] for c in cards() if c['citation']})
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(verify,urls))
    args.report.write_text(json.dumps(results,indent=2)+'\n')
    for row in results:print(('PASS' if row['passed'] else 'FAIL')+' '+str(row['http_status'])+' '+row['url'])
    if not all(row['passed'] for row in results):raise SystemExit('Citation identifiers failed to resolve; see report')
    print(f'{len(results)}/{len(results)} DOI/arXiv identifiers resolve')
