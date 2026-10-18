#!/usr/bin/env python3
"""Retain actual public mainland HTTP probe rows, with service and target identity.

The optional browser probe invokes the public 17CE UI. It does not change DNS,
hosting or account settings. Every missing/error response is retained.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit


def probe(url, output, seconds=40):
    from playwright.sync_api import sync_playwright
    output.mkdir(parents=True,exist_ok=True)
    frames=[];sockets=[];dialogs=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=os.environ.get('DASHBOARD_CHROMIUM'))
        page=browser.new_page()
        page.on('dialog',lambda d:(dialogs.append(d.message),d.dismiss()))
        def connected(ws):
            sockets.append(ws.url)
            ws.on('framereceived',lambda payload:frames.append({'direction':'received','payload':payload}))
            ws.on('framesent',lambda payload:frames.append({'direction':'sent','payload':payload}))
        page.on('websocket',connected)
        page.goto('https://www.17ce.com/',wait_until='domcontentloaded',timeout=30000)
        page.wait_for_function('typeof ajax_check === "function"')
        page.locator('#url').fill(url)
        page.evaluate('ajax_check(0,1,0)')
        # Chunked waits keep progress visible and the total bounded.
        for elapsed in range(0,seconds,5):
            page.wait_for_timeout(5000)
            print('17CE elapsed',elapsed+5,'s; frames',len(frames),flush=True)
        text=page.locator('body').inner_text()
        (output/'page.txt').write_text(text)
        (output/'page.html').write_text(page.content())
        links=page.locator('a[href]').evaluate_all('(xs)=>xs.map(x=>({text:x.textContent,href:x.href}))')
        (output/'links.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in links))
        rows=page.locator('table tr').evaluate_all('(xs)=>xs.map(x=>Array.from(x.querySelectorAll("td")).map(c=>c.innerText.trim()))')
        share=page.locator('.resulturl').evaluate_all('(xs)=>xs.map(x=>x.value)')
        receipt=dict(observed_utc=datetime.now(timezone.utc).isoformat(),service='17CE',service_url='https://www.17ce.com/',target=url,form_value=page.locator('#url').input_value(),share_urls=share,dialogs=dialogs,sockets=sockets,rows=rows,frames=frames)
        (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
        browser.close()
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=40);a=p.parse_args()
    if urlsplit(a.url).scheme!='https' or a.seconds<5 or a.seconds>60:p.error('HTTPS and 5–60 second bound required')
    result=probe(a.url,a.output,a.seconds)
    print('Retained rows',len(result['rows']),'share',result['share_urls'],'dialogs',result['dialogs'])
