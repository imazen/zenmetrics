#!/usr/bin/env python3
"""Check the landing navigation and phone/desktop layout on a packaged site."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('site',type=Path)
parser.add_argument('--report',type=Path,required=True)
parser.add_argument('--chromium',help='explicit installed Chromium executable')
args=parser.parse_args()
server=ThreadingHTTPServer(('0.0.0.0',3371),partial(SimpleHTTPRequestHandler,directory=str(args.site)))
thread=Thread(target=server.serve_forever,daemon=True);thread.start()
results=[]
try:
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'],executable_path=args.chromium)
        for width,theme in [(1440,'light'),(390,'dark')]:
            page=browser.new_page(viewport={'width':width,'height':1000},color_scheme=theme)
            page.goto('http://127.0.0.1:3371/',wait_until='networkidle')
            links=page.locator('nav[aria-label="Results"] a').evaluate_all('(nodes)=>nodes.map(n=>n.getAttribute("href"))')
            assert len(links)==5,links
            assert 'https://gigapixels.imazen.org/' in links
            # Benchmark evidence lives on the metric brief; the workbench links there and hosts no /aic2026/ pages.
            assert 'https://metric-brief.pages.dev/' in links and not any(l.startswith('/aic2026') for l in links),links
            assert page.locator('.evidence-header').count()==1
            overflow=page.evaluate('document.documentElement.scrollWidth > innerWidth')
            assert not overflow,(width,theme)
            results.append({'width':width,'theme':theme,'links':links,'overflow':overflow})
            page.close()
        browser.close()
finally:
    server.shutdown();server.server_close();thread.join()
args.report.parent.mkdir(parents=True,exist_ok=True)
args.report.write_text(json.dumps(results,indent=2)+'\n')
print('Landing navigation/layout checks pass for desktop light and phone dark')
