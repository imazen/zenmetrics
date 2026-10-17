#!/usr/bin/env python3
"""Serve the actual built artifact and exercise browser-visible behavior."""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright


def check(directory, screenshots):
    handler=functools.partial(SimpleHTTPRequestHandler, directory=str(directory))
    server=ThreadingHTTPServer(('0.0.0.0', 3387),handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=os.environ.get('DASHBOARD_CHROMIUM'))
            page=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
            page.on('response',lambda r:errors.append(f'HTTP {r.status}: {r.url}') if r.status>=400 else None)
            for view in ['speed','agreement','quality','catalog']:
                page.goto('http://127.0.0.1:3387/#'+view)
                page.wait_for_function('(v)=>window.dashboardReady===v',arg=view)
                assert page.locator('#table tbody tr').count()>0, view
                if view!='catalog':assert page.locator('#chart svg').count()>0,view
                assert page.locator('#provenance').inner_text(),view
                page.locator('th button').first.click()
                if screenshots:
                    screenshots.mkdir(parents=True,exist_ok=True)
                    page.screenshot(path=str(screenshots/(view+'.png')),full_page=True)
            page.goto('http://127.0.0.1:3387/#speed?metric=butter&threads=1')
            page.wait_for_function('window.dashboardReady==="speed"')
            assert page.locator('[data-filter="metric"]').input_value()=='butter'
            assert page.locator('#table tbody tr').count()==14
            with page.expect_download() as downloaded:page.locator('#download').click()
            csv=Path(downloaded.value.path()).read_text()
            assert 'butter' in csv and 'zensim' not in csv
            page.reload();page.wait_for_function('window.dashboardReady==="speed"')
            assert page.locator('[data-filter="threads"]').input_value()=='1'
            page.locator('[data-filter="mode"]').select_option('cold')
            page.wait_for_function('location.hash.includes("mode=cold")')
            page.wait_for_function('window.dashboardReady==="speed"')
            assert page.locator('#table tbody tr').count()==7
            page.locator('#theme').click();assert page.locator('body').get_attribute('class')=='dark'
            page.reload();page.wait_for_function('window.dashboardReady==="speed"')
            assert page.locator('body').get_attribute('class')=='dark'
            page.goto('http://127.0.0.1:3387/#quality?q=no-such-measured-metric')
            page.wait_for_function('window.dashboardReady==="quality"')
            assert page.locator('#table tbody tr').count()==0
            page.set_viewport_size({'width':390,'height':844})
            page.goto('http://127.0.0.1:3387/#catalog');page.wait_for_function('window.dashboardReady==="catalog"')
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'), 'mobile page overflow'
            if screenshots:page.screenshot(path=str(screenshots/'mobile.png'),full_page=True)
            browser.close()
        assert not errors, errors
        print(json.dumps({'pages':4,'console_errors':errors,'filters':'pass','csv':'pass','URL_restore':'pass','mobile':'pass','theme':'pass'}))
    finally:server.shutdown();server.server_close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--screenshots',type=Path)
    args=parser.parse_args();check(args.directory,args.screenshots)
