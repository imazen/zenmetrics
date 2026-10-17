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


def check(directory, screenshots, port=3387):
    if not 3000<=port<=3999:raise ValueError('dashboard port must be 3000–3999')
    manifest=json.loads((directory/'data/manifest.json').read_text())
    handler=functools.partial(SimpleHTTPRequestHandler, directory=str(directory))
    server=ThreadingHTTPServer(('0.0.0.0', port),handler)
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
                page.goto(f'http://127.0.0.1:{port}/#'+view)
                page.wait_for_function('(v)=>window.dashboardReady===v',arg=view)
                assert page.locator('#table tbody tr').count()==min(2000,manifest['quality_correlation_count'] if view=='quality' else manifest['counts'][view]), view
                if view=='quality' and manifest['quality_point_count']:
                    assert page.locator('#secondary svg').count()>0, 'real per-image scatter missing'
                if view!='catalog':assert page.locator('#chart svg').count()>0,view
                assert page.locator('#provenance').inner_text(),view
                page.locator('th button').first.click()
                if screenshots:
                    screenshots.mkdir(parents=True,exist_ok=True)
                    page.screenshot(path=str(screenshots/(view+'.png')),full_page=True)
            from urllib.parse import urlencode
            source='benchmarks/metric_loop_st_2026-08-30.tsv'
            params=urlencode(dict(source=source,threads='1'))
            page.goto(f'http://127.0.0.1:{port}/#speed?'+params)
            page.wait_for_function('window.dashboardReady==="speed"')
            assert page.locator('[data-filter="source"]').input_value()==source
            assert page.locator('#table tbody tr').count()==manifest['source_counts'][source]
            with page.expect_download() as downloaded:page.locator('#download').click()
            import csv as csv_module
            import io
            downloaded_rows=list(csv_module.DictReader(io.StringIO(Path(downloaded.value.path()).read_text())))
            assert len(downloaded_rows)==manifest['source_counts'][source]
            assert all(row['source']==source for row in downloaded_rows)
            page.reload();page.wait_for_function('window.dashboardReady==="speed"')
            assert page.locator('[data-filter="threads"]').input_value()=='1'
            page.locator('[data-filter="mode"]').select_option('cold')
            page.wait_for_function('location.hash.includes("mode=cold")')
            page.wait_for_function('window.dashboardReady==="speed"')
            assert page.locator('#table tbody tr').count()==manifest['source_mode_counts'][source]['cold']
            page.locator('#theme').click();assert page.locator('body').get_attribute('class')=='dark'
            page.reload();page.wait_for_function('window.dashboardReady==="speed"')
            assert page.locator('body').get_attribute('class')=='dark'
            source=next(s for s in manifest['source_counts'] if s.startswith('benchmarks/dashboard/quality/cid22-correlations-'))
            quality=json.loads((directory/'data/quality.json').read_text())
            dataset=next(row['dataset'] for row in quality if row.get('source')==source)
            page.goto(f'http://127.0.0.1:{port}/#quality?'+urlencode(dict(dataset=dataset)))
            page.wait_for_function('window.dashboardReady==="quality"')
            assert page.locator('#table tbody tr').count()==manifest['source_counts'][source]
            assert page.locator('#secondary svg').count()>0
            if screenshots:page.screenshot(path=str(screenshots/'quality-cid22.png'),full_page=True)
            page.goto(f'http://127.0.0.1:{port}/#quality?q=no-such-measured-metric')
            page.wait_for_function('window.dashboardReady==="quality"')
            assert page.locator('#table tbody tr').count()==0
            page.set_viewport_size({'width':390,'height':844})
            page.goto(f'http://127.0.0.1:{port}/#catalog');page.wait_for_function('window.dashboardReady==="catalog"')
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'), 'mobile page overflow'
            if screenshots:page.screenshot(path=str(screenshots/'mobile.png'),full_page=True)
            browser.close()
        assert not errors, errors
        print(json.dumps({'pages':4,'console_errors':errors,'filters':'pass','csv':'pass','URL_restore':'pass','mobile':'pass','theme':'pass'}))
    except Exception:
        print(json.dumps({'browser_errors':errors}),flush=True)
        raise
    finally:server.shutdown();server.server_close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--screenshots',type=Path);parser.add_argument('--port',type=int,default=int(os.environ.get('DASHBOARD_PORT','3387')))
    args=parser.parse_args();check(args.directory,args.screenshots,args.port)
