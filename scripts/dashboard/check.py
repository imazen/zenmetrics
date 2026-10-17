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
            for view in ['speed','agreement','quality','complexity','catalog']:
                page.goto(f'http://127.0.0.1:{port}/#'+view)
                page.wait_for_function('(v)=>window.dashboardReady===v',arg=view)
                assert page.locator('#table tbody tr').count()==min(2000,manifest['quality_correlation_count'] if view=='quality' else manifest['counts'][view]), view
                if view=='quality' and manifest['quality_point_count']:
                    assert 'scatter was removed' in page.locator('#secondary').inner_text(), 'per-image note missing'
                if view!='catalog':assert page.locator('#chart svg').count()>0,view
                assert page.locator('#chart circle, #secondary circle').count()==0, view+': scatter marks remain'
                assert 'pending cleaner controlled runs' in page.locator('#provisional').inner_text()
                if view in ['speed','agreement']:
                    assert page.locator('#chart-mode [data-mode]').count()==2, view
                    assert page.locator('#chart .bar').count()>0, view+': bars missing'
                    assert page.locator('#chart .whisker').count()==0, view+': default must be plain bars'
                    page.locator('#chart-mode [data-mode="whiskers"]').click()
                    page.wait_for_function('(v)=>window.dashboardReady===v && location.hash.includes("chart=whiskers")',arg=view)
                    assert page.locator('#chart-mode [aria-pressed="true"]').inner_text()=='Bars + whiskers'
                assert page.locator('#provenance').inner_text(),view
                page.locator('th button').first.click()
                if screenshots:
                    screenshots.mkdir(parents=True,exist_ok=True)
                    page.screenshot(path=str(screenshots/(view+'.png')),full_page=True)
            page.goto(f'http://127.0.0.1:{port}/#ranking')
            page.wait_for_function('window.dashboardReady==="ranking"')
            assert page.locator('.rank-bar').count()>0
            assert page.locator('#missing-ranking').count()==1
            assert 'size=' in page.url
            if screenshots:page.screenshot(path=str(screenshots/'ranking.png'),full_page=True)
            assert page.locator('.rank-bar .whisker').count()==0
            page.locator('#chart-mode [data-mode="whiskers"]').click()
            page.wait_for_function('window.dashboardReady==="ranking" && location.hash.includes("layout=whiskers")')
            page.locator('#chart-mode [data-mode="all sizes"]').click()
            page.wait_for_function('window.dashboardReady==="ranking" && document.querySelector(".rank-multiples")')
            assert page.locator('.rank-multiples>section').count()>=14
            if screenshots:page.screenshot(path=str(screenshots/'ranking-all-sizes.png'),full_page=True)
            page.locator('#chart-mode [data-mode="rank change"]').click()
            page.locator('svg[aria-label="Rank change between resolutions"]').wait_for()
            if screenshots:page.screenshot(path=str(screenshots/'ranking-slope.png'),full_page=True)
            page.goto(f'http://127.0.0.1:{port}/#models')
            page.wait_for_function('window.dashboardReady==="models"')
            assert page.locator('.model-cell .model-card').count()==manifest['counts']['models']
            unimplemented=json.loads((directory/'data/unimplemented.json').read_text())
            assert unimplemented and page.locator('.unimplemented-card').count()==len(unimplemented)
            assert page.locator('.unimplemented-card .model-badge.unimplemented').count()==len(unimplemented)
            implemented={c['metric'] for c in json.loads((directory/'data/models.json').read_text())}
            assert not any(m['name'] in implemented for m in unimplemented)
            page.locator('.model-cell [data-star="ssim2"]').click()
            assert page.locator('.model-cell [data-star="ssim2"]').get_attribute('aria-pressed')=='true'
            page.goto(f'http://127.0.0.1:{port}/#models?starred=1')
            page.wait_for_function('window.dashboardReady==="models" && location.hash.includes("starred=1")')
            assert page.locator('.model-cell .model-card').count()==1 and page.locator('.unimplemented-card').count()==0
            page.locator('.model-cell [data-star="ssim2"]').click()
            page.wait_for_function('document.querySelectorAll(".model-cell .model-card").length===0')
            favorites=json.loads((directory/'data/favorites.json').read_text())
            page.goto(f'http://127.0.0.1:{port}/#models?priority=1');page.wait_for_function('window.dashboardReady==="models" && location.hash.includes("priority=1")')
            expected=[n for f in favorites['favorites'] for n in f['cli_names'] if n in implemented]
            shown=page.locator('.model-cell .model-card').evaluate_all('(cards)=>cards.map(c=>c.dataset.model)')
            assert sorted(shown)==sorted(expected), (shown, expected)
            ranks=[next(i for i,f in enumerate(favorites['favorites']) if m in f['cli_names']) for m in shown]
            assert ranks==sorted(ranks), 'priority order'
            assert page.locator('.model-badge.priority').count()==len(shown)
            page.goto(f'http://127.0.0.1:{port}/#models');page.wait_for_function('window.dashboardReady==="models"')
            assert page.locator('.model-export a').count()==3
            if screenshots:page.screenshot(path=str(screenshots/'models.png'))
            page.locator('[data-model="lpips-alex"]').click()
            page.locator('#model-detail').wait_for()
            assert page.locator('#model-detail').evaluate('(d)=>d.open')
            assert 'Richard' in page.locator('#model-detail').inner_text()
            repro=page.locator('#repro-text').inner_text()
            assert 'zenmetrics score --metric lpips-alex' in repro and '--features nn' in repro and 'ZENMETRICS_NN_MODEL_DIR' in repro
            assert 'BSD-2-Clause' in page.locator('#model-detail').inner_text()
            page.locator('#model-detail summary').filter(has_text='BibTeX').click()
            assert page.locator('#model-detail .bibtex').inner_text().startswith('@misc{')
            page.context.grant_permissions(['clipboard-read','clipboard-write'])
            page.locator('#copy-bib').click()
            page.wait_for_function('document.querySelector("#copy-result").textContent==="Copied."')
            if screenshots:page.screenshot(path=str(screenshots/'model-lpips.png'))
            page.locator('#close-model').click()
            page.wait_for_function('!document.querySelector("#model-detail")')
            for suffix in ['json','csv','bib']:
                with page.expect_download() as cards_download:
                    page.locator(f'.model-export a[href$=".{suffix}"]').click()
                content=Path(cards_download.value.path()).read_text()
                assert 'lpips' in content.lower()
                if suffix=='json':assert len(json.loads(content))==manifest['counts']['models']
            page.locator('[data-model-filter="kind"]').select_option('NR')
            page.wait_for_function('window.dashboardReady==="models" && location.hash.includes("kind=NR")')
            assert 0<page.locator('.model-card').count()<manifest['counts']['models']
            assert all(x=='NR' for x in page.locator('.model-kind').all_text_contents())
            page.set_viewport_size({'width':390,'height':844})
            page.goto(f'http://127.0.0.1:{port}/#models?metric=pieapp')
            page.wait_for_function('window.dashboardReady==="models"')
            assert 'Non-Commercial' in page.locator('#model-detail').inner_text()
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            assert page.locator('#model-detail').evaluate('(d)=>d.scrollWidth<=d.clientWidth+2')
            if screenshots:page.screenshot(path=str(screenshots/'model-mobile.png'))
            page.locator('#close-model').click()
            page.set_viewport_size({'width':1440,'height':1000})
            from urllib.parse import urlencode
            ranking_params=dict(host='i265',device='CPU',threads='1',mode='score-only',workload='Shared pairs · RGB8 resident · 80 nits · native CPU · sampling differs',impl='competitors',source='',size='512×512',unit='ms/MP',layout='whiskers')
            page.goto(f'http://127.0.0.1:{port}/#ranking?'+urlencode(ranking_params))
            page.wait_for_function('window.dashboardReady==="ranking"')
            assert page.locator('.whisker').count()>0
            assert 'libjxl v0.12' in page.locator('#chart').inner_text()
            with page.expect_download() as rank_download:page.locator('#download').click()
            assert 'libjxl v0.12' in Path(rank_download.value.path()).read_text()
            page.reload();page.wait_for_function('window.dashboardReady==="ranking"')
            assert page.locator('[data-ranking="unit"]').input_value()=='ms/MP'
            if screenshots:page.screenshot(path=str(screenshots/'ranking-competitor.png'),full_page=True)
            # Both supplied implementations must coexist in all eight paired cells.
            speed_rows=json.loads((directory/'data/speed.json').read_text())
            for resolution in ['2048×1080','3840×2160']:
                for threads in ['1','20']:
                    for mode in ['score-only','score+map']:
                        paired=dict(ranking_params,size=resolution,threads=threads,mode=mode,unit='ms')
                        page.goto(f'http://127.0.0.1:{port}/#ranking?'+urlencode(paired))
                        page.wait_for_function('window.dashboardReady==="ranking"')
                        assert page.locator('.rank-bar').count()==2,(resolution,threads,mode)
                        assert page.locator('.whisker').count()==2
                        assert page.locator('#chart .warning').count()==1
                        assert 'zenmetrics' in page.locator('#chart').inner_text()
                        assert 'libjxl v0.12' in page.locator('#chart').inner_text()
                        expected=sorted(r['ms'] for r in speed_rows if r.get('comparison_group')==paired['workload'] and f"{int(r['width'])}×{int(r['height'])}"==resolution and str(r['threads'])==threads and r['mode']==('score' if mode=='score-only' else mode))
                        assert page.locator('.rank-bar').evaluate_all('(bars)=>bars.map(b=>Number(b.dataset.ms))')==expected
                        if screenshots and threads=='1' and mode=='score-only':page.screenshot(path=str(screenshots/('paired-'+resolution+'.png')),full_page=True)
            mixed=dict(host='r7950x',device='CPU',threads='missing',mode='full',workload='os=Linux (WSL2) · backend=CPU · phase=cold · engine=native · content=missing',impl='zenmetrics',source='',size='512×512')
            page.goto(f'http://127.0.0.1:{port}/#ranking?'+urlencode(mixed))
            page.wait_for_function('window.dashboardReady==="ranking"')
            assert page.locator('#chart .warning').count()==1
            if screenshots:page.screenshot(path=str(screenshots/'ranking-mixed-source.png'),full_page=True)
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
            assert page.locator('#chart svg').count()>0
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
        print(json.dumps({'pages':6,'console_errors':errors,'filters':'pass','csv':'pass','URL_restore':'pass','mobile':'pass','theme':'pass'}))
    except Exception:
        print(json.dumps({'browser_errors':errors}),flush=True)
        raise
    finally:server.shutdown();server.server_close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--screenshots',type=Path);parser.add_argument('--port',type=int,default=int(os.environ.get('DASHBOARD_PORT','3387')))
    args=parser.parse_args();check(args.directory,args.screenshots,args.port)
