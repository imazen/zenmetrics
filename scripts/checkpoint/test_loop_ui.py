#!/usr/bin/env python3
"""Exercise the real trace scoring UI with an explicit deterministic RPC oracle.

This checks UI/routing, not WASM arithmetic. Native scoring is separately tested
by check-pages.py and the IQA harness. No model assets or metric maps are loaded.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
MOCK = """
window.__loopCalls=[];window.__loopPending=[];
window.Worker=class {
  terminate() {}
  postMessage(data) {
    if(data.type==='init')return;
    window.__loopCalls.push(data);
    const send=()=>{
      if(data.type==='compare'&&data.mapMetric!==null)throw new Error('Unexpected map request');
      if(window.__loopFail){window.__loopFail=false;this.onmessage({data:{type:'error',id:data.id,message:'native refusal sentinel'}});return;}
      const score={metric:data.metric,values:[{column:'candidate',value:data.distorted[0]},{column:'reference',value:data.reference[0]}]};
      this.onmessage({data:{type:'score',id:data.id,score}});
      this.onmessage({data:{type:'result',id:data.id,score}});
    };
    if(window.__loopHold)window.__loopPending.push(send);else setTimeout(send,5);
  }
};
"""
HOOK = """
window.__loopReady=()=>{
  S.ready=true;
  S.coverage=new Map([['psnr',{status:'score+map'}],['piqe',{status:'score-only'}],['arniqa',{status:'nn-webgpu'}]]);
  S.metrics=['psnr','piqe','arniqa'];
  $('metric').replaceChildren(...S.metrics.map(m=>new Option(m,m)));
};
"""


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, *_args):
        pass


def fixture():
    frames = [dict(seed=0.6, iter=i, metric='butteraugli', score=1.5-i/10,
                   target=1.0, has_next=False) for i in range(2)]
    head = dict(width=2, height=1, distance=1.0, effort=7, iters=2, bytes=123,
                xsize_blocks=1, ysize_blocks=1, frames=frames)
    header = json.dumps(head).encode()
    data = b'JXLTRACE1\n' + struct.pack('<I', len(header)) + header + bytes([1,2,3,4,5,6])
    for i in range(2):
        data += struct.pack('<4f', 1.0, 1.0, 0.1, 0.2) + bytes([7+i*6,8,9,10,11,12])
    return dict(name='control.ltrace', mimeType='application/octet-stream', buffer=data)



def check_assets():
    scratch = Path.home()/'tmp'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='loop-assets-', dir=scratch) as folder:
        root = Path(folder)
        build = root/'build'
        build.mkdir()
        for name in ('pkg', 'pkg-full', 'codecs', 'ort'):
            (build/name).mkdir()
        # Packaging fixture only; no claim that this is executable WASM.
        (build/'pkg/zenmetrics_browser_bg.wasm').write_bytes(b'packaging control')
        for name in ('index.html','app.js','loop-scores.js','worker.js','nn.js','nn-models.json'):
            (build/name).write_bytes((ROOT/'web/metrics'/name).read_bytes())
        (build/'LICENSE-AGPL3').write_text('packaging fixture')
        source = '0'*40
        subprocess.run([sys.executable,str(ROOT/'scripts/checkpoint/prepare-pages.py'),
                        '--build',str(build),'--source',source,'--out',str(root/'site')], check=True)
        assets = root/'site/assets'/source
        assert (assets/'loop-scores.js').read_bytes() == (ROOT/'web/metrics/loop-scores.js').read_bytes()
        assert "from './loop-scores.js'" in (assets/'app.js').read_text()
        assert not (root/'site/loop-scores.js').exists()
        assert f'assets/{source}/app.js' in (root/'site/index.html').read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chromium', help='Existing Chromium executable; default Playwright installation')
    args = parser.parse_args()
    check_assets()
    server = None
    for port in range(3370, 3400):
        try:
            server = ThreadingHTTPServer(('127.0.0.1', port), partial(Handler, directory=ROOT/'web/metrics'))
            break
        except OSError:
            continue
    if server is None:
        raise RuntimeError('No test port available in 3370..3399')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, executable_path=args.chromium,
                args=['--disable-dev-shm-usage', '--disable-gpu', '--renderer-process-limit=2'])
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.add_init_script(MOCK)
            page.route('**/app.js', lambda route: route.fulfill(content_type='text/javascript',
                body=(ROOT/'web/metrics/app.js').read_text()+'\n'+HOOK))
            page.goto(f'http://127.0.0.1:{server.server_port}/')
            page.wait_for_function('typeof window.__loopReady === "function"')
            page.evaluate('window.__loopReady()')
            page.locator('#loop-auto').uncheck()
            for metric, route_type in [('psnr','compare'), ('piqe','compare'), ('arniqa','nn')]:
                page.locator('#trace').set_input_files(fixture())
                page.wait_for_function('!document.getElementById("loop-score").disabled')
                page.locator('#metric').select_option(metric)
                page.locator('#loop-score').click()
                page.wait_for_function('document.getElementById("loop-score-status").textContent.startsWith("Finished")')
                rows = page.locator('#loop-score-table tbody tr')
                assert rows.count() == 2
                assert rows.nth(0).get_attribute('data-metric') == metric
                assert rows.nth(0).inner_text().endswith('candidate: 7 / reference: 1'), (rows.nth(0).inner_text(), page.evaluate('window.__loopCalls.map(d=>({type:d.type,metric:d.metric,first:d.distorted?.[0]}))'), errors)
                assert rows.nth(1).inner_text().endswith('candidate: 13 / reference: 1')
                assert page.evaluate('window.__loopCalls.at(-1).type') == route_type
                rows.nth(1).click()
                assert page.locator('#loop-frame').input_value() == '1'
            page.evaluate('window.__loopFail=true')
            page.locator('#loop-score').click()
            page.wait_for_function('document.getElementById("loop-score-status").textContent.includes("1 failed")')
            assert page.locator('#loop-score-table tbody tr').nth(0).get_attribute('data-error') == 'native refusal sentinel'
            page.evaluate('window.__loopHold=true')
            page.locator('#loop-score').click()
            page.wait_for_function('window.__loopPending.length===1')
            page.locator('#loop-score-stop').click()
            page.evaluate('window.__loopPending.splice(0).forEach(send=>send())')
            assert page.locator('#loop-score-table tbody tr').count() == 0
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    print('PASS trace scoring UI: FR/NR/learned dispatch, all named values, frame selection, refusal and cancellation')


if __name__ == '__main__':
    main()
