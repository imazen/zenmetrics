#!/usr/bin/env python3
"""Verify isolated threaded Pages startup and refusal without isolation."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import math
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('site', type=Path)
parser.add_argument('--url', help='Check an already deployed HTTPS site instead')
args = parser.parse_args()


class Handler(SimpleHTTPRequestHandler):
    isolated = True

    def end_headers(self):
        if self.isolated:
            for name, value in (("Cross-Origin-Opener-Policy", "same-origin"),
                                ("Cross-Origin-Embedder-Policy", "require-corp"),
                                ("Cross-Origin-Resource-Policy", "cross-origin"),
                                ("Access-Control-Allow-Origin", "*")):
                self.send_header(name, value)
        super().end_headers()

    def log_message(self, *_):
        pass


server = None
if args.url:
    url = args.url
else:
    server = ThreadingHTTPServer(('127.0.0.1', 3319), partial(Handler, directory=str(args.site.resolve())))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = 'http://127.0.0.1:3319/'
try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        page.goto(url)
        page.wait_for_function("document.getElementById('status').textContent.startsWith('Ready:')", timeout=30000)
        assert page.evaluate('crossOriginIsolated && typeof SharedArrayBuffer !== "undefined"')
        result = page.evaluate('''async () => {
          const worker = new Worker(new URL('./worker.js', location.href), {type:'module'});
          const wait = () => new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('Worker deadline expired')), 15000);
            worker.onmessage = ({data}) => { if(data.type==='progress') return; clearTimeout(timer); resolve(data); };
            worker.onerror = error => reject(new Error(error.message));
          });
          let response = wait(); worker.postMessage({type:'init',threads:2});
          const ready = await response;
          if (ready.type !== 'ready' || ready.threads !== 2) throw new Error(JSON.stringify(ready));
          const width=64,height=64,reference=new Uint8Array(width*height*4),distorted=new Uint8Array(reference.length);
          for(let y=0;y<height;y++) for(let x=0;x<width;x++) for(let c=0;c<4;c++) {
            const i=(y*width+x)*4+c; reference[i]=c===3?255:(x*17+y*13+c*41)%256;
            distorted[i]=c===3?255:Math.max(0,reference[i]-7);
          }
          response=wait();worker.postMessage({id:1,metric:'zensim',reference,distorted,width,height});
          const first=await response;
          if(first.type!=='result') throw new Error(JSON.stringify(first));
          response=wait();worker.postMessage({id:2,metric:'zensim',reference,distorted,width,height});
          const second=await response;
          if(second.type!=='result'||JSON.stringify(first.score)!==JSON.stringify(second.score)||
             JSON.stringify(Array.from(first.map))!==JSON.stringify(Array.from(second.map))) throw new Error('Repeat differs');
          worker.terminate(); return {score:first.score,map:Array.from(first.map)};
        }''')
        assert len(result['map']) == 4096
        assert all(math.isfinite(value) for value in result['map'])
        class Unisolated(Handler):
            isolated = False
        control = ThreadingHTTPServer(('127.0.0.1', 3320), partial(Unisolated, directory=str(args.site.resolve())))
        threading.Thread(target=control.serve_forever, daemon=True).start()
        blocked = browser.new_context()
        refusal = blocked.new_page()
        refusal.goto('http://127.0.0.1:3320/')
        refusal.wait_for_function("document.getElementById('status').className === 'error'", timeout=30000)
        assert refusal.locator('#compare').is_disabled()
        control.shutdown()
        control.server_close()
        print('Pages isolation, real two-worker score/map, repeat and unisolated refusal passed.')
        browser.close()
finally:
    if server:
        server.shutdown()
        server.server_close()
