#!/usr/bin/env python3
"""Exercise real worker initialization, raw maps and browser refusal behavior."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('package', type=Path)
parser.add_argument('report', type=Path)
parser.add_argument('--oracle', required=True, type=Path)
args = parser.parse_args()

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        if not getattr(self, 'unisolated', False):
            self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
            self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()
    def do_GET(self):
        self.unisolated = self.path.startswith('/unisolated')
        if self.unisolated: self.path = '/index.html'
        super().do_GET()
    def log_message(self, *_):
        pass

server = ThreadingHTTPServer(('127.0.0.1', 3318), partial(Handler, directory=str(args.package.resolve())))
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.on('console', lambda message: print('console:', message.type, message.text, flush=True))
        page.on('pageerror', lambda error: print('pageerror:', error, flush=True))
        page.goto('http://127.0.0.1:3318/')
        try:
            page.wait_for_function("document.getElementById('status').textContent.startsWith('Ready:')", timeout=15000)
        except Exception:
            print('status:', page.locator('#status').text_content(), flush=True)
            raise
        result = page.evaluate('''async () => {
          const worker = new Worker('/worker.js', {type:'module'});
          const messages = [];
          const wait = () => new Promise((resolve,reject) => {
            const timer = setTimeout(() => reject(new Error('Worker response deadline expired')), 15000);
            worker.onmessage = event => { if(event.data.type==='progress') return; clearTimeout(timer); messages.push(event.data); resolve(event.data); };
            worker.onerror = event => reject(new Error(event.message));
          });
          let promise = wait(); worker.postMessage({type:'init',threads:2});
          const ready = await promise;
          if (ready.type !== 'ready') throw new Error(JSON.stringify(ready));
          const width=64,height=64;
          const reference=new Uint8Array(width*height*4), distorted=new Uint8Array(reference.length);
          for(let y=0;y<height;y++) for(let x=0;x<width;x++) for(let c=0;c<4;c++) {
            const i=(y*width+x)*4+c;
            reference[i]=c===3?255:(x*17+y*13+c*41)%256;
            distorted[i]=c===3?255:Math.max(0,reference[i]-7);
          }
          promise=wait();worker.postMessage({id:1,metric:'zensim',reference,distorted,width,height});
          const result=await promise;
          if(result.type!=='result') throw new Error(JSON.stringify(result));
          const values=Array.from(result.map);
          promise=wait();worker.postMessage({id:2,metric:'zensim',reference,distorted,width,height});
          const repeat=await promise;
          if (JSON.stringify(values)!==JSON.stringify(Array.from(repeat.map))) throw new Error('Repeated maps differ');
          reference[3]=0;
          promise=wait();worker.postMessage({id:3,metric:'zensim',reference,distorted,width,height});
          const refused=await promise;
          if(refused.type!=='error'||!refused.message.includes('opaque')) throw new Error('Alpha silently accepted');
          worker.terminate();
          return {ready, score:result.score, map:values, repeated:true, alpha_refused:true};
        }''')
        assert len(result['map']) == 64 * 64
        assert all(isinstance(value, (int, float)) for value in result['map'])
        oracle = json.loads(args.oracle.read_text())
        assert result['score']['score'] == oracle['score'], (result['score']['score'], oracle['score'])
        assert result['map'] == oracle['map'], 'Browser raw map differs from native library oracle'
        result['native_oracle_exact'] = True
        page.goto('http://127.0.0.1:3318/unisolated')
        page.wait_for_function("document.getElementById('status').textContent.includes('COOP:')")
        assert page.locator('#compare').is_disabled()
        result['isolation_refused'] = True
        result['browser'] = browser.version
        args.report.write_text(json.dumps(result, indent=2) + '\n')
        browser.close()
finally:
    server.shutdown()
    server.server_close()
print('Browser worker, repeated-map, alpha and isolation controls passed')
