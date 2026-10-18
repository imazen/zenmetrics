#!/usr/bin/env python3
"""Qualify private, user-supplied browser weights without serving the weights."""
import argparse
import ast
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('site', type=Path)
parser.add_argument('models', type=Path)
parser.add_argument('native', type=Path)
parser.add_argument('report', type=Path)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
# Use the existing independent pair formulas, without running check-pages.
tree = ast.parse((repo / 'scripts/checkpoint/check-pages.py').read_text())
snippets = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'NN_PAIRS_JS', 'NN_WORKER_JS'} for t in node.targets):
        snippets[node.targets[0].id] = ast.literal_eval(node.value)
source = json.loads((args.site / 'site-source.json').read_text())
nn_path = '/' + source.get('asset_prefix', '').strip('/') + '/nn.js'
manifest = json.loads((repo / 'web/metrics/nn-models.json').read_text())
native = json.loads(args.native.read_text())['scores']


class Handler(SimpleHTTPRequestHandler):
    def handle(self):
        # Closing a browser context cancels outstanding module prefetches.
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def end_headers(self):
        for name, value in [('Cross-Origin-Opener-Policy', 'same-origin'), ('Cross-Origin-Embedder-Policy', 'require-corp')]:
            self.send_header(name, value)
        super().end_headers()

    def log_message(self, *_):
        pass


server = ThreadingHTTPServer(('127.0.0.1', 3321), partial(Handler, directory=str(args.site.resolve())))
threading.Thread(target=server.serve_forever, daemon=True).start()
implementation_files = [
    'crates/zenmetrics-browser/src/nn_prep.rs', 'web/metrics/nn.js',
    'web/metrics/nn-models.json', 'web/metrics/worker.js', 'web/metrics/app.js',
    'web/metrics/index.html', 'scripts/checkpoint/check-local-weights.py',
    'crates/zenmetrics-browser/examples/nn_oracle.rs',
]
report = {
    'schema': 1, 'provider': 'wasm', 'tolerance_jnd': manifest['tolerance_jnd'],
    'site_source_commit': source['source_commit'],
    'implementation_sha256': {f: hashlib.sha256((repo / f).read_bytes()).hexdigest() for f in implementation_files},
    'qualified_graphs': {name: model['graph'] for name, model in manifest['local_models'].items()},
    'native_engine': 'rten',
    'native_scores': {name: {pair: row['scores'] for pair, row in native[name].items()} for name in manifest['local_models']},
    'models': {},
}
try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        report['chromium_version'] = browser.version
        page = browser.new_page()
        onnx_requests = []
        page.on('request', lambda r: onnx_requests.append(r.url) if '.onnx' in r.url else None)
        page.goto('http://127.0.0.1:3321/')
        # Add an input outside the app's controls; a File enters through the
        # same browser file-selection mechanism and worker protocol as the UI.
        page.evaluate("document.body.insertAdjacentHTML('beforeend','<input id=private-export type=file>')")
        for metric, model in manifest['local_models'].items():
            page.locator('#private-export').set_input_files(args.models / model['graph']['file'])
            other = next(name for name in manifest['local_models'] if name != metric)
            js = 'async ([metric,hash,other]) => {' + snippets['NN_PAIRS_JS'] + snippets['NN_WORKER_JS'] + r'''
              const file=document.getElementById('private-export').files[0];
              const notLoaded=await call({type:'nn',metric,ep:'wasm',...pairs.fixture256});
              const noConsent=await call({type:'nn-load-local',metric,file});
              const wrongFile=await call({type:'nn-load-local',metric,file:new File(['wrong'],file.name),consent:hash});
              const wrongHash=await call({type:'nn-load-local',metric,file:new File([new Uint8Array(file.size)],file.name),consent:hash});
              const loaded=await call({type:'nn-load-local',metric,file,consent:hash});
              if(loaded.type!=='local-loaded')throw new Error(JSON.stringify(loaded));
              const noConsentAfterLoad=await call({type:'nn-load-local',metric,file});
              const otherConsent=await call({type:'nn-load-local',metric:other,file,consent:hash});
              const runs={};
              for(const [name,pair] of Object.entries(pairs)){
                const result=await call({type:'nn',metric,ep:'wasm',...pair});
                if(result.type!=='result')throw new Error(JSON.stringify(result));
                runs[name]=result.score;
              }
              const repeat=await call({type:'nn',metric,ep:'wasm',...pairs.fixture256});
              worker.terminate();
              return {runs,repeat:repeat.score,notLoaded:notLoaded.message,noConsent:noConsent.message,
                wrongFile:wrongFile.message,wrongHash:wrongHash.message,noConsentAfterLoad:noConsentAfterLoad.message,otherConsent:otherConsent.message};
            }'''
            got = page.evaluate(js, [metric, model['graph']['sha256'], other])
            assert 'Select local weights' in got['notLoaded'], got
            assert 'Accept the weight terms' in got['noConsent'], got
            assert 'Accept the weight terms' in got['noConsentAfterLoad'], got
            assert f'Accept the weight terms for {other}' in got['otherConsent'], got
            assert 'Select ' in got['wrongFile'], got
            assert 'sha256' in got['wrongHash'], got
            assert got['repeat']['values'] == got['runs']['fixture256']['values'], got
            errors = {}
            for pair, run in got['runs'].items():
                assert run['ep'] == 'wasm' and run['adapter'] is None
                for v in run['values']:
                    errors[f"{pair}/{v['column']}"] = abs(v['value'] - native[metric][pair]['scores'][v['column']]) / model['jnd']
            assert max(errors.values()) <= manifest['tolerance_jnd'], (metric, errors)
            report['models'][metric] = {'errors_jnd': errors, **got}
            args.report.write_text(json.dumps(report, indent=2) + '\n')
            print(metric, errors, flush=True)
        assert not onnx_requests, onnx_requests
        report['onnx_network_requests'] = onnx_requests
        # Exact UI flow: the deep link selects the conditional metric, no
        # score before local weights/consent, then a scalar from local weights.
        page.goto('http://127.0.0.1:3321/?metric=ckdn')
        page.wait_for_function("document.getElementById('status').textContent.includes('Select local weights')", timeout=300000)
        assert page.locator('#score').inner_text() == '–'
        assert page.locator('#local-weights-row').is_visible()
        page.locator('#local-weights').set_input_files(args.models / manifest['local_models']['ckdn']['graph']['file'])
        page.locator('#local-load').click()
        assert page.locator('#score').inner_text() == '–'
        page.locator('#local-consent').check()
        page.locator('#local-load').click()
        page.wait_for_function("document.getElementById('score-label').textContent==='ckdn' && /^[0-9]/.test(document.getElementById('score').textContent)", timeout=300000)
        assert 'WASM CPU' in page.locator('#score-info').inner_text()
        first_score = page.locator('#score').inner_text()
        page.evaluate("() => {document.getElementById('compare').click();document.getElementById('cancel').click();}")
        page.wait_for_function("document.getElementById('status').textContent.startsWith('Ready:')", timeout=60000)
        assert page.locator('#score').inner_text() == '–'
        page.locator('#compare').click()
        page.wait_for_function("document.getElementById('score-label').textContent==='ckdn' && /^[0-9]/.test(document.getElementById('score').textContent)", timeout=300000)
        assert page.locator('#score').inner_text() == first_score
        page.locator('#metric').select_option('hyperiqa')
        page.wait_for_function("document.getElementById('status').textContent.includes('Select local weights')", timeout=300000)
        assert not page.locator('#local-consent').is_checked()
        assert page.locator('#local-weights').input_value() == ''
        assert page.locator('#score').inner_text() == '–'
        report['ui_local_consent_and_score'] = True
        report['worker_replacement_reloads_current_page_file'] = True
        report['ui_other_model_does_not_inherit_consent'] = True
        # JSPI rejection before importing ORT: CPU routes remain usable.
        no_jspi = browser.new_context()
        no_jspi.add_init_script("Object.defineProperty(WebAssembly,'Suspending',{value:undefined})")
        tab = no_jspi.new_page()
        tab.goto('http://127.0.0.1:3321/')
        error = tab.evaluate("async path => {const nn=await import(path);try{await nn.score({metric:'lpips-alex',ep:'wasm'});}catch(e){return String(e);}}", nn_path)
        assert 'require WebAssembly JSPI' in error, error
        report['no_jspi'] = error
        no_jspi.close()
        browser.close()
finally:
    server.shutdown()
    args.report.write_text(json.dumps(report, indent=2) + '\n')
