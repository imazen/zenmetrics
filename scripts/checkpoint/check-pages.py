#!/usr/bin/env python3
"""Verify isolated threaded Pages startup and refusal without isolation."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import math
import json
import os
import gzip
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('site', type=Path)
parser.add_argument('--url', help='Check an already deployed HTTPS site instead')
parser.add_argument('--oracle', type=Path, help='Independent native map outputs')
parser.add_argument('--report', type=Path, help='Measured browser results')
parser.add_argument('--webgpu', choices=('swiftshader', 'gpu'), default='swiftshader',
                    help="WebGPU adapter for learned metrics: Chromium's SwiftShader fallback adapter "
                         "(no GPU needed) or the host GPU through Vulkan")
parser.add_argument('--nn-golden', type=Path,
                    default=Path(__file__).resolve().parents[2] / 'benchmarks/browser_nn_2026-10-17/native-rten.json',
                    help='Native rten scores of the learned metrics on the fixed pairs')
parser.add_argument('--nn-models', default='all',
                    help='Comma-separated learned metrics to run (default all); a subset skips their coverage statuses')
parser.add_argument('--write-coverage', type=Path,
                    help='Write the measured coverage file here instead of checking the site copy against it')
args = parser.parse_args()


class Handler(SimpleHTTPRequestHandler):
    isolated = True

    def translate_path(self, path):
        # Cloudflare Pages answers a missing file with the site's index page
        # (200, text/html), not a 404; serve the same, so a page that
        # mistakes that answer for a file fails here before it deploys.
        local = super().translate_path(path)
        if os.path.exists(local):
            return local
        return super().translate_path('/index.html')

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


WORKBENCH = 'https://metrics.imazen.org/'
DASHBOARD = 'https://metric-brief.pages.dev/#models&m='


def unavailable_reason(name, refusal):
    """The measured reason a CLI name does not run here: its class, and the
    module's own refusal message, which the check received."""
    if name == 'cw-ssim-pyiqa':
        return ('No weights are needed, but its implementation in zenmetrics-pyiqa-ports carries '
                'PolyForm Noncommercial 1.0.0 and S-Lab 1.0 terms. That code is excluded from the '
                f'default public WASM module. Browser refusal: {refusal}')
    if name.endswith('-gpu'):
        return (f'GPU implementation (CubeCL on CUDA or native wgpu); this page has no CubeCL runtime. '
                f'Score the CPU metric {name[:-4]} here. Browser refusal: {refusal}')
    if name in ('hdrvdp-aic2026', 'hdrvdp3-aic2026'):
        return ('The explicit SDR display route requires cpu-hdrvdp, which is not compiled into this WASM module. '
                f'Browser refusal: {refusal}')
    if name.startswith('hdrvdp'):
        return ('Scores absolute-luminance HDR input; this page decodes SDR sRGB8 only. '
                f'Browser refusal: {refusal}')
    return f'Browser refusal: {refusal}'


REPO = Path(__file__).resolve().parents[2]
WEBGPU_ARGS = {
    'swiftshader': ['--enable-unsafe-webgpu'],
    'gpu': ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=native', '--use-angle=vulkan'],
}
# The fixed pairs of the learned-metric check, with the formulas of
# crates/zenmetrics-browser/examples/nn_oracle.rs.
NN_PAIRS_JS = r"""
  const pairs={};
  {const s=256,r=new Uint8Array(s*s*4),d=new Uint8Array(r.length);
   for(let i=0;i<r.length;i++){r[i]=i%4===3?255:(i*13+Math.floor(i/(s*4))*17)%256;d[i]=i%4===3?255:Math.max(0,r[i]-7);}
   pairs.fixture256={width:s,height:s,reference:r,distorted:d};}
  {const w=480,h=352,r=new Uint8Array(w*h*4),d=new Uint8Array(r.length);
   for(let y=0;y<h;y++)for(let x=0;x<w;x++)for(let c=0;c<4;c++){const i=(y*w+x)*4+c;if(c===3){r[i]=d[i]=255;continue;}
     const v=(x*7+y*5+c*61+(x*y)%23)%256;r[i]=v;d[i]=Math.floor(y/24)%3===0?Math.floor(v/4)*3+32:(c===2&&Math.floor(x/32)%3===0?v>>1:v);}
   pairs.unified480x352={width:w,height:h,reference:r,distorted:d};}
"""
NN_WORKER_JS = r"""
  const worker=new Worker(new URL('./worker.js',location.href),{type:'module'});
  let nextId=0;
  const call=(message,deadline=600000)=>new Promise((resolve,reject)=>{
    const id=++nextId,timer=setTimeout(()=>reject(new Error('deadline '+message.type)),deadline);
    worker.onmessage=({data})=>{
      if(['control','score','progress','module','prefetch-failed'].includes(data.type))return;
      clearTimeout(timer);resolve(data);
    };
    worker.postMessage({...message,id});
  });
  await call({type:'init',threads:2});
"""
# The CLI's learned metrics (zenmetrics-nn): the CPU modules refuse them.
LEARNED = {'lpips-alex', 'lpips-vgg', 'lpips-squeeze', 'dists', 'stlpips-alex', 'stlpips-vgg', 'deepdc', 'ckdn',
           'afine', 'dreamsim', 'arniqa', 'maniqa', 'liqe', 'musiq', 'hyperiqa', 'paq2piq', 'pieapp',
           'wadiqam-fr-live', 'wadiqam-fr-tid', 'wadiqam-nr-live', 'wadiqam-nr-tid', 'wd', 'ahiq', 'dmm',
           'topiq-fr', 'topiq-fr-pipal', 'wadiqam-pyiqa', 'ahiq-pyiqa', 'dmm-pyiqa', 'pieapp-pyiqa'}


NN_MANIFEST = json.loads((REPO / 'web/metrics/nn-models.json').read_text())
NN_MODELS = set(NN_MANIFEST['models'])
LOCAL_MODELS = set(NN_MANIFEST.get('local_models', {}))


def card_of(name):
    if name in {'topiq-fr', 'topiq-fr-pipal'}:
        return 'topiq'
    if name.endswith('-pyiqa'):
        return name
    for prefix in ('stlpips', 'lpips', 'wadiqam'):
        if name.startswith(prefix + '-'):
            return prefix
    return name


def learned_blocker(name, models):
    """Why a learned metric has no browser route: its weights' licence class
    (card.toml), or the absence of a whole-graph export the browser can run."""
    import tomllib
    crate = 'zenmetrics-pyiqa-ports' if name.endswith('-pyiqa') or name.startswith('topiq-') else 'zenmetrics-nn'
    card = REPO / 'crates' / crate / 'models' / card_of(name) / 'card.toml'
    meta = tomllib.loads(card.read_text())
    weights = meta.get('weights', {})
    licence_class = weights.get('licence_class', 'unknown')
    rel = card.relative_to(REPO)
    if licence_class != 'permissive':
        return (f'Weights licence class "{licence_class}" ([weights] in {rel}): '
                'the browser does not redistribute these weights.')
    assert name not in models
    if name in {'dmm', 'dmm-pyiqa'}:
        return (f'Permissive weights ([weights] in {rel}), but the score needs a 16×16 SVD per patch over '
                f'two VGG16 taps (crates/{crate}/src/dmm.rs, a LAPACK port whose residual floors are '
                'owner-gated); the browser has no graph or route for it and does not restate the solver.')
    return (f'Permissive weights ([weights] in {rel}), but the frozen export set has no whole-graph '
            'export for it: its score needs zenmetrics-nn host code between graph runs, '
            'which the browser does not run.')


def check_learned(playwright, browser, url, result):
    """Every learned metric in nn-models.json on WebGPU against the native
    rten scores, then the refusals and recoveries of the learned route."""
    models = json.loads((REPO / 'web/metrics/nn-models.json').read_text())
    golden = json.loads(args.nn_golden.read_text())['scores']
    tolerance = models['tolerance_jnd']
    context = browser.new_context()
    page = context.new_page()
    page.goto(url)
    run_js = 'async ([metrics, eps]) => {' + NN_PAIRS_JS + NN_WORKER_JS + r"""
      const out={};
      for(const metric of metrics)for(const ep of eps){
        out[metric]??={};out[metric][ep]={};
        for(const [name,pair] of Object.entries(pairs)){
          const got=await call({type:'nn',metric,ep,...pair});
          if(got.type!=='result')throw new Error(metric+' '+ep+' '+name+': '+got.message);
          out[metric][ep][name]={values:got.score.values,ep:got.score.ep,adapter:got.score.adapter,placement:got.score.placement,timings:got.timings,runtime:got.score.runtime};
        }
      }
      worker.terminate();return out;
    }"""
    selected = list(models['models']) if args.nn_models == 'all' else args.nn_models.split(',')
    assert set(selected) <= set(models['models']), selected
    learned = page.evaluate(run_js, [selected, ['webgpu', 'wasm']])

    def deviations(metric, runs):
        """Per pair/column |browser − native| in JND units."""
        jnd = models['models'][metric]['jnd']
        scale = lambda column: jnd[column] if isinstance(jnd, dict) else jnd
        return {f"{pair}/{v['column']}": abs(v['value'] - golden[metric][pair][v['column']]) / scale(v['column'])
                for pair, got in runs.items() for v in got['values']}

    # i265's GPU flips a bit every few minutes (CLAUDE.md Known Bugs): a
    # WebGPU result outside the gate is rerun once and both runs are kept.
    reruns = {}
    for metric in selected:
        if max(deviations(metric, learned[metric]['webgpu']).values()) > tolerance:
            reruns[metric] = page.evaluate(run_js, [[metric], ['webgpu']])[metric]['webgpu']
    statuses, parity = {}, {}
    for metric, by_ep in learned.items():
        for runs in by_ep.values():
            for got in runs.values():
                assert all(math.isfinite(v['value']) for v in got['values']), (metric, got)
        assert all(g['ep'] == 'webgpu' and g['adapter'] for g in by_ep['webgpu'].values()), metric
        assert all(g['ep'] == 'wasm' and g['adapter'] is None for g in by_ep['wasm'].values()), metric
        # ORT's profile of each session's first run: WebGPU sessions ran nodes
        # on WebGPU (ORT may keep some on its CPU kernels); WASM ones none.
        nodes = lambda runs, provider: sum(graph.get(provider, 0) for got in runs.values()
                                           for graph in got['placement'].values() if graph)
        assert nodes(by_ep['webgpu'], 'WebGpuExecutionProvider') > 0, (metric, 'no WebGPU nodes profiled')
        assert nodes(by_ep['wasm'], 'WebGpuExecutionProvider') == 0, (metric, 'WebGPU nodes in a WASM session')
        gpu, cpu = deviations(metric, by_ep['webgpu']), deviations(metric, by_ep['wasm'])
        again = deviations(metric, reruns[metric]) if metric in reruns else None
        parity[metric] = {'webgpu': gpu, 'webgpu_rerun': again, 'wasm': cpu,
                          'placement': {ep: next(iter(runs.values()))['placement'] for ep, runs in by_ep.items()}}
        worst = lambda d: max(d.values())
        if worst(gpu) <= tolerance or (again and worst(again) <= tolerance):
            statuses[metric] = ('nn-webgpu', None)
        elif worst(cpu) <= tolerance:
            pair = max(gpu, key=gpu.get)
            statuses[metric] = ('nn-cpu-wasm',
                f'The WebGPU provider differs from the native rten score by {worst(gpu):.3g} JND at {pair} '
                f'(rerun {worst(again):.3g}), over the card gate of {tolerance:g} JND; the WASM provider is '
                f'within {worst(cpu):.3g} JND, so the page runs this metric on the CPU unless WebGPU is chosen.')
        else:
            statuses[metric] = ('unavailable',
                f'Neither provider meets the card gate of {tolerance:g} JND against the native rten score '
                f'(WebGPU {worst(gpu):.3g}, WASM {worst(cpu):.3g}).')
    adapter = next(iter(learned[next(iter(learned))]['webgpu'].values()))['adapter']
    result['learned'] = {'adapter': adapter, 'statuses': statuses, 'parity': parity, 'runs': learned, 'reruns': reruns}
    refusals = page.evaluate('async () => {' + NN_PAIRS_JS + NN_WORKER_JS + r"""
      // Bad input: transparent pixels, an image below the model's minimum
      // and an unknown provider.
      const alpha=new Uint8Array(64*64*4).fill(100);
      const refusedAlpha=await call({type:'nn',metric:'lpips-squeeze',ep:'webgpu',reference:alpha,distorted:alpha,width:64,height:64});
      const tiny=new Uint8Array(16*16*4);for(let i=0;i<tiny.length;i++)tiny[i]=i%4===3?255:i%251;
      const refusedSmall=await call({type:'nn',metric:'lpips-alex',ep:'webgpu',reference:tiny,distorted:tiny,width:16,height:16});
      const unknownEp=await call({type:'nn',metric:'lpips-squeeze',ep:'webgl',...pairs.fixture256});
      worker.terminate();
      return {refusedAlpha:refusedAlpha.message||null,refusedSmall:refusedSmall.message||null,unknownEp:unknownEp.message||null};
    }""")
    assert refusals['refusedAlpha'] and 'opaque' in refusals['refusedAlpha'], refusals
    assert refusals['refusedSmall'] and 'too small' in refusals['refusedSmall'], refusals
    assert refusals['unknownEp'] and 'execution provider' in refusals['unknownEp'], refusals
    result['learned']['refusals'] = refusals
    context.close()

    # Recovery: a failed or truncated weight download is reported, and the
    # same worker downloads the weights once the network allows it.
    context = browser.new_context()
    page = context.new_page()
    page.goto(url)
    page.route('https://zentrain.r2.imazen.org/**', lambda route: route.abort())
    failed = page.evaluate('async () => {' + NN_PAIRS_JS + NN_WORKER_JS + r"""
      window.nnCall=call;window.nnPairs=pairs;
      return (await call({type:'nn',metric:'lpips-squeeze',ep:'webgpu',...pairs.fixture256})).message||null;
    }""")
    assert failed and 'Weight download failed' in failed, failed
    page.unroute('https://zentrain.r2.imazen.org/**')
    page.route('https://zentrain.r2.imazen.org/**', lambda route: route.fulfill(status=200, body=b'not an onnx graph'))
    truncated = page.evaluate("async () => (await nnCall({type:'nn',metric:'lpips-squeeze',ep:'webgpu',...nnPairs.fixture256})).message||null")
    assert truncated and 'truncated' in truncated, truncated
    page.unroute('https://zentrain.r2.imazen.org/**')
    recovered = page.evaluate("async () => (await nnCall({type:'nn',metric:'lpips-squeeze',ep:'webgpu',...nnPairs.fixture256})).score?.values||null")
    assert recovered and recovered[0]['value'] == learned['lpips-squeeze']['webgpu']['fixture256']['values'][0]['value'], recovered
    context.close()
    result['learned']['weight_fetch_failure_recovery'] = {'aborted': failed, 'truncated': truncated}

    # No WebGPU adapter: Chromium without the WebGPU flags has none here. The
    # WebGPU request fails and names it; the WASM provider is a separate,
    # explicit choice that still matches the native scores.
    plain = playwright.chromium.launch()
    page = plain.new_page()
    page.goto(url)
    no_gpu = page.evaluate('async () => {' + NN_PAIRS_JS + NN_WORKER_JS + r"""
      const refused=await call({type:'nn',metric:'lpips-squeeze',ep:'webgpu',...pairs.fixture256});
      const cpu=await call({type:'nn',metric:'lpips-squeeze',ep:'wasm',...pairs.fixture256});
      return {refused:refused.message||null,cpu:cpu.score?{ep:cpu.score.ep,adapter:cpu.score.adapter,values:cpu.score.values}:cpu.message};
    }""")
    plain.close()
    assert no_gpu['refused'] and 'WebGPU unavailable' in no_gpu['refused'], no_gpu
    cpu = no_gpu['cpu']
    assert cpu['ep'] == 'wasm' and cpu['adapter'] is None, cpu
    want = golden['lpips-squeeze']['fixture256']['lpips_squeeze']
    assert abs(cpu['values'][0]['value'] - want) <= tolerance * models['models']['lpips-squeeze']['jnd'], (cpu, want)
    result['learned']['no_webgpu'] = no_gpu



def check_learned_ui(browser, url, result):
    """The page: a learned-metric link names its provider; cancelling an
    in-flight inference emits no score and the fresh worker scores again."""
    ui = browser.new_page()
    ui.goto(url + '?metric=lpips-squeeze')
    ui.wait_for_function("document.getElementById('score-label').textContent==='lpips-squeeze' && document.getElementById('score-info').textContent.includes('WebGPU')", timeout=300000)
    first_score = ui.locator('#score').inner_text()
    ui.evaluate("() => {document.getElementById('compare').click();const c=document.getElementById('cancel');if(c.disabled)throw new Error('no active job');c.click();}")
    ui.wait_for_function("document.getElementById('status').textContent.startsWith('Ready:')", timeout=60000)
    assert ui.locator('#score').inner_text() == '–'
    ui.locator('#compare').click()
    ui.wait_for_function("/^[0-9]/.test(document.getElementById('score').textContent)", timeout=300000)
    assert ui.locator('#score').inner_text() == first_score, (ui.locator('#score').inner_text(), first_score)
    ui.locator('#ep').select_option('wasm')
    ui.wait_for_function("document.getElementById('score-info').textContent.includes('WASM CPU') && new URL(location.href).searchParams.get('ep')==='wasm'", timeout=300000)
    ui.close()
    result['learned']['ui_cancel_recovery_and_provider'] = True


def measured_coverage(result):
    core = set(result['modules']['core']['metrics'])
    entries = []
    for name in result['cliNames']:
        entry = {'name': name, 'status': 'unavailable', 'module': None, 'maps': [], 'reason': None}
        if name in result['metrics']:
            maps = result['mapNames'].get(name, [])
            entry.update(status='score+map' if maps else 'score-only', maps=maps,
                         module='core' if name in core else 'full')
        elif name in result['learned']['statuses'] and result['learned']['statuses'][name][0] != 'unavailable':
            status, reason = result['learned']['statuses'][name]
            entry.update(status=status, module='nn', reason=reason)
        elif name in result['learned']['statuses']:
            entry['reason'] = result['learned']['statuses'][name][1]
        elif name in LOCAL_MODELS:
            # Private exports are explicitly supplied to check-local-weights.py.
            # The public deployment does not download or host those weights.
            entry.update(status='nn-local', module='nn', reason='Select the qualified local ONNX export and accept its weight terms. CPU WebAssembly only; weights stay in worker memory.')
        elif name in LEARNED and args.nn_models != 'all' and name in NN_MODELS:
            entry['not_run'] = True
        elif name in LEARNED:
            entry['reason'] = learned_blocker(name, result['learned']['statuses'])
        else:
            entry['reason'] = unavailable_reason(name, result['refused'][name])
        entries.append(entry)
    return {'schema': 1, 'workbench': WORKBENCH, 'contract': 'docs/BROWSER_WORKBENCH_LINKS.md',
            'workbench_url': WORKBENCH + '?metric={name}', 'dashboard_url': DASHBOARD + '{name}',
            'metrics': entries}


def coverage_text(coverage):
    """One line per metric, so diffs show which statuses changed."""
    head = ',\n'.join(f'  {json.dumps(k)}: {json.dumps(v)}' for k, v in coverage.items() if k != 'metrics')
    rows = ',\n'.join('    ' + json.dumps(e, separators=(', ', ': ')) for e in coverage['metrics'])
    return '{\n' + head + ',\n  "metrics": [\n' + rows + '\n  ]\n}\n'


def check_coverage(committed, measured):
    """The shipped coverage file must describe what the browser just did."""
    for key in ('schema', 'workbench', 'workbench_url', 'dashboard_url'):
        assert committed.get(key) == measured[key], ('coverage.json', key)
    got = {e['name']: e for e in committed['metrics']}
    want = {e['name']: e for e in measured['metrics']}
    assert list(got) == list(want), ('coverage.json names differ from the CLI catalog',
                                     sorted(set(got) ^ set(want)))
    for name, w in want.items():
        g = got[name]
        if w.get('not_run'):
            # A learned metric outside --nn-models: its route was not executed.
            assert g['status'] in ('nn-webgpu', 'nn-cpu-wasm') and g['module'] == 'nn', (name, g)
            continue
        for key in ('status', 'module', 'maps'):
            assert g.get(key) == w[key], (name, key, g.get(key), w[key])
        # Unavailable entries and CPU-only learned entries carry their reason.
        assert bool(g.get('reason')) == (g['status'] in ('unavailable', 'nn-cpu-wasm', 'nn-local')), (name, 'reason')


server = None
if args.url:
    url = args.url
else:
    server = ThreadingHTTPServer(('127.0.0.1', 3319), partial(Handler, directory=str(args.site.resolve())))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = 'http://127.0.0.1:3319/'
try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=WEBGPU_ARGS[args.webgpu])
        context = browser.new_context()
        source_file=args.site/'site-source.json'
        asset_prefix=json.loads(source_file.read_text()).get('asset_prefix','') if source_file.is_file() else ''
        if asset_prefix:
            # Historical fixed package URLs must never be used by a new viewer,
            # even if their cached glue allocates an obsolete memory minimum.
            from urllib.parse import urlparse
            context.route('**/pkg/**', lambda route: route.abort() if urlparse(route.request.url).path.startswith('/pkg/') else route.continue_())
        page = context.new_page()
        page.goto(url)
        # The demo scene starts encoding as soon as the worker is ready, so
        # the "Ready:" status is transient; the startup line stays.
        page.wait_for_function("document.getElementById('startup-timing').textContent.includes('Rayon pool')", timeout=30000)
        assert page.evaluate('crossOriginIsolated && typeof SharedArrayBuffer !== "undefined"')
        assert 'Shields off' in page.locator('details').filter(has_text='Brave:').text_content()
        page.wait_for_function("!document.getElementById('capabilities').textContent.startsWith('Checking')",timeout=10000)
        assert 'WebGPU' in page.locator('#capabilities').inner_text()
        if asset_prefix:
            assert asset_prefix in page.locator('script[type=module]').get_attribute('src')
        # The demo scene loads by default: encoded, scored and mapped, and its
        # loop encodes live in the worker, steered by the default metric.
        page.wait_for_function("document.getElementById('status').textContent.includes('768×512 · native-cpu-wasm')",timeout=60000)
        assert page.locator('#name-reference').inner_text() == 'Demo scene · 768×512'
        assert page.locator('#name-distorted').inner_text().startswith('zenjpeg · quality 40')
        # A finished loop encode plays on its own.
        loop_done = lambda steering: page.wait_for_function(
            "document.getElementById('loop-info').textContent.startsWith('Frame ') && document.getElementById('loop-info').textContent.includes(%s) && document.getElementById('loop-play').textContent.includes('Pause')" % json.dumps(steering), timeout=300000)
        loop_done('steered by zensim · ')
        default_loop = page.locator('#loop-info').inner_text()
        assert 'live encode' in default_loop and 'external' in default_loop, default_loop
        assert page.locator('#loop-save').is_enabled() and page.locator('#ls-encode').inner_text().startswith('768×512 · d1 e7 · ')
        page.locator('#loop-play').click()
        assert page.locator('#loop-play').inner_text() == '▶ Play'
        # From iteration 2 the loop only refines blocks above the target; on
        # the demo none are, so one frame reports the field converged. The
        # readout keeps its size, so stepping never moves the image or chart.
        frames = int(default_loop.split(' of ')[1].split(' ')[0])
        seek = lambda k: page.evaluate("k => {const s=document.getElementById('loop-frame');s.value=k;s.dispatchEvent(new Event('input'));}", k)
        boxes, converged = set(), False
        for k in range(frames):
            seek(k)
            boxes.add((page.locator('#loop-view').bounding_box()['y'], page.locator('#loop-chart').bounding_box()['y']))
            converged = converged or page.locator('#ls-next').inner_text().startswith('unchanged: converged')
        assert converged and len(boxes) == 1, (converged, boxes)
        # The top bar's Map menu and the loop's Steer menu both switch metric,
        # and each choice re-encodes the loop steered by it.
        page.locator('#map-name').select_option('msssim|diffmap')
        page.wait_for_function("document.getElementById('map-info').textContent.startsWith('msssim · diffmap ·')",timeout=60000)
        assert page.locator('#metric').input_value() == 'msssim'
        assert page.locator('#loop-steer').input_value() == 'msssim|diffmap'
        loop_done('steered by msssim · diffmap · ')
        page.locator('#loop-steer').select_option('gmsd|gms')
        page.wait_for_function("document.getElementById('map-info').textContent.startsWith('gmsd · gms ·')",timeout=60000)
        assert page.locator('#metric').input_value() == 'gmsd' and page.locator('#map-name').input_value() == 'gmsd|gms'
        loop_done('steered by gmsd · gms · ')
        page.locator('#loop-steer').select_option('builtin')
        loop_done('steered by butteraugli (built-in)')
        assert page.locator('#metric').input_value() == 'gmsd'
        # The distance slider in the top bar re-encodes at its distance.
        page.evaluate("() => {const s=document.getElementById('loop-distance-range');s.value=543;s.dispatchEvent(new Event('change'));}")
        assert page.locator('#loop-distance').input_value() == '2'
        page.wait_for_function("document.getElementById('ls-encode').textContent.startsWith('768×512 · d2 e7 · ')", timeout=300000)
        ui_linked_menus = True
        page.locator('#metric').select_option('zensim')
        # Negative control for the reported old-glue/new-module memory mismatch.
        import re
        glue=(args.site/asset_prefix/'pkg/zenmetrics_browser.js').read_text()
        initial=int(re.search(r'initial:\s*(\d+)',glue).group(1))
        mismatch=page.evaluate('''async ({prefix,pages}) => {
          const module=await import(new URL(`${prefix?prefix+'/':''}pkg/zenmetrics_browser.js?memory-negative-control`,location.href));
          try {
            await module.default({memory:new WebAssembly.Memory({initial:pages,maximum:32768,shared:true})});
            throw new Error('Undersized shared memory was accepted');
          } catch(error) {
            if(!(error instanceof WebAssembly.LinkError)||!String(error).includes('memory import'))throw error;
            return String(error);
          }
        }''',{'prefix':asset_prefix,'pages':initial-1})
        result = page.evaluate('''async () => {
          const worker = new Worker(new URL('./worker.js', location.href), {type:'module'});
          let publishedScore, sharedControls=0, earlyScores=0;
          const wait = () => new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('Worker deadline expired')), 15000);
            worker.onmessage = ({data}) => {
              if(data.type==='control'){if(!(data.buffer instanceof SharedArrayBuffer)||data.offset%4)throw new Error('Shared atomic control required');sharedControls++;return;}
              if(data.type==='score'){publishedScore=data.score;earlyScores++;return;}
              if(data.type==='progress'||data.type==='module')return;
              if(data.type==='result'&&JSON.stringify(data.score)!==JSON.stringify(publishedScore))throw new Error('Final score differs from published scalar');
              clearTimeout(timer);resolve(data);
            };
            worker.onerror = error => reject(new Error(error.message));
          });
          let response = wait(); worker.postMessage({type:'init',threads:2});
          const ready = await response;
          // The core module answers at start; `catalog` imports the full one.
          if(ready.module!=='core'||JSON.stringify(ready.metrics)!=='["zensim"]'||JSON.stringify(ready.maps)!=='["zensim"]')throw new Error('Core module catalog '+JSON.stringify([ready.metrics,ready.maps]));
          response=wait();worker.postMessage({type:'catalog'});
          const catalog=await response;
          if(catalog.type!=='catalog'||!catalog.modules.full)throw new Error('Full module did not load: '+JSON.stringify(catalog));
          const allMaps=[...new Set([...ready.maps,...catalog.modules.full.maps])],allScored=[...new Set([...ready.metrics,...catalog.modules.full.metrics])];
          if (ready.type !== 'ready' || ready.threads !== 2 || !Number.isFinite(ready.startup?.wasmMilliseconds) || !Number.isFinite(ready.startup?.poolMilliseconds)) throw new Error(JSON.stringify(ready));
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
          for(const got of [first,second]) {
            for(const key of ['referenceMilliseconds','scoreMilliseconds','mapMilliseconds','milliseconds'])if(!Number.isFinite(got[key])||got[key]<0)throw new Error('Invalid timing '+key);
            if(got.referenceMilliseconds+got.scoreMilliseconds+got.mapMilliseconds>got.milliseconds+.01)throw new Error('Compute phases exceed worker total');
          }
          if(!first.referencePrepared||second.referencePrepared)throw new Error('Warm reference timing is not identified');
          if(second.type!=='result'||JSON.stringify(first.score)!==JSON.stringify(second.score)||
             JSON.stringify(Array.from(first.map))!==JSON.stringify(Array.from(second.map))) throw new Error('Repeat differs');
          const spatial={};
          for(const mapMetric of ['butteraugli','dssim','cvvdp','psnr','gmsd','fsim','fsimc','haarpsi','vsi','msssim','ssim2']) {
            response=wait();worker.postMessage({type:'compare',metric:'zensim',mapMetric,reference,distorted,width,height,display:'standard_4k'});
            const got=await response;if(got.type!=='result')throw new Error(JSON.stringify(got));
            spatial[mapMetric]={width:got.mapWidth,height:got.mapHeight,similarity:got.similarity,geometry:got.geometry,values:Array.from(got.map)};
          }
          // The unified-interface maps need a larger pair (NLPD >= 96, vifvec 72, IW-SSIM 176, VMAF one side >= 216).
          const lw=480,lh=352,unified=new Uint8Array(lw*lh*4),striped=new Uint8Array(unified.length);
          for(let y=0;y<lh;y++)for(let x=0;x<lw;x++)for(let c=0;c<4;c++){
            const i=(y*lw+x)*4+c;if(c===3){unified[i]=striped[i]=255;continue;}
            const v=(x*7+y*5+c*61+(x*y)%23)%256;unified[i]=v;
            striped[i]=Math.floor(y/24)%3===0?Math.floor(v/4)*3+32:(c===2&&Math.floor(x/32)%3===0?v>>1:v);
          }
          for(const mapMetric of ['nlpd','psnrhvs','mdctpsnr','mad','iwssim','vif','vifvec','vmaf-v1']) {
            response=wait();worker.postMessage({type:'compare',metric:'psnr',mapMetric,reference:unified,distorted:striped,width:lw,height:lh});
            const got=await response;if(got.type!=='result')throw new Error(mapMetric+': '+JSON.stringify(got));
            spatial[mapMetric]={width:got.mapWidth,height:got.mapHeight,similarity:got.similarity,geometry:got.geometry,values:Array.from(got.map)};
          }
          for(const mapMetric of ['nlpd','vifvec','iwssim','vmaf-v1']) {
            response=wait();worker.postMessage({type:'compare',metric:'psnr',mapMetric,reference,distorted,width,height});
            const got=await response;if(got.type!=='error')throw new Error('Undersized '+mapMetric+' map was not refused');
          }
          const profiles = {};
          for(const zensimProfile of ['b','d','a','preview-v0.1','preview-v0.2','b']) {
            response=wait();worker.postMessage({type:'compare',metric:'zensim',mapMetric:'zensim',zensimProfile,reference,distorted,width,height});
            const got=await response;
            if(got.type!=='result'||got.score.profile!==zensimProfile||got.mapProfile!==zensimProfile)throw new Error('Selected model was not used for score/map');
            profiles[zensimProfile]={score:got.score.score,map:Array.from(got.map)};
          }
          response=wait();worker.postMessage({type:'compare',metric:'zensim',mapMetric:'psnr',zensimProfile:'c',reference,distorted,width,height});
          const candidate=await response;if(candidate.type!=='result'||candidate.score.profile!=='c')throw new Error(JSON.stringify(candidate));
          profiles.c={score:candidate.score.score,map:null};
          response=wait();worker.postMessage({type:'compare',metric:'zensim',mapMetric:'zensim',zensimProfile:'c',reference,distorted,width,height});
          const noMap=await response;if(noMap.type!=='error'||!noMap.message.includes('Model C spatial maps'))throw new Error('C map was silently substituted');
          response=wait();worker.postMessage({type:'compare',metric:'zensim',mapMetric:'zensim',zensimProfile:'b-hdr',reference,distorted,width,height});
          if((await response).type!=='error')throw new Error('HDR model accepted through SDR ingress');
          for(const name of ['fsim','fsimc','haarpsi','vsi','msssim','ssim2','nlpd','psnrhvs','mdctpsnr','mad','iwssim','vif','vifvec','vmaf-v1'])if(!allMaps.includes(name))throw new Error('Missing map catalog '+name);
          const oddWidth=257,oddHeight=253,odd=new Uint8Array(oddWidth*oddHeight*4);
          for(let i=0;i<odd.length;i++)odd[i]=i%4===3?255:(i*17+Math.floor(i/(oddWidth*4))*13)%256;
          const oddMaps={};
          for(const mapMetric of ['fsim','haarpsi','vsi','gmsd']) {
            response=wait();worker.postMessage({type:'compare',metric:'psnr',mapMetric,reference:odd,distorted:odd,width:oddWidth,height:oddHeight});
            const got=await response;if(got.type!=='result')throw new Error(JSON.stringify(got));
            if(!got.similarity||!got.map.every(Number.isFinite))throw new Error('Invalid identity map '+mapMetric);
            if(mapMetric==='vsi'){
              if(Math.min(...got.map)!==0||Math.max(...got.map)!==1)throw new Error('VSI identity map lost native saliency weighting');
            } else if(!got.map.every(v=>Math.abs(v-1)<.00001))throw new Error('Identity similarity map differs '+mapMetric);
            if(got.geometry.inputWidth!==oddWidth||got.geometry.inputHeight!==oddHeight||got.geometry.cellsX!==got.mapWidth||got.geometry.cellsY!==got.mapHeight)throw new Error('Invalid native geometry');
            oddMaps[mapMetric]=got.geometry;
          }
          // Identical inputs: these components vanish exactly; VIF keeps the reference's 1e-10 regularization and IW-SSIM cs its f32 rounding.
          for(const mapMetric of ['ssim2','msssim','iwssim','nlpd','psnrhvs','mdctpsnr','mad','vif','vifvec']) {
            response=wait();worker.postMessage({type:'compare',metric:'psnr',mapMetric,reference:odd,distorted:odd,width:oddWidth,height:oddHeight});
            const got=await response;if(got.type!=='result')throw new Error(JSON.stringify(got));
            const regularized=mapMetric.startsWith('vif');
            if(got.similarity||!got.map.every(v=>regularized?v>0&&v<1e-9:mapMetric==='iwssim'?v>=0&&v<=1e-6:v===0))throw new Error('Identity component map differs '+mapMetric);
            if(got.geometry.inputWidth!==oddWidth||got.geometry.inputHeight!==oddHeight||got.geometry.cellsX!==got.mapWidth||got.geometry.cellsY!==got.mapHeight)throw new Error('Invalid native geometry '+mapMetric);
            oddMaps[mapMetric]=got.geometry;
          }
          response=wait();worker.postMessage({type:'compare',metric:'psnr',mapMetric:'vmaf-v1',reference:odd,distorted:odd,width:oddWidth,height:oddHeight});
          if((await response).type!=='error')throw new Error('Odd-sized 4:2:0 VMAF map was not refused');
          const mixed=new Uint8Array(128*128*4);
          for(let y=0;y<128;y++)for(let x=0;x<128;x++)for(let c=0;c<4;c++)mixed[(y*128+x)*4+c]=c===3?255:(x<64&&y<64?50:(x*17+y*13+c*41)%256);
          response=wait();worker.postMessage({type:'compare',metric:'zensim',mapMetric:'haarpsi',reference:mixed,distorted:mixed,width:128,height:128});
          const undefinedMap=await response;
          if(undefinedMap.type!=='result'||!undefinedMap.map.some(Number.isNaN)||!undefinedMap.map.some(Number.isFinite))throw new Error('HaarPSI native undefined regions were lost');
          response=wait();worker.postMessage({type:'generate',reference,width,height,quality:40});
          const jpeg=await response;if(jpeg.type!=='decoded'||!jpeg.encoded||jpeg.encoded[0]!==255||jpeg.encoded[1]!==216)throw new Error('JPEG generation failed');
          // The independent browser decoder validates the Zen encoder's bytes.
          const chrome=await createImageBitmap(new Blob([jpeg.encoded],{type:'image/jpeg'}));
          if(chrome.width!==width||chrome.height!==height)throw new Error('Independent JPEG dimensions differ');chrome.close();
          response=wait();worker.postMessage({type:'decode',bytes:new Uint8Array([255,216,1]),filename:'broken.jpg'});
          if((await response).type!=='error')throw new Error('Broken JPEG accepted');
          response=wait();worker.postMessage({type:'decode',bytes:jpeg.encoded,filename:'uploaded.jpg'});
          const uploaded=await response;
          if(uploaded.type!=='decoded'||JSON.stringify(Array.from(uploaded.rgba))!==JSON.stringify(Array.from(jpeg.rgba)))throw new Error('Upload decoding or recovery differs');
          const pngCanvas=new OffscreenCanvas(8,8),pngContext=pngCanvas.getContext('2d');
          pngContext.fillStyle='rgba(100,80,60,0.5)';pngContext.fillRect(0,0,8,8);
          const png=new Uint8Array(await (await pngCanvas.convertToBlob({type:'image/png'})).arrayBuffer());
          response=wait();worker.postMessage({type:'decode',bytes:png,filename:'transparent.png'});
          const transparent=await response;
          if(transparent.type!=='decoded'||transparent.rgba[3]!==128)throw new Error('PNG alpha lost');
          response=wait();worker.postMessage({type:'compare',metric:'zensim',reference:transparent.rgba,distorted:transparent.rgba,width:8,height:8});
          const alpha=await response;if(alpha.type!=='error'||!alpha.message.includes('opaque'))throw new Error('PNG alpha silently accepted');
          const input={width:256,height:256},r=new Uint8Array(256*256*4),d=new Uint8Array(r.length);
          for(let i=0;i<r.length;i++){r[i]=i%4===3?255:(i*13+Math.floor(i/1024)*17)%256;d[i]=i%4===3?255:Math.max(0,r[i]-7);}
          const inputBig={width:1024,height:1024},rBig=new Uint8Array(1024*1024*4),dBig=new Uint8Array(rBig.length);
          for(let i=0;i<rBig.length;i++){rBig[i]=i%4===3?255:(i*13+Math.floor(i/4096)*17)%256;dBig[i]=i%4===3?255:Math.max(0,rBig[i]-7);}
          // Every metric the modules score: its score, and its own native
          // map where it publishes one. Every other CLI name must be refused.
          const metrics={},mapNames={},scalarFixture={},refused={};
          for(const metric of allScored) {
            const mapMetric=allMaps.includes(metric)?metric:null;
            response=wait();worker.postMessage({type:'compare',metric,mapMetric,reference:r,distorted:d,...input,display:'standard_4k'});
            let got=await response;scalarFixture[metric]=256;
            // Some metrics refuse 256x256 by design (e.g. five-scale MS-SSIM
            // variants with a fixed pre-downscale need >= 704 px); score those once at 1024x1024.
            if(got.type==='error'&&/requires at least|too small/i.test(got.message||'')){
              response=wait();worker.postMessage({type:'compare',metric,mapMetric,reference:rBig,distorted:dBig,...inputBig,display:'standard_4k'});
              got=await response;scalarFixture[metric]=1024;
            }
            if(metric==='ms-ssim-pyiqa-f32-k6'){
              if(got.type!=='error'||!got.message.includes('scale 5 is 10x10'))throw new Error('k6 must refuse the 1024 fixture');
              const side=1152,r6=new Uint8Array(side*side*4),d6=new Uint8Array(r6.length);
              for(let i=0;i<r6.length;i++){r6[i]=i%4===3?255:(i*13+Math.floor(i/(side*4))*17)%256;d6[i]=i%4===3?255:Math.max(0,r6[i]-7);}
              response=wait();worker.postMessage({type:'compare',metric,mapMetric,reference:r6,distorted:d6,width:side,height:side,display:'standard_4k'});
              got=await response;scalarFixture[metric]=side;
            }
            if(got.type!=='result')throw new Error(metric+': '+JSON.stringify(got));metrics[metric]=got.score;
            if(got.score.module!==(ready.metrics.includes(metric)?'core':'full'))throw new Error(metric+' scored in module '+got.score.module);
            if(mapMetric){
              if(!got.map||got.map.length!==got.mapWidth*got.mapHeight||got.mapMetric!==metric)throw new Error(metric+' map missing');
              mapNames[metric]=got.mapNames;
            } else if(got.map!==null)throw new Error(metric+' returned a map it does not publish');
          }
          for(const metric of catalog.all.filter(m=>!allScored.includes(m))) {
            response=wait();worker.postMessage({type:'compare',metric,mapMetric:null,reference:r,distorted:d,...input,display:'standard_4k'});
            const got=await response;
            if(got.type!=='error')throw new Error(metric+' is outside the catalog but returned '+got.type);
            refused[metric]=got.message;
          }
          const hd={width:1920,height:1080},hr=new Uint8Array(hd.width*hd.height*4),hdist=new Uint8Array(hr.length);
          for(let i=0;i<hr.length;i++){hr[i]=i%4===3?255:(i*17+Math.floor(i/(hd.width*4))*13)%256;hdist[i]=i%4===3?255:Math.max(0,hr[i]-7);}
          response=wait();worker.postMessage({type:'compare',metric:'ssim2',mapMetric:'psnr',reference:hr,distorted:hdist,...hd});
          const timed=await response;if(timed.type!=='result')throw new Error(JSON.stringify(timed));
          // The previous viewer trapped in native thread creation at this size.
          response=wait();worker.postMessage({type:'compare',metric:'zensim',mapMetric:'zensim',reference:hr,distorted:hdist,...hd});
          const large=await response;if(large.type!=='result'||large.map.length!==hd.width*hd.height||!Number.isFinite(large.score.score)||!large.map.every(Number.isFinite))throw new Error('Full-HD zensim score/map failed');
          // Main-thread atomic cancellation remains available during synchronous WASM.
          const cancelImage=new Uint8Array(2048*2048*4);for(let i=0;i<cancelImage.length;i++)cancelImage[i]=i%4===3?255:i%251;
          const cancelled=await new Promise((resolve,reject)=>{
            const deadline=setTimeout(()=>reject(new Error('Native cancellation deadline expired')),10000);
            worker.onmessage=({data})=>{
              if(data.type==='control'){const state=new Int32Array(data.buffer,data.offset,4);Atomics.store(state,0,1);}
              else if(data.type==='error'){clearTimeout(deadline);resolve(data);}
              else if(data.type==='result'||data.type==='score'){clearTimeout(deadline);reject(new Error('Cancelled operation emitted a score'));}
            };
            worker.postMessage({id:99,type:'compare',metric:'zensim',mapMetric:'zensim',reference:cancelImage,distorted:cancelImage,width:2048,height:2048});
          });
          if(!cancelled.message.toLowerCase().includes('cancel'))throw new Error(JSON.stringify(cancelled));
          response=wait();worker.postMessage({type:'compare',metric:'zensim',mapMetric:'zensim',reference,distorted,width,height});
          const afterCancel=await response;if(afterCancel.type!=='result'||afterCancel.score.score!==first.score.score)throw new Error('Cancellation damaged next score');
          if(sharedControls<38||earlyScores<38)throw new Error("Missing live controls or early scores");
          worker.terminate();return {oddMaps,mapNames,scalarFixture,refused,cliNames:catalog.all,modules:catalog.modules,startup:ready.startup,timings:{cold:{reference:first.referenceMilliseconds,score:first.scoreMilliseconds,map:first.mapMilliseconds,total:first.milliseconds},warm:{reference:second.referenceMilliseconds,score:second.scoreMilliseconds,map:second.mapMilliseconds,total:second.milliseconds}},profiles,shared_control_messages:sharedControls,early_score_messages:earlyScores,zensim_1080p_ms:large.milliseconds,atomic_cancellation:true,score:first.score,map:Array.from(first.map),spatial,metrics,ssim2_1080p_worker_ms:timed.milliseconds,ssim2_1080p_score_ms:timed.scoreMilliseconds,jpeg_upload_recovery:true,jpeg_bytes:Array.from(jpeg.encoded)};
        }''')
        assert len(result['map']) == 4096
        assert all(math.isfinite(value) for value in result['map'])
        # Map selection, the encoder modules and the live loop, through the
        # worker protocol.
        result['ui_linked_menus'] = ui_linked_menus
        result['default_loop'] = default_loop
        result['worker_maps_codecs_loop'] = page.evaluate('''async () => {
          const worker = new Worker(new URL('./worker.js', location.href), {type:'module'});
          let progress=0,frames=0;
          const wait = () => new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('Worker deadline expired')), 120000);
            worker.onmessage = ({data}) => {
              if(['control','score','progress','module'].includes(data.type))return;
              if(data.type==='loop-progress'){if(!data.message)progress++;return;}
              if(data.type==='loop-frame'){if(data.recon.length!==data.width*data.height*4||data.map.length!==data.width*data.height)throw new Error('loop frame size');frames++;return;}
              clearTimeout(timer);data.type==='error'?reject(new Error(data.message)):resolve(data);
            };
          });
          let response=wait();worker.postMessage({type:'init',threads:2});
          const ready=await response;
          if(!Array.isArray(ready.loopDefaults))throw new Error('No loop defaults');
          const width=240,height=200,reference=new Uint8Array(width*height*4),distorted=new Uint8Array(reference.length);
          for(let y=0;y<height;y++) for(let x=0;x<width;x++) for(let c=0;c<4;c++) {
            const i=(y*width+x)*4+c; reference[i]=c===3?255:((x>>3)*37+(y>>2)*11+c*53+((x*y)&15))%256;
            distorted[i]=c===3?255:Math.max(0,reference[i]-((x+y)%7));
          }
          const out={};
          for(const metric of ['msssim','dssim']){
            response=wait();worker.postMessage({type:'compare',metric:'psnr',mapMetric:metric,mapName:'',reference,distorted,width,height});
            const first=await response;
            // The fused diffmap is the default; the list keeps the metric's order.
            if(first.mapName!=='diffmap'||first.mapNames.length<2||!first.mapNames.includes('diffmap'))throw new Error(metric+' maps '+JSON.stringify(first.mapNames));
            const name=first.mapNames.find(n=>n!=='diffmap');
            response=wait();worker.postMessage({type:'map',mapMetric:metric,mapName:name});
            const cached=await response;
            response=wait();worker.postMessage({type:'compare',metric:'psnr',mapMetric:metric,mapName:name,reference,distorted,width,height});
            const fresh=await response;
            if(cached.mapName!==name||fresh.mapName!==name||cached.map.length!==cached.mapWidth*cached.mapHeight)throw new Error(metric+' map '+name);
            if(JSON.stringify(Array.from(cached.map))!==JSON.stringify(Array.from(fresh.map)))throw new Error(metric+' cached map differs');
            out[metric]={names:first.mapNames,second:name,polarity:cached.mapPolarity};
          }
          response=wait();worker.postMessage({type:'compare',metric:'psnr-y',mapMetric:null,reference,distorted,width,height});
          const scoreOnly=await response;
          if(scoreOnly.map!==null)throw new Error('A score-only metric returned a map');
          const magic={webp:[82,73,70,70],avif:[0,0,0],jxl:[255,10]};
          out.codecs={};
          for(const codec of ['jpeg','webp','avif','jxl']){
            response=wait();worker.postMessage({type:'generate',codec,reference,width,height,quality:50});
            const got=await response;
            if(got.rgba.length!==width*height*4||!(got.bytes>0))throw new Error(codec+' roundtrip');
            if(magic[codec]&&!magic[codec].every((v,i)=>got.encoded[i]===v))throw new Error(codec+' magic '+Array.from(got.encoded.slice(0,4)));
            out.codecs[codec]=got.bytes;
          }
          response=wait();worker.postMessage({type:'loop',reference,width,height,distance:1.5,effort:7,iters:2,steer:{metric:'msssim',map:'',scale:1,exponent:1},steering:'msssim test'});
          const run=await response;
          const head=new TextDecoder().decode(run.trace.subarray(0,10));
          if(head!=='JXLTRACE1\\n'||!(progress>0)||run.file.length<2||run.file[0]!==255)throw new Error('loop '+head+' '+progress);
          if(frames!==progress)throw new Error('live frames '+frames+' vs scored '+progress);
          out.loop={scored:progress,frames,bytes:run.file.length,trace:run.trace.byteLength};
          frames=0;
          response=wait();worker.postMessage({type:'loop',reference,width,height,distance:1.5,effort:7,iters:2,steer:null,steering:'butteraugli (built-in)'});
          const builtin=await response;
          if(!(frames>0)||builtin.file[0]!==255)throw new Error('built-in loop frames '+frames);
          out.builtin_loop={frames,bytes:builtin.file.length};
          response=wait();worker.postMessage({type:'loop',reference,width,height,distance:1.5,effort:7,iters:2,steer:{metric:'gmsd',map:'',scale:0,exponent:1},steering:'bad'});
          try{await response;throw new Error('A zero scale was accepted');}catch(error){if(!String(error.message).includes('scale and exponent'))throw error;}
          worker.terminate();return out;
        }''')
        # Both file controls exercise the UI's independent upload revisions.
        jpeg=bytes(result.pop('jpeg_bytes'))
        # The full module is fetched only after the demo's first score, then
        # in the background, so offline use below needs no fetch.
        page.wait_for_function("(document.body.dataset.modules||'').includes('full')",timeout=120000)
        first_score_ms=float(page.evaluate("document.body.dataset.firstScoreMs"))
        prefetch_ms=float(page.evaluate("document.body.dataset.prefetchMs"))
        assert first_score_ms < prefetch_ms, (first_score_ms, prefetch_ms)
        result['lazy_full_module']={'first_score_ms':first_score_ms,'prefetch_posted_ms':prefetch_ms}
        observed=[]
        context.on('request',lambda request:observed.append(request.url))
        context.set_offline(True)
        page.locator('#reference').set_input_files({'name':'reference.jpg','mimeType':'image/jpeg','buffer':jpeg})
        page.locator('#distorted').set_input_files({'name':'distorted.jpg','mimeType':'image/jpeg','buffer':jpeg})
        page.wait_for_function("document.getElementById('status').textContent.includes('native-cpu-wasm')",timeout=30000)
        assert '64×64' in page.locator('#dimensions').inner_text()
        assert page.locator('#score').inner_text()!='–'
        page.locator('#metric').select_option('ssim2')
        page.wait_for_function("document.getElementById('score-label').textContent==='ssim2' && document.getElementById('map-info').textContent.startsWith('ssim2 · diffmap ·')",timeout=30000)
        page.locator('#metric').select_option('gmsd')
        page.wait_for_function("document.getElementById('score-label').textContent==='gmsd' && document.getElementById('map-info').textContent.startsWith('gmsd ·')",timeout=30000)
        assert not observed,observed
        result['offline_upload_ui_and_scoring']=True
        context.set_offline(False)
        context.set_offline(False)
        page.locator('#metric').select_option('zensim')
        page.locator('#demo').click()
        page.wait_for_function("document.getElementById('status').textContent.includes('768×512 · native-cpu-wasm')",timeout=30000)
        assert page.locator('#score').inner_text()!='–'
        # Verify actual rendered maps at the demo size, rather than only the
        # existence of a finite raw map. A universal fixed maximum of one
        # saturated Butteraugli/CVVDP and hid Zensim's small-valued output.
        assert page.locator('#norm').input_value() == 'sqrt'
        rendered = {}
        for map_metric in ('zensim', 'butteraugli', 'cvvdp'):
            page.locator('#metric').select_option(map_metric)
            page.wait_for_function("metric => !document.getElementById('compare').disabled && document.getElementById('map-info').textContent.startsWith(metric+' ·') && document.getElementById('map-info').textContent.includes('per-image scale')",arg=map_metric,timeout=30000)
            page.locator('[data-mode="map"]').click()
            colors = page.evaluate("""() => {
              const canvas=document.getElementById('view');
              const pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
              const counts=new Map();
              for(let i=0;i<pixels.length;i+=4){const key=(pixels[i]<<16)|(pixels[i+1]<<8)|pixels[i+2];counts.set(key,(counts.get(key)||0)+1);}
              return {colors:counts.size,largest_fraction:Math.max(...counts.values())/(canvas.width*canvas.height)};
            }""")
            assert colors['colors'] > 32, (map_metric, colors)
            assert colors['largest_fraction'] < .5, (map_metric, colors)
            assert 'per-image scale' in page.locator('#map-info').inner_text()
            assert page.locator('#scale').is_disabled()
            rendered[map_metric] = colors
        # Fixed calibration is still explicit and does not recompute raw maps.
        page.locator('#norm').select_option('fixed')
        assert page.locator('#scale').is_enabled()
        assert 'fixed' in page.locator('#map-info').inner_text()
        page.locator('#norm').select_option('sqrt')
        page.locator('#metric').select_option('zensim')
        page.wait_for_function("document.getElementById('map-info').textContent.startsWith('zensim ·')",timeout=30000)
        result['rendered_demo_maps'] = rendered
        # Invoke both UI handlers in one browser turn: a fast comparison must
        # not finish between two separate automation commands.
        page.evaluate('''() => {
          document.getElementById('compare').click();
          const cancel=document.getElementById('cancel');
          if(cancel.disabled)throw new Error('Cancel was not enabled for the active job');
          cancel.click();
        }''' )
        page.wait_for_function("document.getElementById('status').textContent.startsWith('Ready:')",timeout=30000)
        assert page.locator('#cancel').is_disabled()
        page.locator('#compare').click()
        page.wait_for_function("document.getElementById('status').textContent.includes('768×512 · native-cpu-wasm')",timeout=30000)
        # Switching away and back must rebuild the cached reference for the
        # selected model, update both outputs and name the CLI profile.
        page.locator('details').filter(has_text='Run the same native metric').locator('summary').click()
        default_score = page.locator('#score').inner_text()
        for model in ('d', 'a', 'b'):
            page.locator('#zensim-model').select_option(model)
            page.wait_for_function("model => !document.getElementById('compare').disabled && document.getElementById('map-info').textContent.includes('model '+model+' ·')",arg=model,timeout=30000)
            assert '--zensim-profile '+model in page.locator('#cli-snippet').inner_text()
            assert 'model '+model in page.locator('#score-info').inner_text()
        page.locator('#zensim-model').select_option('c')
        page.wait_for_function("!document.getElementById('compare').disabled && document.getElementById('status').textContent.includes('Model C spatial maps')",timeout=30000)
        assert page.locator('#zensim-model-note').is_visible()
        assert page.locator('#map-info').inner_text() == 'No map yet'
        assert 'Map unavailable' in page.locator('#compute-timing').inner_text()
        assert 'model c' in page.locator('#score-info').inner_text()
        page.locator('#zensim-model').select_option('b')
        page.wait_for_function("!document.getElementById('compare').disabled && document.getElementById('map-info').textContent.includes('model b ·')",timeout=30000)
        assert page.locator('#score').inner_text() == default_score
        result['ui_model_selection'] = True
        assert 'Reference ' in page.locator('#compute-timing').inner_text()
        assert 'Score ' in page.locator('#compute-timing').inner_text()
        assert 'Map ' in page.locator('#compute-timing').inner_text()
        assert 'Worker startup ' in page.locator('#startup-timing').inner_text()
        result['ui_separate_timings']=True
        # MS-SSIM stays last: the border check below reads its cropped origin.
        for map_metric in ('ssim2','nlpd','psnrhvs','mdctpsnr','mad','iwssim','vif','vifvec','vmaf-v1','fsim','haarpsi','vsi','dssim','msssim'):
            page.locator('#metric').select_option(map_metric)
            page.wait_for_function("metric => !document.getElementById('compare').disabled && document.getElementById('map-info').textContent.startsWith(metric+' ·')",arg=map_metric,timeout=30000)
            assert 'native samples' in page.locator('#map-info').inner_text()
            assert page.locator('#score').inner_text()!='–'
        page.locator('[data-mode="map"]').click()
        border=page.evaluate("Array.from(document.getElementById('view').getContext('2d').getImageData(0,0,1,1).data)")
        assert border == [190,190,190,255], border
        page.locator('#view').dispatch_event('pointermove',{'clientX':page.locator('#view').bounding_box()['x']+1,'clientY':page.locator('#view').bounding_box()['y']+1})
        assert 'no native map sample' in page.locator('#pixel').inner_text()
        result['new_map_ui_and_cropped_border']=True
        result['demo_completion']=True
        result['ui_cancel_recovery']=True
        result['capability_report']=page.locator('#capabilities').inner_text()
        result['incompatible_memory_refused']=mismatch
        # Encoder loop player: a synthetic two-frame trace, 16x8 pixels in two
        # 8x8 blocks, in the format jxl-encoder's loop_trace_dump writes.
        import struct
        lw, lh = 16, 8
        head = {'width': lw, 'height': lh, 'distance': 1.0, 'effort': 7, 'iters': 1, 'bytes': 123,
                'xsize_blocks': 2, 'ysize_blocks': 1,
                'frames': [{'seed': 0.6, 'iter': 0, 'metric': 'butteraugli', 'score': 1.5, 'target': 1.0, 'has_next': True},
                           {'seed': 0.6, 'iter': 1, 'metric': 'butteraugli', 'score': 1.2, 'target': 1.0, 'has_next': False}]}
        hb = json.dumps(head).encode()
        f32 = lambda v: struct.pack('<%df' % len(v), *v)
        trace = b'JXLTRACE1\n' + struct.pack('<I', len(hb)) + hb + bytes([10, 20, 30] * (lw * lh))
        trace += f32([1, 1]) + f32([2, 0.5]) + f32([1.5, 0.5]) + f32([i / (lw * lh) for i in range(lw * lh)]) + bytes([200, 100, 50] * (lw * lh))
        trace += f32([2, 0.5]) + f32([1.1, 0.9]) + f32([0.1] * (lw * lh)) + bytes([210, 110, 60] * (lw * lh))
        page.locator('#trace').set_input_files({'name': 'synthetic.ltrace', 'mimeType': 'application/octet-stream', 'buffer': trace})
        page.wait_for_function("document.getElementById('loop-info').textContent.startsWith('Frame ') && document.getElementById('loop-info').textContent.includes('synthetic.ltrace') && document.getElementById('loop-play').textContent.includes('Pause')", timeout=10000)
        page.locator('#loop-play').click()
        seek(0)
        info = page.locator('#loop-info').inner_text()
        assert info.startswith('Frame 1 of 2 · butteraugli'), info
        text = lambda id: page.locator(id).inner_text()
        assert text('#ls-iter') == '0 of 1 · seed 0.6', text('#ls-iter')
        assert text('#ls-score') == '1.500 / 1.000 (150%)', text('#ls-score')
        assert text('#ls-blocks') == '1 of 2 (50.0%)' and text('#ls-worst') == 'block 0, 0 · 1.500 (150%)', (text('#ls-blocks'), text('#ls-worst'))
        assert text('#ls-next') == '1 finer · 1 coarser' and text('#ls-encode') == '16×8 · d1 e7 · 123 B'
        loop_px = lambda x, y: page.evaluate(f"Array.from(document.getElementById('loop-view').getContext('2d').getImageData({x},{y},1,1).data)")
        # Default layer: the block above the target is outlined in magenta,
        # the block below it is not; the map over the reconstruction leaves a
        # zero-map pixel exactly the reconstruction.
        assert loop_px(0, 0) == [255, 0, 255, 255] and loop_px(7, 7) == [255, 0, 255, 255], (loop_px(0, 0), loop_px(7, 7))
        assert loop_px(8, 0) != [255, 0, 255, 255]
        page.locator('#loop-layer').select_option('overlay')
        assert loop_px(0, 0) == [200, 100, 50, 255]
        page.locator('#loop-layer').select_option('dqf')
        assert loop_px(1, 1) == [255, 0, 0, 255], loop_px(1, 1)  # left block finer: red
        assert loop_px(9, 1) == [0, 0, 255, 255], loop_px(9, 1)  # right block coarser: blue
        page.locator('#loop-next').click()
        info = page.locator('#loop-info').inner_text()
        assert info.startswith('Frame 2 of 2') and text('#ls-next') == 'none: final pass, score only', (info, text('#ls-next'))
        page.locator('#loop-layer').select_option('recon')
        assert loop_px(3, 3) == [210, 110, 60, 255]
        page.locator('#trace').set_input_files({'name': 'bad.ltrace', 'mimeType': 'application/octet-stream', 'buffer': b'nope'})
        page.wait_for_function("document.getElementById('loop-info').textContent.includes('Not a jxl-encoder loop trace')", timeout=10000)
        result['loop_player'] = True
        # The URL contract (docs/BROWSER_WORKBENCH_LINKS.md) and the links back.
        links = context.new_page()
        links.goto(url + '?metric=psnr-y&codec=jxl&distance=2')
        links.wait_for_function("document.getElementById('score-label').textContent==='psnr-y' && document.getElementById('status').textContent.includes('768×512')", timeout=120000)
        assert links.locator('#metric').input_value() == 'psnr-y'
        assert links.locator('#name-distorted').inner_text().startswith('jxl-encoder · distance 2 · ')
        assert links.locator('#jxl-distance').input_value() == '2'
        from urllib.parse import urlparse, parse_qs
        query = parse_qs(urlparse(links.url).query)
        assert query.get('metric') == ['psnr-y'] and query.get('codec') == ['jxl'] and query.get('distance') == ['2'], links.url
        hrefs = links.locator('#metric-links a').evaluate_all('a => a.map(x => x.href)')
        assert 'https://imazen.github.io/zenmetrics/#models?metric=psnr-y' in hrefs, hrefs
        assert 'https://imazen.github.io/zenmetrics/#speed?metric=psnr-y' in hrefs, hrefs
        links.locator('#metric').select_option('gmsd')
        links.wait_for_function("new URL(location.href).searchParams.get('metric')==='gmsd'", timeout=10000)
        # JPEG XL impact: two distances scored with the selected metric; a row
        # loads its encode as the distorted image.
        links.locator('#jxl-sweep-distances').fill('1, 4')
        links.locator('#jxl-sweep-run').click()
        links.wait_for_function("document.getElementById('jxl-status').textContent.startsWith('Done: 2 encodes scored with gmsd')", timeout=300000)
        rows = links.locator('#jxl-table tbody tr')
        assert rows.count() == 2
        sweep = [[c.inner_text() for c in rows.nth(i).locator('td').all()] for i in range(2)]
        assert all(math.isfinite(float(r[3])) and int(r[1]) > 0 for r in sweep), sweep
        assert links.locator('#jxl-chart circle').count() == 2
        rows.nth(1).click()
        links.wait_for_function("document.getElementById('name-distorted').textContent.startsWith('jxl-encoder · distance 4 · ') && document.getElementById('map-info').textContent.startsWith('gmsd ·')", timeout=120000)
        # The viewer's comparison of the loaded row repeats the sweep's score.
        row_value = float(rows.nth(1).get_attribute('data-value'))
        shown = links.locator('#score').inner_text()
        assert shown == links.evaluate('v => v.toPrecision(7)', row_value), (shown, row_value)
        result['jxl_impact_sweep'] = sweep
        links.close()
        # A metric with no browser route opens with its measured reason, no score.
        gpu = context.new_page()
        gpu.goto(url + '?metric=ssim2-gpu')
        gpu.wait_for_function("document.getElementById('status').textContent.includes('ssim2-gpu is not runnable in the browser')", timeout=120000)
        assert gpu.locator('#metric').input_value() == 'ssim2-gpu' and gpu.locator('#score').inner_text() == '–'
        assert 'not runnable in the browser' in gpu.locator('#metric-status').inner_text()
        gpu.close()
        # Bad parameters are refused by name; the fragment form is read too.
        bad = context.new_page()
        bad.goto(url + '?metric=nope&q=500#codec=webp')
        bad.wait_for_function("document.getElementById('metric-links').textContent.includes('Link parameters refused')", timeout=120000)
        refusal_text = bad.locator('#link-errors').inner_text()
        assert 'metric=nope' in refusal_text and 'q=500' in refusal_text, refusal_text
        assert bad.locator('#codec').input_value() == 'webp'
        bad.close()
        result['url_contract'] = True
        # The full module's fetch fails: the page says so, scores nothing in
        # its place, and loads it once the network allows.
        lazy = context.new_page()
        lazy.route('**/pkg-full/**', lambda route: route.abort())
        lazy.goto(url + '?metric=psnr')
        lazy.wait_for_function("document.getElementById('status').textContent.includes('Failed to fetch') || document.getElementById('status').textContent.includes('did not load')", timeout=120000)
        assert lazy.locator('#score').inner_text() == '–', lazy.locator('#score').inner_text()
        lazy.unroute('**/pkg-full/**')
        lazy.locator('#compare').click()
        lazy.wait_for_function("document.getElementById('score-label').textContent==='psnr' && /^[0-9]/.test(document.getElementById('score').textContent)", timeout=120000)
        lazy.close()
        result['full_module_fetch_failure_recovery'] = True
        check_learned(playwright, browser, url, result)
        # Every CLI metric name's browser status, from the executions above.
        measured = measured_coverage(result)
        result['coverage'] = measured
        if args.write_coverage:
            assert args.nn_models == 'all', 'coverage is written only from a run of every learned metric'
            args.write_coverage.write_text(coverage_text(measured))
            if not args.url:
                # The page reads its statuses from this file.
                (args.site / 'coverage.json').write_text(coverage_text(measured))
        else:
            check_coverage(json.loads((args.site / 'coverage.json').read_text()), measured)
        check_learned_ui(browser, url, result)
        if args.oracle:
            oracle=json.loads(args.oracle.read_text())
            assert result['score']['score']==oracle['score']
            assert result['map']==oracle['map']
            for name, expected in oracle.get('profiles',{}).items():
                got=result['profiles'][name]
                delta=max((abs(a-b) for a,b in zip(got['map'] or [],expected['map'] or [])),default=0)
                assert got == expected, {'profile':name,'browser_score':got['score'],'native_score':expected['score'],'max_map_delta':delta}
            # Every browser-scored metric against the native build of the same
            # entry on the same fixture. Differences are recorded per column.
            values=lambda s:{v['column']:v['value'] for v in s.get('values',[{'column':s.get('metric','zensim'),'value':s.get('score')}])}
            assert set(oracle['scalars'])==set(result['metrics']),sorted(set(oracle['scalars'])^set(result['metrics']))
            result['native_scalar_delta']={}
            for name,expected in oracle['scalars'].items():
                assert result['scalarFixture'][name]==expected['fixture'],name
                got,want=values(result['metrics'][name]),values(expected['score'])
                assert list(got)==list(want),(name,got,want)
                result['native_scalar_delta'][name]={c:(0.0 if got[c]==want[c] else abs(got[c]-want[c]) if isinstance(got[c],(int,float)) and isinstance(want[c],(int,float)) else f'{got[c]} vs {want[c]}') for c in got}
            result['native_map_max_abs_delta']={}
            for name,expected in oracle['spatial_maps'].items():
                got=result['spatial'][name]
                assert (got['width'],got['height'],got['similarity'])==(expected['width'],expected['height'],expected['similarity'])
                assert len(got['values'])==len(expected['values'])
                if 'geometry' in expected: assert got['geometry']==expected['geometry'],name
                assert all(math.isfinite(v) for v in got['values'])
                delta=max(abs(a-b) for a,b in zip(got['values'],expected['values']))
                result['native_map_max_abs_delta'][name]=delta
                # IW-SSIM and mDCT-PSNR record their delta: neither claims cross-tier bits, and
                # mDCT-PSNR calls platform libm (glibc natively, not on wasm32).
                if name in ('psnr','fsim','fsimc','gmsd','haarpsi','msssim','vsi','ssim2','nlpd','psnrhvs','mad','vif','vifvec','vmaf-v1'): assert delta==0, (name,delta)
        if args.report:
            source_file=args.site/'site-source.json'
            asset_prefix=json.loads(source_file.read_text()).get('asset_prefix','') if source_file.is_file() else ''
            wasm=(args.site/asset_prefix/'pkg/zenmetrics_browser_bg.wasm').read_bytes()
            result['wasm_bytes']=len(wasm);result['wasm_gzip_bytes']=len(gzip.compress(wasm,mtime=0))
            result['browser']=browser.version
            args.report.write_text(json.dumps(result,indent=2)+'\n')
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
