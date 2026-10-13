#!/usr/bin/env python3
"""Verify isolated threaded Pages startup and refusal without isolation."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import math
import json
import gzip
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('site', type=Path)
parser.add_argument('--url', help='Check an already deployed HTTPS site instead')
parser.add_argument('--oracle', type=Path, help='Independent native map outputs')
parser.add_argument('--report', type=Path, help='Measured browser results')
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
          let publishedScore, sharedControls=0, earlyScores=0;
          const wait = () => new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('Worker deadline expired')), 15000);
            worker.onmessage = ({data}) => {
              if(data.type==='control'){if(!(data.buffer instanceof SharedArrayBuffer)||data.offset%4)throw new Error('Shared atomic control required');sharedControls++;return;}
              if(data.type==='score'){publishedScore=data.score;earlyScores++;return;}
              if(data.type==='progress')return;
              if(data.type==='result'&&JSON.stringify(data.score)!==JSON.stringify(publishedScore))throw new Error('Final score differs from published scalar');
              clearTimeout(timer);resolve(data);
            };
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
          const spatial={};
          for(const mapMetric of ['butteraugli','dssim','cvvdp','psnr','gmsd']) {
            response=wait();worker.postMessage({type:'compare',metric:'zensim',mapMetric,reference,distorted,width,height,display:'standard_4k'});
            const got=await response;if(got.type!=='result')throw new Error(JSON.stringify(got));
            spatial[mapMetric]={width:got.mapWidth,height:got.mapHeight,similarity:got.similarity,values:Array.from(got.map)};
          }
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
          const metrics={};
          for(const metric of ready.metrics) {
            response=wait();worker.postMessage({type:'compare',metric,mapMetric:'psnr',reference:r,distorted:d,...input,display:'standard_4k'});
            const got=await response;if(got.type!=='result')throw new Error(metric+': '+JSON.stringify(got));metrics[metric]=got.score;
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
          worker.terminate();return {shared_control_messages:sharedControls,early_score_messages:earlyScores,zensim_1080p_ms:large.milliseconds,atomic_cancellation:true,score:first.score,map:Array.from(first.map),spatial,metrics,ssim2_1080p_worker_ms:timed.milliseconds,ssim2_1080p_score_ms:timed.scoreMilliseconds,jpeg_upload_recovery:true,jpeg_bytes:Array.from(jpeg.encoded)};
        }''')
        assert len(result['map']) == 4096
        assert all(math.isfinite(value) for value in result['map'])
        # Both file controls exercise the UI's independent upload revisions.
        jpeg=bytes(result.pop('jpeg_bytes'))
        observed=[]
        context.on('request',lambda request:observed.append(request.url))
        context.set_offline(True)
        page.locator('#reference').set_input_files({'name':'reference.jpg','mimeType':'image/jpeg','buffer':jpeg})
        page.locator('#distorted').set_input_files({'name':'distorted.jpg','mimeType':'image/jpeg','buffer':jpeg})
        page.wait_for_function("document.getElementById('status').textContent.includes('native-cpu-wasm')",timeout=30000)
        assert '64×64' in page.locator('#dimensions').inner_text()
        assert page.locator('#score').inner_text()!='–'
        page.locator('#metric').select_option('ssim2')
        page.wait_for_function("document.getElementById('score-label').textContent==='ssim2'",timeout=30000)
        page.locator('#map-metric').select_option('gmsd')
        page.wait_for_function("document.getElementById('map-info').textContent.startsWith('gmsd')",timeout=30000)
        assert not observed,observed
        result['offline_upload_ui_and_scoring']=True
        context.set_offline(False)
        context.set_offline(False)
        page.locator('#metric').select_option('zensim')
        page.locator('#map-metric').select_option('zensim')
        page.locator('#demo').click()
        page.wait_for_function("document.getElementById('status').textContent.includes('768×512 · native-cpu-wasm')",timeout=30000)
        assert page.locator('#score').inner_text()!='–'
        page.locator('#compare').click()
        page.locator('#cancel').click()
        page.wait_for_function("document.getElementById('status').textContent.startsWith('Ready:')",timeout=30000)
        assert page.locator('#cancel').is_disabled()
        page.locator('#compare').click()
        page.wait_for_function("document.getElementById('status').textContent.includes('768×512 · native-cpu-wasm')",timeout=30000)
        result['demo_completion']=True
        result['ui_cancel_recovery']=True
        if args.oracle:
            oracle=json.loads(args.oracle.read_text())
            assert result['score']['score']==oracle['score']
            assert result['map']==oracle['map']
            result['native_map_max_abs_delta']={}
            for name,expected in oracle['spatial_maps'].items():
                got=result['spatial'][name]
                assert (got['width'],got['height'],got['similarity'])==(expected['width'],expected['height'],expected['similarity'])
                assert len(got['values'])==len(expected['values'])
                assert all(math.isfinite(v) for v in got['values'])
                delta=max(abs(a-b) for a,b in zip(got['values'],expected['values']))
                result['native_map_max_abs_delta'][name]=delta
                if name=='psnr': assert delta==0
        if args.report:
            wasm=(args.site/'pkg/zenmetrics_browser_bg.wasm').read_bytes()
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
