import init, { initThreadPool, Reference, thread_count, available_metrics, score_rgba, decode_image, encode_jpeg, spatial_map, JobControl } from './pkg/zenmetrics_browser.js';
let ready = false, cached, previousReference, wasm, control;
self.onmessage = async ({data}) => {
  try {
    if (data.type === 'init') {
      if (!self.crossOriginIsolated || typeof SharedArrayBuffer === 'undefined') {
        throw new Error('Threaded WASM requires cross-origin isolation and SharedArrayBuffer');
      }
      postMessage({type:'progress', message:'Instantiating WASM…'});
      const relaxedProbe = new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,15,1,13,0,65,0,253,15,65,0,253,15,253,128,2,11]);
      if (!WebAssembly.validate(relaxedProbe)) throw new Error('Relaxed SIMD128 is required');
      const wasmStart = performance.now();
      wasm = await init(); // Compilation rejects unsupported SIMD/relaxed SIMD; no alternate module.
      const wasmMilliseconds = performance.now()-wasmStart;
      postMessage({type:'progress', message:'Starting Rayon workers…'});
      const poolStart = performance.now();
      let timer;
      try {
        await Promise.race([initThreadPool(data.threads), new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error('Rayon worker initialization deadline expired')), 15000);
        })]);
      } finally { clearTimeout(timer); }
      const poolMilliseconds=performance.now()-poolStart;
      ready = true;
      if (thread_count() !== data.threads) throw new Error('Rayon pool size differs from the request');
      postMessage({type: 'ready', threads: thread_count(), metrics: JSON.parse(available_metrics()), startup:{wasmMilliseconds,poolMilliseconds}});
      return;
    }
    if (!ready) throw new Error('WASM worker has not initialized');
    control = new JobControl();
    postMessage({type:'control',id:data.id,buffer:wasm.memory.buffer,offset:control.state_ptr()});
    const phase = (number, total=0) => control.begin(number,total);
    const unpack = packed => {
      const width=packed[0],height=packed[1],rgba=new Uint8Array(width*height*4);
      for(let i=0;i<width*height;i++){const p=packed[i+2];rgba[i*4]=p&255;rgba[i*4+1]=(p>>>8)&255;rgba[i*4+2]=(p>>>16)&255;rgba[i*4+3]=p>>>24;}
      return {width,height,rgba};
    };
    if(data.type==='decode') {
      phase(1);const image=unpack(decode_image(data.bytes,data.filename));
      postMessage({type:'decoded',id:data.id,...image},[image.rgba.buffer]);return;
    }
    if(data.type==='generate') {
      phase(2);const encoded=encode_jpeg(data.reference,data.width,data.height,data.width*4,data.quality);
      phase(1);const image=unpack(decode_image(encoded,'generated.jpg'));
      postMessage({type:'decoded',id:data.id,bytes:encoded.length,encoded,...image},[image.rgba.buffer,encoded.buffer]);return;
    }
    const profile = data.zensimProfile || 'b';
    const start = performance.now();
    const {reference, distorted, width, height, metric, id} = data;

    let referencePrepared=false;
    if ((metric==='zensim'||!metric||data.mapMetric==='zensim'||!data.mapMetric) && (!previousReference || previousReference.profile !== profile || previousReference.width !== width || previousReference.height !== height ||
        previousReference.bytes.length !== reference.length || !reference.every((v,i) => v === previousReference.bytes[i]))) {
      referencePrepared=true;phase(3);const prepared = Reference.with_profile(reference, width, height, width * 4, profile);
      if (cached) cached.free();
      cached = prepared;
      previousReference = {width, height, profile, bytes: reference.slice()};
    }
    const referenceMilliseconds=performance.now()-start;
    const selected = metric || 'zensim';
    const mapMetric = data.mapMetric || 'zensim';
    let score;const scoreStart=performance.now();phase(4);
    if(selected==='zensim') score={metric:'zensim',profile,backend:'native-cpu-wasm',score:cached.score(distorted,width*4)};
    else score=JSON.parse(score_rgba(selected,reference,distorted,width,height,width*4,data.display||''));
    const scoreMilliseconds=performance.now()-scoreStart;
    postMessage({type:'score',id,score,scoreMilliseconds,referenceMilliseconds,referencePrepared});
    const mapStart=performance.now();phase(5);
    let map, mapWidth=width, mapHeight=height, similarity=false;
    if(mapMetric==='zensim') {
      const compared=cached.compare(distorted,width*4);map=compared.map;compared.free();
    } else {
      const spatial=spatial_map(mapMetric,reference,distorted,width,height,width*4,data.display||'standard_4k');
      map=spatial.values;mapWidth=spatial.width;mapHeight=spatial.height;similarity=spatial.similarity;spatial.free();
    }
    postMessage({type:'result',id,score,map,mapMetric,mapProfile:mapMetric==='zensim'?profile:null,mapWidth,mapHeight,similarity,referenceMilliseconds,referencePrepared,scoreMilliseconds,mapMilliseconds:performance.now()-mapStart,milliseconds:performance.now()-start},[map.buffer]);
  } catch (error) {
    postMessage({type: 'error', id: data.id, message: String(error), fatal: error instanceof WebAssembly.RuntimeError});
  }
};
