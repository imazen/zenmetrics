import initCore, * as coreApi from './pkg/zenmetrics_browser.js';
// Two builds of one module. `core` (decoders for JPEG, the JPEG encoder and
// Zensim) loads at start; `full` (every CPU metric, PNG/WebP/JPEG XL
// decoding) is imported the first time something needs it. Each has its own
// memory and Rayon pool. No request is ever answered by a module that does
// not compile the requested metric.
const modules={core:null,full:null};
let threads=1,cached,previousReference;
// Module a request used last for spatial maps, so `map` reads its cache.
let lastMapModule=null;
// Encoder modules (web/codecs), imported the first time each is used.
const codecs = {};
async function codec(name) {
  if (!codecs[name]) {
    const module = await import(new URL(`./codecs/${name}/zenmetrics_codec_${name}.js`, import.meta.url));
    await module.default();
    codecs[name] = module;
  }
  return codecs[name];
}
async function startModule(name, api, init) {
  const start=performance.now();
  const wasm=await init(); // Compilation rejects unsupported SIMD/relaxed SIMD; no alternate module.
  const wasmMilliseconds=performance.now()-start;
  const poolStart=performance.now();
  let timer;
  try {
    await Promise.race([api.initThreadPool(threads), new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('Rayon worker initialization deadline expired')), 15000);
    })]);
  } finally { clearTimeout(timer); }
  if (api.thread_count() !== threads) throw new Error('Rayon pool size differs from the request');
  const module={name,api,wasm,metrics:JSON.parse(api.available_metrics()),maps:JSON.parse(api.available_spatial_maps()),
    startup:{wasmMilliseconds,poolMilliseconds:performance.now()-poolStart}};
  if(api.module_name()!==name)throw new Error(`Module ${name} reports ${api.module_name()}`);
  modules[name]=module;
  return module;
}
let fullLoading=null,fullAttempts=0;
async function full(announce=true) {
  if(modules.full)return modules.full;
  if(announce)postMessage({type:'progress',message:'Loading the full metric module…'});
  fullLoading??=(async()=>{
    // A failed dynamic import stays failed for its URL, so a retry names
    // the same module with a distinct query.
    const retry=fullAttempts++?`?retry=${fullAttempts}`:'';
    const api=await import(new URL(`./pkg-full/zenmetrics_browser.js${retry}`, import.meta.url));
    const module=await startModule('full',api,api.default);
    postMessage({type:'module',name:'full',startup:module.startup,metrics:module.metrics,maps:module.maps});
    return module;
  })();
  try { return await fullLoading; } catch(error) { fullLoading=null; throw error; }
}
// Extra modules (extra-modules.json beside the page, absent on the public
// site): each lists its metrics and their maps, is imported the first time
// one of them is used, and answers the same calls as core and full. They
// may be single-threaded (no Rayon pool) and use unshared memory.
let extras=[];
const extraLoading={};
async function loadExtras() {
  const response=await fetch(new URL('extra-modules.json', import.meta.url),{cache:'no-store'}).catch(()=>null);
  // Pages answers a missing file with the index page (200, text/html), so
  // only a JSON response is a manifest; anything unreadable means none.
  if(!response||!response.ok||!(response.headers.get('content-type')||'').includes('json'))return [];
  const manifest=await response.json().catch(()=>null);
  return (manifest?.modules||[]).filter(m=>m&&m.name&&m.url&&Array.isArray(m.metrics));
}
const extraFor = metric => extras.find(e=>e.metrics.some(m=>m.name===metric));
async function extra(entry) {
  if(modules[entry.name])return modules[entry.name];
  postMessage({type:'progress',message:`Loading the ${entry.label||entry.name} module…`});
  extraLoading[entry.name]??=(async()=>{
    const api=await import(new URL(entry.url, import.meta.url));
    if(api.initThreadPool)return startModule(entry.name,api,api.default);
    const start=performance.now();
    const wasm=await api.default();
    const module={name:entry.name,api,wasm,metrics:JSON.parse(api.available_metrics()),maps:JSON.parse(api.available_spatial_maps()),
      startup:{wasmMilliseconds:performance.now()-start,poolMilliseconds:0}};
    if(api.module_name()!==entry.name)throw new Error(`Module ${entry.name} reports ${api.module_name()}`);
    modules[entry.name]=module;
    postMessage({type:'module',name:entry.name,startup:module.startup,metrics:module.metrics,maps:module.maps});
    return module;
  })();
  try { return await extraLoading[entry.name]; } catch(error) { delete extraLoading[entry.name]; throw error; }
}
const forMetric = async metric => modules.core.metrics.includes(metric) ? modules.core : extraFor(metric) ? extra(extraFor(metric)) : full();
const forMap = async metric => modules.core.maps.includes(metric) ? modules.core : extraFor(metric) ? extra(extraFor(metric)) : full();
function mapMessage(spatial) {
  const out = {map:spatial.values,mapWidth:spatial.width,mapHeight:spatial.height,similarity:spatial.similarity,geometry:JSON.parse(spatial.geometry),
    mapName:spatial.name,mapNames:JSON.parse(spatial.names),mapPolarity:spatial.polarity};
  spatial.free();
  return out;
}
self.onmessage = async ({data}) => {
  try {
    if (data.type === 'init') {
      if (!self.crossOriginIsolated || typeof SharedArrayBuffer === 'undefined') {
        throw new Error('Threaded WASM requires cross-origin isolation and SharedArrayBuffer');
      }
      postMessage({type:'progress', message:'Instantiating WASM…'});
      const relaxedProbe = new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,15,1,13,0,65,0,253,15,65,0,253,15,253,128,2,11]);
      if (!WebAssembly.validate(relaxedProbe)) throw new Error('Relaxed SIMD128 is required');
      threads=data.threads;
      postMessage({type:'progress', message:'Starting Rayon workers…'});
      const core=await startModule('core',coreApi,initCore);
      extras=await loadExtras();
      postMessage({type: 'ready', threads: coreApi.thread_count(), module:'core', metrics: core.metrics, maps:core.maps, allMetrics:JSON.parse(coreApi.all_metric_names()),
        loopDefaults:JSON.parse(coreApi.loop_map_defaults()), startup:core.startup,
        extras:extras.map(({name,label,metrics,loopDefaults})=>({name,label:label||name,metrics,loopDefaults:loopDefaults||[]}))});
      return;
    }
    if (!modules.core) throw new Error('WASM worker has not initialized');
    if (data.type === 'prefetch') {
      // Load the full module in the background after the first score, so
      // later metric choices (and offline use) need no fetch.
      try { await full(false); } catch(error) { postMessage({type:'prefetch-failed',message:String(error)}); }
      return;
    }
    if (data.type === 'catalog') {
      // Load the full module and report what each module scores.
      const loaded=await full();
      postMessage({type:'catalog',id:data.id,modules:Object.fromEntries(Object.values(modules).filter(Boolean).map(m=>[m.name,{metrics:m.metrics,maps:m.maps,startup:m.startup}])),
        all:JSON.parse(loaded.api.all_metric_names())});
      return;
    }
    if (data.type === 'nn-load-local') {
      const nn=await import(new URL('./nn.js', import.meta.url));
      await nn.loadLocal(data);
      postMessage({type:'local-loaded',id:data.id,metric:data.metric});return;
    }
    if (data.type === 'nn') {
      // Learned metrics: tensors from the full module, inference in nn.js.
      const module=await full(false);
      const nn=await import(new URL('./nn.js', import.meta.url));
      const start=performance.now();
      const result=await nn.score({...data,prepare:module.api.nn_prepare,wdLayer:module.api.nn_wd_layer,progress:message=>postMessage({type:'progress',message})});
      postMessage({type:'result',id:data.id,...result,map:null,mapWidth:0,mapHeight:0,mapMetric:null,mapNames:[],milliseconds:performance.now()-start});
      return;
    }
    // Each module a request runs in gets its own shared cancellation words.
    const controls=new Map();
    const control=module=>{
      if(!controls.has(module.name)){
        const c=new module.api.JobControl();controls.set(module.name,c);
        // Only shared memory can carry cancellation into a running call;
        // an extra module's unshared memory would only be copied.
        if(module.wasm.memory.buffer instanceof SharedArrayBuffer)
          postMessage({type:'control',id:data.id,module:module.name,buffer:module.wasm.memory.buffer,offset:c.state_ptr()});
      }
      return controls.get(module.name);
    };
    const phase = (module, number, total=0) => control(module).begin(number,total);
    const unpack = packed => {
      const width=packed[0],height=packed[1],rgba=new Uint8Array(width*height*4);
      for(let i=0;i<width*height;i++){const p=packed[i+2];rgba[i*4]=p&255;rgba[i*4+1]=(p>>>8)&255;rgba[i*4+2]=(p>>>16)&255;rgba[i*4+3]=p>>>24;}
      return {width,height,rgba};
    };
    if(data.type==='decode') {
      // JPEG decodes in the core module; other formats in the full module.
      let module=modules.core,packed;
      phase(module,1);
      try { packed=module.api.decode_image(data.bytes,data.filename); }
      catch(error) {
        if(!String(error).includes('decoder not in this module'))throw error;
        module=await full();phase(module,1);packed=module.api.decode_image(data.bytes,data.filename);
      }
      const image=unpack(packed);
      postMessage({type:'decoded',id:data.id,module:module.name,...image},[image.rgba.buffer]);return;
    }
    if(data.type==='generate') {
      const {width,height,quality}=data,name=data.codec||'jpeg';
      const core=modules.core;
      phase(core,2);
      if(name==='jpeg') {
        const encoded=core.api.encode_jpeg(data.reference,data.width,data.height,data.width*4,quality);
        phase(core,1);const image=unpack(core.api.decode_image(encoded,'generated.jpg'));
        postMessage({type:'decoded',id:data.id,bytes:encoded.length,encoded,...image},[image.rgba.buffer,encoded.buffer]);return;
      }
      postMessage({type:'progress',message:`Loading the ${name} encoder module…`});
      const module=await codec(name);
      // JPEG XL takes a Butteraugli distance directly when one is given.
      const coded=name==='avif'?module.roundtrip(data.reference,width,height,quality,data.speed||6)
        :name==='jxl'?(data.distance!==undefined?module.roundtrip_distance(data.reference,width,height,data.distance,data.effort||7):module.roundtrip(data.reference,width,height,quality,data.effort||7))
        :module.roundtrip(data.reference,width,height,quality);
      const encoded=coded.file,rgba=coded.rgba;coded.free();
      postMessage({type:'decoded',id:data.id,bytes:encoded.length,encoded,width,height,rgba,distance:data.distance},[rgba.buffer,encoded.buffer]);return;
    }
    if(data.type==='loop') {
      // jxl-encoder's perceptual loop, scored by its own butteraugli or by a
      // viewer metric's map put on butteraugli's scale (loop_map).
      const {reference,width,height,steer,id}=data;
      postMessage({type:'loop-progress',id,message:'Loading the JPEG XL encoder module…'});
      const module=await codec('jxl');
      if(steer&&!(steer.scale>0&&Number.isFinite(steer.scale)&&steer.exponent>0&&Number.isFinite(steer.exponent)))
        throw new Error('scale and exponent must be finite and positive');
      // The scorer runs synchronously inside the encoder, so the module that
      // computes the steering map is loaded before encoding starts.
      const mapper=steer?await forMap(steer.metric):null;
      postMessage({type:'loop-progress',id,message:`Encoding ${width}×${height}, steered by ${data.steering}…`});
      // A failed scorer ends the loop's seed early inside the encoder; report
      // it instead of returning that encode as if it were steered.
      let scored=0,failure=null;
      const scorer=steer?recon=>{
        try{
          const m=mapper.api.loop_map(steer.metric,steer.map,reference,recon,width,height,width*4,steer.display||'standard_4k',steer.scale,steer.exponent);
          const out={values:m.values,score:m.score};m.free();
          postMessage({type:'loop-progress',id,scored:++scored,score:out.score});
          return out;
        }catch(error){failure??=String(error);throw error;}
      }:undefined;
      // Each scored iteration streams to the page as it happens.
      const observer=(recon,map,score,index)=>{
        const values=new Float32Array(map);
        postMessage({type:'loop-frame',id,recon,map:values,score,index,width,height},[recon.buffer,values.buffer]);
      };
      const start=performance.now();
      const run=module.loop_encode(reference,width,height,data.distance,data.effort,data.iters,scorer,observer,data.steering);
      if(failure){run.free();throw new Error(`The steering metric failed: ${failure}`);}
      const trace=run.trace,file=run.file,rgba=run.rgba;run.free();
      postMessage({type:'loop',id,trace,file,rgba,milliseconds:performance.now()-start},[trace.buffer,file.buffer,rgba.buffer]);return;
    }
    if(data.type==='map') {
      // Another map of the last evaluation, without evaluating again.
      if(!lastMapModule)throw new Error('No evaluation has published maps yet');
      phase(lastMapModule,5);const mapStart=performance.now();
      postMessage({type:'result',id:data.id,...mapMessage(lastMapModule.api.cached_spatial_map(data.mapName)),mapMetric:data.mapMetric,mapMilliseconds:performance.now()-mapStart},[]);return;
    }
    const profile = data.zensimProfile || 'b';
    const {reference, distorted, width, height, metric, id} = data;
    const mapMetric = data.mapMetric === undefined ? 'zensim' : data.mapMetric;
    const selected = metric || 'zensim';
    // Load what this comparison needs before timing it.
    const scorer=await forMetric(selected);
    const mapper=mapMetric&&mapMetric!=='zensim'?await forMap(mapMetric):null;
    const core=modules.core;
    const start = performance.now();
    let referencePrepared=false;
    if ((selected==='zensim'||mapMetric==='zensim') && (!previousReference || previousReference.profile !== profile || previousReference.width !== width || previousReference.height !== height ||
        previousReference.bytes.length !== reference.length || !reference.every((v,i) => v === previousReference.bytes[i]))) {
      referencePrepared=true;phase(core,3);const prepared = core.api.Reference.with_profile(reference, width, height, width * 4, profile);
      if (cached) cached.free();
      cached = prepared;
      previousReference = {width, height, profile, bytes: reference.slice()};
    }
    const referenceMilliseconds=performance.now()-start;
    let score;const scoreStart=performance.now();phase(scorer,4);
    if(selected==='zensim') score={metric:'zensim',profile,backend:'native-cpu-wasm',module:'core',score:cached.score(distorted,width*4)};
    else score={...JSON.parse(scorer.api.score_rgba(selected,reference,distorted,width,height,width*4,data.display||'')),module:scorer.name};
    const scoreMilliseconds=performance.now()-scoreStart;
    postMessage({type:'score',id,score,scoreMilliseconds,referenceMilliseconds,referencePrepared});
    const mapStart=performance.now();
    let maps={map:null,mapWidth:0,mapHeight:0,similarity:false,geometry:null,mapName:null,mapNames:[],mapPolarity:null};
    if(mapMetric==='zensim') {
      phase(core,5);
      const compared=cached.compare(distorted,width*4);
      maps={...maps,map:compared.map,mapWidth:width,mapHeight:height,mapName:'diffmap',mapNames:['diffmap'],mapPolarity:'difference'};compared.free();
      lastMapModule=null;
    } else if(mapMetric) {
      phase(mapper,5);
      maps=mapMessage(mapper.api.spatial_map(mapMetric,data.mapName||'',reference,distorted,width,height,width*4,data.display||'standard_4k'));
      lastMapModule=mapper;
    }
    postMessage({type:'result',id,score,...maps,mapMetric,mapProfile:mapMetric==='zensim'?profile:null,referenceMilliseconds,referencePrepared,scoreMilliseconds,mapMilliseconds:performance.now()-mapStart,milliseconds:performance.now()-start},maps.map?[maps.map.buffer]:[]);
  } catch (error) {
    postMessage({type: 'error', id: data.id, message: String(error), fatal: error instanceof WebAssembly.RuntimeError});
  }
};
