import init, { initThreadPool, Reference, thread_count, available_metrics, available_spatial_maps, score_rgba, decode_image, encode_jpeg, spatial_map, cached_spatial_map, loop_map, loop_map_defaults, JobControl } from './pkg/zenmetrics_browser.js';
let ready = false, cached, previousReference, wasm, control;
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
      postMessage({type: 'ready', threads: thread_count(), metrics: JSON.parse(available_metrics()), maps:JSON.parse(available_spatial_maps()), loopDefaults:JSON.parse(loop_map_defaults()), startup:{wasmMilliseconds,poolMilliseconds}});
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
      const {width,height,quality}=data,name=data.codec||'jpeg';
      phase(2);
      if(name==='jpeg') {
        const encoded=encode_jpeg(data.reference,data.width,data.height,data.width*4,quality);
        phase(1);const image=unpack(decode_image(encoded,'generated.jpg'));
        postMessage({type:'decoded',id:data.id,bytes:encoded.length,encoded,...image},[image.rgba.buffer,encoded.buffer]);return;
      }
      postMessage({type:'progress',message:`Loading the ${name} encoder module…`});
      const module=await codec(name);
      const coded=name==='avif'?module.roundtrip(data.reference,width,height,quality,data.speed||6)
        :name==='jxl'?module.roundtrip(data.reference,width,height,quality,data.effort||7)
        :module.roundtrip(data.reference,width,height,quality);
      const encoded=coded.file,rgba=coded.rgba;coded.free();
      postMessage({type:'decoded',id:data.id,bytes:encoded.length,encoded,width,height,rgba},[rgba.buffer,encoded.buffer]);return;
    }
    if(data.type==='loop') {
      // jxl-encoder's perceptual loop, scored by its own butteraugli or by a
      // viewer metric's map put on butteraugli's scale (loop_map).
      const {reference,width,height,steer,id}=data;
      postMessage({type:'loop-progress',id,message:'Loading the JPEG XL encoder module…'});
      const module=await codec('jxl');
      postMessage({type:'loop-progress',id,message:`Encoding ${width}×${height}, steered by ${data.steering}…`});
      if(steer&&!(steer.scale>0&&Number.isFinite(steer.scale)&&steer.exponent>0&&Number.isFinite(steer.exponent)))
        throw new Error('scale and exponent must be finite and positive');
      // A failed scorer ends the loop's seed early inside the encoder; report
      // it instead of returning that encode as if it were steered.
      let scored=0,failure=null;
      const scorer=steer?recon=>{
        try{
          const m=loop_map(steer.metric,steer.map,reference,recon,width,height,width*4,steer.display||'standard_4k',steer.scale,steer.exponent);
          const out={values:m.values,score:m.score};m.free();
          postMessage({type:'loop-progress',id,scored:++scored,score:out.score});
          return out;
        }catch(error){failure??=String(error);throw error;}
      }:undefined;
      const start=performance.now();
      const run=module.loop_encode(reference,width,height,data.distance,data.effort,data.iters,scorer,data.steering);
      if(failure){run.free();throw new Error(`The steering metric failed: ${failure}`);}
      const trace=run.trace,file=run.file,rgba=run.rgba;run.free();
      postMessage({type:'loop',id,trace,file,rgba,milliseconds:performance.now()-start},[trace.buffer,file.buffer,rgba.buffer]);return;
    }
    if(data.type==='map') {
      // Another map of the last evaluation, without evaluating again.
      phase(5);const mapStart=performance.now();
      postMessage({type:'result',id:data.id,...mapMessage(cached_spatial_map(data.mapName)),mapMetric:data.mapMetric,mapMilliseconds:performance.now()-mapStart},[]);return;
    }
    const profile = data.zensimProfile || 'b';
    const start = performance.now();
    const {reference, distorted, width, height, metric, id} = data;
    const mapMetric = data.mapMetric === undefined ? 'zensim' : data.mapMetric;

    let referencePrepared=false;
    if ((metric==='zensim'||!metric||mapMetric==='zensim') && (!previousReference || previousReference.profile !== profile || previousReference.width !== width || previousReference.height !== height ||
        previousReference.bytes.length !== reference.length || !reference.every((v,i) => v === previousReference.bytes[i]))) {
      referencePrepared=true;phase(3);const prepared = Reference.with_profile(reference, width, height, width * 4, profile);
      if (cached) cached.free();
      cached = prepared;
      previousReference = {width, height, profile, bytes: reference.slice()};
    }
    const referenceMilliseconds=performance.now()-start;
    const selected = metric || 'zensim';
    let score;const scoreStart=performance.now();phase(4);
    if(selected==='zensim') score={metric:'zensim',profile,backend:'native-cpu-wasm',score:cached.score(distorted,width*4)};
    else score=JSON.parse(score_rgba(selected,reference,distorted,width,height,width*4,data.display||''));
    const scoreMilliseconds=performance.now()-scoreStart;
    postMessage({type:'score',id,score,scoreMilliseconds,referenceMilliseconds,referencePrepared});
    const mapStart=performance.now();phase(5);
    let maps={map:null,mapWidth:0,mapHeight:0,similarity:false,geometry:null,mapName:null,mapNames:[],mapPolarity:null};
    if(mapMetric==='zensim') {
      const compared=cached.compare(distorted,width*4);
      maps={...maps,map:compared.map,mapWidth:width,mapHeight:height,mapName:'diffmap',mapNames:['diffmap'],mapPolarity:'difference'};compared.free();
    } else if(mapMetric) {
      maps=mapMessage(spatial_map(mapMetric,data.mapName||'',reference,distorted,width,height,width*4,data.display||'standard_4k'));
    }
    postMessage({type:'result',id,score,...maps,mapMetric,mapProfile:mapMetric==='zensim'?profile:null,referenceMilliseconds,referencePrepared,scoreMilliseconds,mapMilliseconds:performance.now()-mapStart,milliseconds:performance.now()-start},maps.map?[maps.map.buffer]:[]);
  } catch (error) {
    postMessage({type: 'error', id: data.id, message: String(error), fatal: error instanceof WebAssembly.RuntimeError});
  }
};
