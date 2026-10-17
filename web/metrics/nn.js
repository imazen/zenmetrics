// Learned metrics in the coordinator worker: onnxruntime-web (vendored,
// pinned in nn-models.json) runs each model's exported `*_full` graph; the
// full metric module builds the input tensors with zenmetrics-nn's own
// preprocessing (nn_prep.rs). Weights come from R2, are checked against
// their sha256 and kept in Cache Storage. The requested execution provider
// is the only one a session gets: no request falls back to another.
const base=new URL('./', import.meta.url);
let ort, manifest;
const sessions=new Map();
const CACHE='zenmetrics-nn-v1';
// ORT's WebAssembly runtime binds the console when it loads, so lines are
// intercepted here, before it is imported: while `capture` is set they are
// collected (ORT's profile), otherwise passed through.
let capture=null;
for(const k of ['log','info','warn']){
  const original=console[k].bind(console);
  console[k]=(...a)=>{if(capture)capture.push(a.map(String).join(' '));else original(...a);};
}

async function models(){
  if(!manifest){
    const response=await fetch(new URL('nn-models.json',base));
    if(!response.ok)throw new Error(`nn-models.json: HTTP ${response.status}`);
    manifest=await response.json();
  }
  return manifest;
}
async function runtime(){
  if(!ort){
    ort=await import(new URL('ort/ort.jspi.min.mjs',base));
    ort.env.wasm.wasmPaths=new URL('ort/',base).href;
    ort.env.logLevel='warning';
  }
  return ort;
}
// The adapter a WebGPU session will use, or an error naming why there is none.
async function adapterInfo(){
  if(!self.navigator?.gpu)throw new Error('WebGPU unavailable: navigator.gpu is missing in this browser');
  const adapter=await navigator.gpu.requestAdapter();
  if(!adapter)throw new Error('WebGPU unavailable: requestAdapter() returned no adapter');
  const info=adapter.info||{};
  return {vendor:info.vendor||'',architecture:info.architecture||'',device:info.device||'',description:info.description||'',fallback:Boolean(adapter.isFallbackAdapter??info.isFallbackAdapter)};
}
async function sha256(bytes){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');}
// Weights: Cache Storage first, else a streamed download with progress.
async function weights(graph,progress){
  const start=performance.now();
  const cache=await caches.open(CACHE).catch(()=>null);
  let response=await cache?.match(graph.url),source='cache';
  if(!response){
    source='network';
    let fetched;
    try{fetched=await fetch(graph.url,{mode:'cors'});}
    catch(error){throw new Error(`Weight download failed (${graph.url}): ${error.message}`);}
    if(!fetched.ok)throw new Error(`Weight download failed (${graph.url}): HTTP ${fetched.status}`);
    const reader=fetched.body.getReader(),chunks=[];let received=0,last=0;
    for(;;){const {done,value}=await reader.read();if(done)break;chunks.push(value);received+=value.length;
      if(performance.now()-last>250){last=performance.now();progress(`Downloading weights: ${(received/1048576).toFixed(1)} of ${(graph.bytes/1048576).toFixed(1)} MiB`);}}
    response=new Response(new Blob(chunks));
    const bytes=new Uint8Array(await response.clone().arrayBuffer());
    if(bytes.length!==graph.bytes)throw new Error(`Weight download truncated: ${bytes.length} of ${graph.bytes} bytes`);
    if(await sha256(bytes)!==graph.sha256)throw new Error('Downloaded weights do not match their sha256');
    await cache?.put(graph.url,response.clone()).catch(()=>{});
    return {bytes,source,milliseconds:performance.now()-start};
  }
  const bytes=new Uint8Array(await response.arrayBuffer());
  if(await sha256(bytes)!==graph.sha256){await cache.delete(graph.url);throw new Error('Cached weights do not match their sha256; removed, try again');}
  return {bytes,source,milliseconds:performance.now()-start};
}
// One session per metric and provider. Node placement messages from the
// session log (nodes a provider does not implement) are kept with it.
async function session(metric,graph,ep,progress,part='graph'){
  const key=`${metric}|${ep}|${part}`;
  if(sessions.has(key))return Object.assign(Object.create(sessions.get(key)),{cached:true,entry:sessions.get(key)});
  const ort=await runtime();
  const adapter=ep==='webgpu'?await adapterInfo():null;
  const w=await weights(graph,progress);
  progress(`Creating the ${ep} session for ${metric}${part==='graph'?'':` (${part})`}…`);
  const start=performance.now();
  // The first run is profiled to record which provider ran each node.
  const created=await ort.InferenceSession.create(w.bytes,{executionProviders:[ep],graphOptimizationLevel:'all',enableProfiling:true});
  const entry={session:created,adapter,weights:{source:w.source,milliseconds:w.milliseconds,bytes:graph.bytes},
    sessionMilliseconds:performance.now()-start,placement:null};
  sessions.set(key,entry);
  return Object.assign(Object.create(entry),{cached:false,entry});
}
// After a session's first run: end ORT's profiler (it prints the profile to
// the console) and count the distinct nodes each provider ran.
function placement(s){
  const entry=s.entry;
  if(entry.placement)return;
  capture=[];
  try{s.session.endProfiling();}finally{var lines=capture;capture=null;}
  const nodes={};
  for(const line of lines.join('\n').split('\n')){
    const name=/"name"\s*:\s*"([^"]+)_kernel_time"/.exec(line),provider=/"provider"\s*:\s*"([A-Za-z]+)"/.exec(line);
    if(name&&provider)(nodes[provider[1]]??=new Set()).add(name[1]);
  }
  entry.placement=Object.fromEntries(Object.entries(nodes).map(([k,v])=>[k,v.size]));
}
function tensors(ort,prepared,rename){
  const layout=JSON.parse(prepared.layout),data=prepared.data;prepared.free();
  return Object.fromEntries(layout.map(t=>[rename?.[t.name]??t.name,new ort.Tensor('float32',data.slice(t.offset,t.offset+t.length),t.shape)]));
}
// WaDIQaM's graphs over patch chunks of at most 2000 (wadiqam.rs
// PATCH_CHUNK, upstream evaluate.py's batch). `rows` per-patch outputs
// append in order; `ya` outputs split into their y and a halves.
const PATCH_CHUNK=2000;
async function chunked(ort,s,input,name,x,n,row,output,kind){
  const out=kind==='ya'?{y:new Float32Array(n),a:new Float32Array(n)}:new Float32Array(n*512);
  for(let p=0;p<n;p+=PATCH_CHUNK){
    const cn=Math.min(PATCH_CHUNK,n-p),shape=input==='f'?[cn,row]:[cn,3,32,32];
    const result=await s.session.run({[input]:new ort.Tensor('float32',x.slice(p*row,(p+cn)*row),shape)});placement(s);
    const name=s.session.outputNames.includes(output)?output:s.session.outputNames[0],t=result[name];
    const want=kind==='ya'?[2,cn]:[cn,512];
    if(JSON.stringify(t.dims)!==JSON.stringify(want))throw new Error(`${name}: want [${want}], got [${t.dims}]`);
    if(kind==='ya'){out.y.set(t.data.subarray(0,cn),p);out.a.set(t.data.subarray(cn),p);}else out.set(t.data,p*512);
    for(const v of Object.values(result))v.dispose?.();
  }
  return out;
}
// Upstream pooling, Σ(y·a)/Σa in f64 in patch order (wadiqam.rs `pool`).
function pool({y,a}){let num=0,den=0;for(let i=0;i<y.length;i++){num+=y[i]*a[i];den+=a[i];}return num/den;}
async function run(s,feeds,output){
  const result=await s.session.run(feeds);placement(s);
  const name=s.session.outputNames.includes(output)?output:s.session.outputNames[0];
  const value=result[name].data[0];
  for(const t of Object.values(result))t.dispose?.();
  return Number(value);
}
// Score `metric` on one RGBA pair. `prepare` is the full module's nn_prepare.
export async function score({metric,ep,reference,distorted,width,height,prepare,wdLayer,progress}){
  if(!['webgpu','wasm'].includes(ep))throw new Error(`Unknown execution provider ${ep}`);
  const all=await models(),model=all.models[metric];
  if(!model)throw new Error(`${metric} has no browser model`);
  const ort=await runtime();
  const prep=rgba=>prepare(model.prep,rgba,width,height,width*4,model.min_size);
  let values,runMilliseconds=0;
  if(model.kind==='wadiqam-fr'||model.kind==='wadiqam-nr')return wadiqam();
  if(model.kind==='wd')return wd();
  const s=await session(metric,model.graph,ep,progress);
  const prepStart=performance.now();
  if(model.kind==='pair'){
    const feeds={...tensors(ort,prep(reference),{x:'r'}),...tensors(ort,prep(distorted),{x:'d'})};
    const prepMilliseconds=performance.now()-prepStart,runStart=performance.now();
    values=[{column:model.columns[0],value:await run(s,feeds,model.output)}];
    runMilliseconds=performance.now()-runStart;
    return result(prepMilliseconds,s);
  }
  // No-reference models score the distorted image, then the reference under
  // `<column>_ref`, as the CLI's PairedNoReference does.
  const dFeeds=tensors(ort,prep(distorted)),rFeeds=tensors(ort,prep(reference));
  const prepMilliseconds=performance.now()-prepStart,runStart=performance.now();
  values=[{column:model.columns[0],value:await run(s,dFeeds,model.output)},{column:model.columns[1],value:await run(s,rFeeds,model.output)}];
  runMilliseconds=performance.now()-runStart;
  return result(prepMilliseconds,s);
  // Wasserstein Distortion: the VGG16 trunk's four taps for each image, then
  // the statistics head per tap in the full module (wd.rs `head`), its five
  // values added in f32 in tap order.
  async function wd(){
    const s=await session(metric,model.graph,ep,progress);
    const prepStart=performance.now();
    const rf=tensors(ort,prep(reference)),df=tensors(ort,prep(distorted));
    const prepMilliseconds=performance.now()-prepStart,runStart=performance.now();
    const taps=async feeds=>{const out=await s.session.run(feeds);placement(s);return ['tap0','tap1','tap2','tap3'].map(name=>{if(!out[name])throw new Error(`WD trunk has no output ${name}`);return out[name];});};
    const a=await taps(rf),b=await taps(df);
    const total=new Float32Array(5);
    for(let k=0;k<4;k++){
      const [n,c,h,w]=a[k].dims;
      if(n!==1||JSON.stringify(a[k].dims)!==JSON.stringify(b[k].dims))throw new Error('WD tap shapes differ');
      const layer=wdLayer(a[k].data,b[k].data,c,h,w);
      for(let i=0;i<5;i++)total[i]+=layer[i];
      a[k].dispose?.();b[k].dispose?.();
    }
    values=model.columns.map((column,i)=>({column,value:total[i]}));
    runMilliseconds=performance.now()-runStart;
    return result(prepMilliseconds,s);
  }
  async function wadiqam(){
    const fr=model.kind==='wadiqam-fr';
    const graphs=fr?{trunk:await session(metric,model.graphs.trunk,ep,progress,'trunk'),head:await session(metric,model.graphs.head,ep,progress,'head')}
      :{graph:await session(metric,model.graph,ep,progress)};
    const prepStart=performance.now();
    const patches=rgba=>{const p=prep(rgba),layout=JSON.parse(p.layout),data=p.data;p.free();return {x:data,n:layout[0].shape[0]};};
    const d=patches(distorted),r=patches(reference);
    const prepMilliseconds=performance.now()-prepStart,runStart=performance.now(),row=3*32*32;
    if(fr){
      // Trunk features of both images, the upstream concat (h − h_ref, h,
      // h_ref) per patch in f32, then the head (wadiqam.rs `score_against`).
      const hr=await chunked(ort,graphs.trunk,'x','trunk',r.x,r.n,row,'h','rows');
      const h=await chunked(ort,graphs.trunk,'x','trunk',d.x,d.n,row,'h','rows');
      const f=new Float32Array(d.n*1536);
      for(let p=0;p<d.n;p++)for(let i=0;i<512;i++){const a=h[p*512+i],b=hr[p*512+i];f[p*1536+i]=a-b;f[p*1536+512+i]=a;f[p*1536+1024+i]=b;}
      values=[{column:model.columns[0],value:pool(await chunked(ort,graphs.head,'f','head',f,d.n,1536,'ya','ya'))}];
    } else {
      values=[{column:model.columns[0],value:pool(await chunked(ort,graphs.graph,'x','graph',d.x,d.n,row,'ya','ya'))},
              {column:model.columns[1],value:pool(await chunked(ort,graphs.graph,'x','graph',r.x,r.n,row,'ya','ya'))}];
    }
    runMilliseconds=performance.now()-runStart;
    const first=Object.values(graphs)[0];
    const fresh=Object.values(graphs).filter(g=>!g.cached);
    return {score:{metric,backend:'onnxruntime-web',runtime:`${all.runtime.package} ${all.runtime.version}`,ep,adapter:first.adapter,placement:Object.fromEntries(Object.entries(graphs).map(([k,g])=>[k,g.entry.placement])),values,module:'nn'},
      timings:{weights:fresh.length?{source:fresh.map(g=>g.weights.source).join('+'),milliseconds:fresh.reduce((s,g)=>s+g.weights.milliseconds,0),bytes:fresh.reduce((s,g)=>s+g.weights.bytes,0)}:null,
        session:fresh.length?fresh.reduce((s,g)=>s+g.sessionMilliseconds,0):null,prepare:prepMilliseconds,run:runMilliseconds}};
  }
  function result(prepMilliseconds,s){
    return {score:{metric,backend:'onnxruntime-web',runtime:`${all.runtime.package} ${all.runtime.version}`,ep,adapter:s.adapter,placement:{graph:s.entry.placement},values,module:'nn'},
      timings:{weights:s.cached?null:s.weights,session:s.cached?null:s.sessionMilliseconds,prepare:prepMilliseconds,run:runMilliseconds}};
  }
}
