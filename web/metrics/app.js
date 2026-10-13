const $ = id => document.getElementById(id);
let worker;
const S = {reference:null, distorted:null, map:null, mode:'overlay', zoom:1, ready:false, pending:false, dirty:false, id:0, revision:0, blink:false, loads:{reference:0,distorted:0}};
const requests = new Map();
function status(message, error=false) { $('status').textContent=message; $('status').className=error?'error':'status'; }
const phases={1:'Decoding',2:'Encoding JPEG',3:'Preparing reference',4:'Scoring',5:'Building spatial map'};
function showScore(result) {
  const values=result.score.values||[{column:result.score.metric,value:result.score.score}];
  $('score').textContent=values.map(v=>typeof v.value==='number'?v.value.toPrecision(7):v.value).join(' / ');
  $('score-label').textContent=result.score.metric;
  $('score-info').textContent=values.map(v=>v.column).join(' / ');
  $('elapsed').textContent=`${result.scoreMilliseconds.toFixed(0)} ms scoring`;
}
function finishRequest(id,error,result) {
  const request=requests.get(id);if(!request)return;
  clearTimeout(request.timer);requests.delete(id);
  if(error||request.cancelled)request.reject(error||new Error('Cancelled'));
  else request.resolve(result);
  $('cancel').disabled=!requests.size;
  if(!requests.size)$('job-progress').hidden=true;
}
function restartWorker(message) {
  worker?.terminate();S.ready=false;$('compare').disabled=true;
  for(const id of [...requests.keys()])finishRequest(id,new Error(message));
  startWorker();
}
function rpc(data) {
  return new Promise((resolve,reject)=>{
    const id=++S.id;
    const request={resolve,reject,revision:S.revision,start:performance.now(),kind:data.type};
    request.timer=setTimeout(()=>restartWorker('Job exceeded the 120 second browser deadline.'),120000);
    requests.set(id,request);$('cancel').disabled=false;$('job-progress').hidden=false;
    worker.postMessage({...data,id});
  });
}
function startWorker() {
  worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
  worker.onmessage=({data})=>{
    if(data.type==='progress'){status(data.message);return;}
    if(data.type==='ready'){
      S.ready=true;const selected=$('metric').value;
      $('metric').replaceChildren(...data.metrics.map(name=>{const o=document.createElement('option');o.value=name;o.textContent=name;return o;}));
      $('metric').value=data.metrics.includes(selected)?selected:'zensim';snippet();$('compare').disabled=false;
      status(`Ready: ${data.threads} Rayon workers, SIMD128 + relaxed SIMD`);return;
    }
    const request=requests.get(data.id);if(!request){if(data.type==='error'){S.ready=false;$('compare').disabled=true;status(data.message,true);}return;}
    if(data.type==='control'){request.control=new Int32Array(data.buffer,data.offset,4);if(request.cancelled)Atomics.store(request.control,0,1);return;}
    if(data.type==='score'){if(!request.cancelled&&request.revision===S.revision)showScore(data);return;}
    finishRequest(data.id,data.type==='error'?new Error(request.cancelled?'Cancelled':data.message):null,data);
    if(data.fatal)restartWorker('WASM runtime failed; starting a fresh worker.');
  };
  worker.onerror=event=>{worker.terminate();S.ready=false;$('compare').disabled=true;for(const id of [...requests.keys()])finishRequest(id,new Error(event.message));status(`Worker failed: ${event.message}`,true);};
  worker.postMessage({type:'init',threads:Math.max(1,Math.min(8,navigator.hardwareConcurrency||1))});
}
setInterval(()=>{
  const request=[...requests.values()].find(r=>r.control&&!r.cancelled);if(!request)return;
  const phase=Atomics.load(request.control,1),completed=Atomics.load(request.control,2);
  $('progress-label').textContent=`${phases[phase]||'Starting job'} · ${(performance.now()-request.start).toFixed(0)} ms${completed?` · ${completed} input rows prepared`:''}`;
  // Kernel totals are unknown. Elapsed time and cancellation polls are not progress units.
  $('progress-bar').removeAttribute('value');
},80);
$('cancel').onclick=()=>{
  S.dirty=false;++S.revision;
  for(const request of requests.values()){
    request.cancelled=true;if(request.control)Atomics.store(request.control,0,1);
  }
  status('Cancelling…');
  // Some native metric adapters have no Stop API; dispose of their worker too.
  const interrupted=worker;
  setTimeout(()=>{if(worker===interrupted)restartWorker('Cancelled');},250);
};
async function loadFile(side,file) {
  if(!file)return;
  if(!S.ready){status('Wait for the metric worker to finish starting.',true);return;}
  S.encoded=null;$('download-jpeg').disabled=true;
  const load=++S.loads[side];++S.revision;S.map=null;S[side]=null;status(`Decoding ${file.name}…`);
  try {
    const bytes=new Uint8Array(await file.arrayBuffer());
    const result=await rpc({type:'decode',bytes,filename:file.name});
    if(load!==S.loads[side])return;
    ++S.revision;
    S[side]={width:result.width,height:result.height,data:new Uint8ClampedArray(result.rgba)};
    $(`name-${side}`).textContent=`${file.name} · ${result.width}×${result.height} · ${(file.size/1024).toFixed(1)} KiB`;
    status(`${file.name} decoded with Zen codecs.`);fit();paint();schedule();
  } catch(error){status(`Could not decode ${file.name}: ${error.message}`,true);}
}
for(const side of ['reference','distorted']) {
  $(side).onchange=()=>loadFile(side,$(side).files[0]);
  const drop=$(`drop-${side}`);
  drop.ondragover=event=>{event.preventDefault();drop.classList.add('drag');};
  drop.ondragleave=()=>drop.classList.remove('drag');
  drop.ondrop=event=>{event.preventDefault();drop.classList.remove('drag');loadFile(side,event.dataTransfer.files[0]);};
}
function schedule(){S.dirty=true;if($('live').checked)compare();}
async function compare() {
  if(!S.ready||S.pending||!S.reference||!S.distorted)return;
  const r=S.reference,d=S.distorted;
  if(r.width!==d.width||r.height!==d.height){status(`Sizes differ: ${r.width}×${r.height} vs ${d.width}×${d.height}. Images are never resized for scoring.`,true);return;}
  const revision=S.revision;S.pending=true;S.dirty=false;$('compare').disabled=true;$('score').textContent='–';$('score-info').textContent='';status(`Computing ${$('metric').value}…`);
  try {
    const result=await rpc({type:'compare',metric:$('metric').value,mapMetric:$('map-metric').value,reference:r.data,distorted:d.data,width:r.width,height:r.height,display:$('display').value});
    if(revision!==S.revision)return;
    S.map=result.map;S.mapMetric=result.mapMetric;S.mapWidth=result.mapWidth;S.mapHeight=result.mapHeight;S.similarity=result.similarity;
    showScore(result);
    $('elapsed').textContent=`${result.milliseconds.toFixed(0)} ms total`;
    status(`${r.width}×${r.height} · ${result.score.backend} · ${result.mapMetric} map`);paint();
  } catch(error){S.map=null;status(error.message,true);paint();}
  finally{S.pending=false;$('compare').disabled=!S.ready;if(S.dirty&&$('live').checked)compare();}
}
$('compare').onclick=compare;
$('metric').onchange=()=>{$('display-row').hidden=$('metric').value!=='cvvdp'&&$('map-metric').value!=='cvvdp';snippet();S.revision++;schedule();};
$('display').onchange=()=>{snippet();S.revision++;schedule();};
$('live').onchange=()=>{if($('live').checked)compare();};
$('map-metric').onchange=()=>{$('display-row').hidden=$('metric').value!=='cvvdp'&&$('map-metric').value!=='cvvdp';S.map=null;S.revision++;schedule();};
$('swap').onclick=()=>{[S.reference,S.distorted]=[S.distorted,S.reference];const a=$('name-reference'),b=$('name-distorted');[a.textContent,b.textContent]=[b.textContent,a.textContent];S.map=null;S.revision++;paint();schedule();};
$('quality').oninput=()=>{$('quality-value').textContent=$('quality').value;};
async function generate() {
  if(!S.ready||!S.reference)return;
  const revision=++S.revision;S.map=null;status('Encoding JPEG with zenjpeg…');
  try{
    const r=S.reference;const result=await rpc({type:'generate',reference:r.data,width:r.width,height:r.height,quality:Number($('quality').value)});
    if(revision!==S.revision)return;
    S.encoded=result.encoded;$('download-jpeg').disabled=false;
    S.distorted={width:result.width,height:result.height,data:new Uint8ClampedArray(result.rgba)};
    $('name-distorted').textContent=`zenjpeg · quality ${$('quality').value} · ${(result.bytes/1024).toFixed(1)} KiB`;
    paint();schedule();
  }catch(error){status(error.message,true);}
}
$('generate').onclick=generate;
$('demo').onclick=async()=>{
  const canvas=document.createElement('canvas');canvas.width=768;canvas.height=512;const g=canvas.getContext('2d');
  const sky=g.createLinearGradient(0,0,0,320);sky.addColorStop(0,'#7fa7d6');sky.addColorStop(1,'#e9d9bf');g.fillStyle=sky;g.fillRect(0,0,768,512);
  g.fillStyle='#3f5d3a';g.beginPath();g.moveTo(0,320);for(let x=0;x<=768;x+=8)g.lineTo(x,320-38*Math.sin(x/70)-18*Math.sin(x/23));g.lineTo(768,512);g.lineTo(0,512);g.fill();
  for(let y=340;y<512;y+=3){g.strokeStyle=`rgba(30,40,20,${.25+.2*Math.sin(y)})`;g.beginPath();g.moveTo(0,y);for(let x=0;x<=768;x+=12)g.lineTo(x,y+1.5*Math.sin(x/9+y));g.stroke();}
  g.fillStyle='#f4efe6';g.fillRect(60,70,255,100);g.fillStyle='#222';g.font='23px sans-serif';g.fillText('Ringing near edges',75,113);g.font='16px serif';g.fillText('Fine detail and flat areas',75,147);
  g.fillStyle='#c8452b';g.beginPath();g.arc(550,125,50,0,Math.PI*2);g.fill();
  for(let i=0;i<6;i++){g.strokeStyle='#fff';g.lineWidth=2;g.beginPath();g.moveTo(425+i*9,40);g.lineTo(476+i*9,230);g.stroke();}
  S.reference={width:768,height:512,data:g.getImageData(0,0,768,512).data};$('name-reference').textContent='Demo scene · 768×512';fit();paint();await generate();
};
const palettes={viridis:[[0,[68,1,84]],[.25,[59,82,139]],[.5,[33,145,140]],[.75,[94,201,98]],[1,[253,231,37]]],turbo:[[0,[48,18,59]],[.2,[62,155,254]],[.4,[70,248,132]],[.6,[225,220,55]],[.8,[240,92,18]],[1,[122,4,3]]]};
function errorValue(v){return S.similarity?Math.max(0,1-v):Math.max(0,v);}
function mapIndex(x,y){return Math.min(S.mapHeight-1,Math.floor(y*S.mapHeight/S.reference.height))*S.mapWidth+Math.min(S.mapWidth-1,Math.floor(x*S.mapWidth/S.reference.width));}
const ramp=new Uint8ClampedArray(768);
function updatePalette(){const stops=palettes[$('palette').value];
for(let i=0;i<256;i++){let k=1;const x=i/255;while(k<stops.length-1&&x>stops[k][0])k++;const [x0,a]=stops[k-1],[x1,b]=stops[k];for(let c=0;c<3;c++)ramp[i*3+c]=a[c]+(b[c]-a[c])*(x-x0)/(x1-x0);}
$('palette').closest('section').querySelector('.ramp').style.background=`linear-gradient(90deg,${stops.map(([_,c])=>`rgb(${c.join(',')})`).join(',')})`;}

function mapTop(){if(!S.map)return 1;if($('norm').value==='fixed')return Number($('scale').value);if($('norm').value==='max'){let top=0;for(const v of S.map)top=Math.max(top,errorValue(v));return top;}
  const sampled=[],step=Math.max(1,Math.floor(S.map.length/200000));for(let i=0;i<S.map.length;i+=step)if(errorValue(S.map[i])>0)sampled.push(errorValue(S.map[i]));sampled.sort((a,b)=>a-b);return sampled.length?sampled[Math.min(sampled.length-1,Math.floor(sampled.length*.995))]:0;}
function paint(){
  const r=S.reference;if(!r)return;const d=S.distorted,canvas=$('view');canvas.width=r.width;canvas.height=r.height;
  const compatible=d&&d.width===r.width&&d.height===r.height,background=$('background').value==='distorted'&&compatible?d:r;
  const output=new ImageData(new Uint8ClampedArray(r.data),r.width,r.height),alpha=Number($('opacity').value),top=mapTop();
  if(!(top>=0&&Number.isFinite(top))){status('Map maximum must be a finite positive value.',true);return;}
  for(let i=0,j=0;i<r.width*r.height;i++,j+=4){
    const x=i%r.width;let source=r;
    if(compatible&&(S.mode==='distorted'||(S.mode==='split'&&x>=r.width*Number($('split').value))||(S.mode==='blink'&&S.blink)))source=d;
    if(S.mode==='overlay')source=background;
    for(let c=0;c<3;c++)output.data[j+c]=source.data[j+c];
    if((S.mode==='overlay'||S.mode==='map')&&S.map){
      let t=top>0?Math.min(1,Math.max(0,errorValue(S.map[mapIndex(x,Math.floor(i/r.width))])/top)):0;if($('norm').value==='sqrt')t=Math.sqrt(t);const k=Math.round(t*255);
      const weight=S.mode==='map'?1:alpha*Math.sqrt(t),grey=.2126*source.data[j]+.7152*source.data[j+1]+.0722*source.data[j+2];
      for(let c=0;c<3;c++)output.data[j+c]=($('grey').checked?grey:source.data[j+c])*(1-weight)+ramp[k*3+c]*weight;
    }
    output.data[j+3]=255;
  }
  canvas.getContext('2d').putImageData(output,0,0);canvas.style.width=`${r.width*S.zoom}px`;canvas.style.height=`${r.height*S.zoom}px`;
  $('zoom').textContent=`${Math.round(S.zoom*100)}%`;$('dimensions').textContent=`${r.width}×${r.height}`;
  $('map-info').textContent=S.map?`${S.mapMetric} · scale ${top.toPrecision(4)}`:'No map yet';
}
function fit(){if(S.reference)S.zoom=Math.min(1,Math.max(.05,($('viewport').clientWidth||768)/S.reference.width));paint();}
for(const id of ['opacity','scale','norm','background','grey','split'])$(id).oninput=paint;
$('palette').onchange=()=>{updatePalette();paint();};updatePalette();
$('views').onclick=event=>{if(!event.target.dataset.mode)return;S.mode=event.target.dataset.mode;for(const b of $('views').children)b.classList.toggle('on',b===event.target);paint();};
setInterval(()=>{if(S.mode==='blink'){S.blink=!S.blink;paint();}},500);
$('fit').onclick=fit;$('actual').onclick=()=>{S.zoom=1;paint();};
$('zoom-in').onclick=()=>{S.zoom=Math.min(8,S.zoom*1.25);paint();};$('zoom-out').onclick=()=>{S.zoom=Math.max(.05,S.zoom/1.25);paint();};
$('view').onpointermove=event=>{if(!S.reference)return;const box=$('view').getBoundingClientRect(),x=Math.floor((event.clientX-box.left)/S.zoom),y=Math.floor((event.clientY-box.top)/S.zoom),i=y*S.reference.width+x;
  if(x<0||y<0||x>=S.reference.width||y>=S.reference.height)return;const v=S.map?.[mapIndex(x,y)];$('pixel').textContent=`${x}, ${y} · RGB ${Array.from(S.reference.data.subarray(i*4,i*4+3)).join(', ')}${v===undefined?'':` · ${S.mapMetric} ${v.toPrecision(6)}`}`;};
let drag;
$('viewport').onpointerdown=event=>{drag={x:event.clientX,y:event.clientY,left:$('viewport').scrollLeft,top:$('viewport').scrollTop};$('viewport').setPointerCapture(event.pointerId);};
$('viewport').addEventListener('pointermove',event=>{if(drag){$('viewport').scrollLeft=drag.left+drag.x-event.clientX;$('viewport').scrollTop=drag.top+drag.y-event.clientY;}});
$('viewport').onpointerup=()=>{drag=null;};$('viewport').onpointercancel=()=>{drag=null;};
$('download').onclick=()=>{if(!S.map)return;const blob=new Blob([S.map],{type:'application/octet-stream'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`${S.mapMetric}-${S.mapWidth}x${S.mapHeight}-f32le.bin`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('theme').onclick=()=>{document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark';};
if(!crossOriginIsolated||typeof SharedArrayBuffer==='undefined')status('Threaded WASM requires HTTPS with COOP: same-origin and COEP: require-corp.',true);
else startWorker();

function snippet(){const metric=$('metric').value;
  $('cli-snippet').textContent=`zenmetrics${metric==='zensim'?' --zensim-profile b':''} score --metric ${metric} --reference reference.png --distorted distorted.jpg${metric==='cvvdp'?` --display-model ${$('display').value}`:''}`;
}
$('copy-cli').onclick=async()=>{try{await navigator.clipboard.writeText($('cli-snippet').textContent);status('CLI command copied.');}catch(error){status(error.message,true);}};
$('download-jpeg').onclick=()=>{if(!S.encoded)return;const url=URL.createObjectURL(new Blob([S.encoded],{type:'image/jpeg'})),a=document.createElement('a');a.href=url;a.download='distorted.jpg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};

// A real WebGPU compute/readback probe checks the browser backend used by wgpu.
async function checkGpu() {
  let device;
  try {
    if(!navigator.gpu)throw new Error('WebGPU API unavailable');
    const adapter=await navigator.gpu.requestAdapter();
    if(!adapter)throw new Error('no WebGPU adapter');
    device=await adapter.requestDevice();
    const shader=device.createShaderModule({code:'@group(0) @binding(0) var<storage, read_write> output: array<u32>; @compute @workgroup_size(1) fn main() { output[0] = 42u; }'});
    const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:shader,entryPoint:'main'}});
    const buffer=device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const readback=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const bindings=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer}}]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,bindings);pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(buffer,0,readback,0,4);
    device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
    const value=new Uint32Array(readback.getMappedRange())[0];readback.unmap();
    if(value!==42)throw new Error('compute/readback failed');
    return 'WebGPU compute ready (wgpu browser backend)';
  } catch(error) { return `WebGPU unavailable: ${error.message}`; }
  finally {device?.destroy();}
}
checkGpu().then(gpu=>{$('capabilities').textContent=`${crossOriginIsolated?'Shared memory isolation ready':'Shared memory isolation unavailable'} · ${gpu}`;});
