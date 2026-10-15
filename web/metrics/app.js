const $ = id => document.getElementById(id);
let worker;
const S = {reference:null, distorted:null, map:null, mode:'overlay', zoom:1, ready:false, pending:false, dirty:false, id:0, revision:0, blink:false, loads:{reference:0,distorted:0},
  catalog:[], metrics:[], mapName:'', mapNames:[], loopDefaults:[], defaultMaps:{}, generated:false, codec:'jpeg', steerBuiltin:false};
// The default map of each metric: its fused map where it publishes one.
const mapLabels={zensim:'Zensim error',butteraugli:'Butteraugli error',dssim:'DSSIM fused error (all scales)',cvvdp:'CVVDP masked error',gmsd:'GMSD similarity',psnr:'PSNR squared error',fsim:'FSIM luminance similarity',fsimc:'FSIMc luminance similarity',haarpsi:'HaarPSI similarity',vsi:'VSI saliency-weighted contribution',msssim:'MS-SSIM fused −ln error (all levels)',nlpd:'NLPD fused difference (all levels)',psnrhvs:'PSNR-HVS 8×8 CSF-weighted block error',mdctpsnr:'mDCT-PSNR window detection probability',mad:'MAD high-quality strategy (masked local MSE)',iwssim:'IW-SSIM fused −ln error (all scales)',vif:'VIF fused information loss (all scales)',vifvec:'VIF vector fused information loss (all subbands)','vmaf-v1':'VMAF v1 SPEED chroma-U block difference',ssim2:'SSIMULACRA2 weighted error (fused terms)'};
const codecInfo={jpeg:{label:'zenjpeg',ext:'jpg',mime:'image/jpeg'},webp:{label:'zenwebp',ext:'webp',mime:'image/webp'},avif:{label:'zenavif',ext:'avif',mime:'image/avif'},jxl:{label:'jxl-encoder',ext:'jxl',mime:'image/jxl'}};
const requests = new Map();
function status(message, error=false) { $('status').textContent=message; $('status').className=error?'error':'status'; }
const phases={1:'Decoding',2:'Encoding',3:'Preparing reference',4:'Scoring',5:'Building spatial map'};
// The map route of a scoring metric: the metric itself when this build
// exports its spatial map. The map always follows the selected metric.
function mapRoute(metric){return S.catalog.includes(metric)?metric:null;}
function showScore(result) {
  const values=result.score.values||[{column:result.score.metric,value:result.score.score}];
  $('score').textContent=values.map(v=>typeof v.value==='number'?v.value.toPrecision(7):v.value).join(' / ');
  $('score-label').textContent=result.score.metric;
  $('bar-label').textContent=result.score.metric;$('bar-score').textContent=$('score').textContent;
  $('score-info').textContent=values.map(v=>v.column).join(' / ')+(result.score.profile?` · model ${result.score.profile}`:'');
  $('elapsed').textContent=`${result.scoreMilliseconds.toFixed(1)} ms scoring`;
  $('compute-timing').textContent=`Reference ${result.referenceMilliseconds.toFixed(1)} ms (${result.referencePrepared?'prepared':'cache check'}) · Score ${result.scoreMilliseconds.toFixed(1)} ms · Map pending`;
}
function finishRequest(id,error,result) {
  const request=requests.get(id);if(!request)return;
  if(result)result.requestMilliseconds=performance.now()-request.start;
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
function rpc(data, deadline=120) {
  return new Promise((resolve,reject)=>{
    const id=++S.id;
    const request={resolve,reject,revision:S.revision,start:performance.now(),kind:data.type};
    request.timer=setTimeout(()=>restartWorker(`Job exceeded the ${deadline} second browser deadline.`),deadline*1000);
    requests.set(id,request);$('cancel').disabled=false;$('job-progress').hidden=false;
    worker.postMessage({...data,id});
  });
}
function startWorker() {
  const startupStart=performance.now();
  $('startup-timing').textContent='Worker startup: starting…';
  worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
  worker.onmessage=({data})=>{
    if(data.type==='progress'){status(data.message);return;}
    if(data.type==='loop-frame'){paintLive(data);return;}
    if(data.type==='loop-progress'){if(L.running&&L.runEpoch===L.epoch)$('loop-info').textContent=data.message||`Encoding: scored ${data.scored} reconstructions (last ${data.score.toFixed(4)})…`;return;}
    if(data.type==='ready'){
      const startupMilliseconds=performance.now()-startupStart;
      $('startup-timing').textContent=`Worker startup ${startupMilliseconds.toFixed(1)} ms · WASM load/init ${data.startup.wasmMilliseconds.toFixed(1)} ms · Rayon pool ${data.startup.poolMilliseconds.toFixed(1)} ms`;
      S.ready=true;S.catalog=data.maps;S.metrics=data.metrics;S.loopDefaults=data.loopDefaults;const selected=$('metric').value;
      $('metric').replaceChildren(...data.metrics.map(name=>{const o=document.createElement('option');o.value=name;o.textContent=S.catalog.includes(name)?name:`${name} (score only)`;return o;}));
      $('metric').value=data.metrics.includes(selected)?selected:'zensim';syncMetric();$('compare').disabled=false;
      status(`Ready: ${data.threads} Rayon workers, SIMD128 + relaxed SIMD`);
      if(!S.reference)demoPair();
      return;
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
  // Some native metric adapters and the encoders have no Stop API; dispose of their worker too.
  const interrupted=worker;
  setTimeout(()=>{if(worker===interrupted)restartWorker('Cancelled');},250);
};
async function loadFile(side,file) {
  if(!file)return;
  if(!S.ready){status('Wait for the metric worker to finish starting.',true);return;}
  if(side==='distorted'){S.encoded=null;S.generated=false;$('download-jpeg').disabled=true;}
  const load=++S.loads[side];++S.revision;S.map=null;S[side]=null;status(`Decoding ${file.name}…`);
  try {
    const bytes=new Uint8Array(await file.arrayBuffer());
    const result=await rpc({type:'decode',bytes,filename:file.name});
    if(load!==S.loads[side])return;
    ++S.revision;
    S[side]={width:result.width,height:result.height,data:new Uint8ClampedArray(result.rgba)};
    $(`name-${side}`).textContent=`${file.name} · ${result.width}×${result.height} · ${(file.size/1024).toFixed(1)} KiB`;
    status(`${file.name} decoded with Zen codecs.`);fit();paint();
    if(side==='reference'){if(S.generated)await generate();else schedule();scheduleLoop();}else schedule();
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
function mapUnavailable(metric){return `${metric} has no spatial map in this build; the score is computed without one.`;}
function setMaps(result){
  S.map=result.map;S.mapMetric=result.mapMetric;S.mapProfile=result.mapProfile;S.mapWidth=result.mapWidth;S.mapHeight=result.mapHeight;S.similarity=result.similarity;S.geometry=result.geometry;
  S.mapName=result.mapName||'';S.mapNames=result.mapNames||[];S.mapPolarity=result.mapPolarity;S.pendingMap=undefined;
  if(result.defaultMap)S.defaultMaps[result.mapMetric]=result.defaultMap;
  buildMenus();syncLoopSteer();
}
async function compare() {
  if(!S.ready||S.pending||!S.reference||!S.distorted)return;
  const r=S.reference,d=S.distorted;
  if(r.width!==d.width||r.height!==d.height){status(`Sizes differ: ${r.width}×${r.height} vs ${d.width}×${d.height}. Images are never resized for scoring.`,true);return;}
  const metric=$('metric').value,route=mapRoute(metric);
  const revision=S.revision;S.pending=true;S.dirty=false;$('compare').disabled=true;$('score').textContent='–';$('score-info').textContent='';$('compute-timing').textContent='Computing…';status(`Computing ${metric}…`);
  try {
    const request={type:'compare',metric,mapMetric:route,mapName:route===S.mapMetric?S.mapName:(S.requestedMap||''),zensimProfile:$('zensim-model').value,reference:r.data,distorted:d.data,width:r.width,height:r.height,display:$('display').value};
    const result=await rpc(request);
    if(revision!==S.revision)return;
    if(!request.mapName)result.defaultMap=result.mapName;
    S.requestedMap=undefined;
    setMaps(result);
    showScore(result);
    $('elapsed').textContent=`${result.milliseconds.toFixed(1)} ms total`;
    $('compute-timing').textContent=`Reference ${result.referenceMilliseconds.toFixed(1)} ms (${result.referencePrepared?'prepared':'cache check'}) · Score ${result.scoreMilliseconds.toFixed(1)} ms · Map ${route?`${result.mapMilliseconds.toFixed(1)} ms`:'none for this metric'} · Request wall ${result.requestMilliseconds.toFixed(1)} ms`;
    status(route?`${r.width}×${r.height} · ${result.score.backend} · ${result.mapMetric} map`:`${r.width}×${r.height} · ${result.score.backend} · ${mapUnavailable(metric)}`);paint();
  } catch(error){
    S.map=null;
    const cancelled=error.message==='Cancelled';
    $('compute-timing').textContent=$('compute-timing').textContent==='Computing…'?(cancelled?'Comparison cancelled':'Comparison failed'):$('compute-timing').textContent.replace('Map pending',cancelled?'Map cancelled':'Map unavailable');
    status(error.message,true);paint();
  }
  finally{S.pending=false;$('compare').disabled=!S.ready;if(S.dirty&&$('live').checked)compare();}
}
$('compare').onclick=compare;
// One metric choice drives the score, the map and the loop's steering.
function syncMetric(){
  const metric=$('metric').value;
  $('display-row').hidden=metric!=='cvvdp';
  buildMenus();snippet();syncLoopSteer(true);
}
// The Map menu (top bar) and the loop's Steer menu both list every metric's
// maps, grouped by metric: the evaluated metric's own list, else the maps
// with fitted loop defaults (the default map first). Choosing another
// metric's map switches the metric.
function mapChoices(){
  return S.catalog.filter(m=>S.metrics.includes(m)).map(metric=>{
    let maps=metric===S.mapMetric&&S.mapNames.length?S.mapNames:S.loopDefaults.filter(d=>d.metric===metric).map(d=>d.map);
    return {metric,maps:maps.length?maps:['']};
  });
}
function defaultMapOf(metric){return S.defaultMaps[metric]??S.loopDefaults.find(d=>d.metric===metric)?.map;}
function fillMapMenu(select,first){
  select.replaceChildren(...first,...mapChoices().map(({metric,maps})=>{
    const group=document.createElement('optgroup');group.label=metric;
    for(const map of maps){const o=new Option(!map?`${metric} · default map`:map===defaultMapOf(metric)?`${metric} · ${map} (default)`:`${metric} · ${map}`,`${metric}|${map}`);
      if(map===defaultMapOf(metric)&&mapLabels[metric])o.title=mapLabels[metric];group.append(o);}
    return group;
  }));
}
function selectedMapValue(){
  const route=mapRoute($('metric').value);if(!route)return '';
  const map=route===S.mapMetric?S.pendingMap??S.mapName:(S.requestedMap||defaultMapOf(route)||'');
  return `${route}|${map}`;
}
function buildMenus(){
  const metric=$('metric').value,route=mapRoute(metric),value=selectedMapValue();
  fillMapMenu($('map-name'),route?[]:[new Option(`${metric} has no spatial map`,'')]);
  $('map-name').value=value;
  if(route&&$('map-name').value!==value)$('map-name').value=[...$('map-name').options].find(o=>o.value.startsWith(route+'|'))?.value??'';
  fillMapMenu($('loop-steer'),[new Option("jxl-encoder's own butteraugli",'builtin')]);
  $('loop-steer').value=S.steerBuiltin||!route?'builtin':$('map-name').value;
}
function chooseMap(value){
  const [metric,map]=value.split('|');if(!metric)return;
  if(metric!==S.mapMetric||!S.mapNames.includes(map)){
    // Another metric (or a map not evaluated yet): switch and evaluate.
    $('metric').value=metric;S.map=null;S.mapMetric=null;S.mapName='';S.requestedMap=map;
    syncMetric();S.revision++;paint();schedule();return;
  }
  // Another map of the evaluated metric: the menus show it while it loads.
  S.pendingMap=map;showCachedMap(map);
}
// Choosing a metric or map in the top bar steers the loop with it and
// re-encodes; the loop panel can still pick the encoder's own butteraugli.
function chooseSteering(){
  S.steerBuiltin=false;if(mapRoute($('metric').value))$('loop-steer').value=$('map-name').value;
  syncLoopSteer();scheduleLoop();
}
$('metric').onchange=()=>{S.map=null;S.mapName='';S.mapMetric=null;S.requestedMap=undefined;syncMetric();S.revision++;paint();schedule();chooseSteering();};
$('zensim-model').onchange=()=>{snippet();S.map=null;S.revision++;paint();schedule();};
$('display').onchange=()=>{snippet();S.revision++;schedule();};
$('live').onchange=()=>{if($('live').checked)compare();};
$('map-name').onchange=()=>{chooseMap($('map-name').value);chooseSteering();};
async function showCachedMap(name){
  const revision=++S.revision;
  try{
    const result=await rpc({type:'map',mapMetric:S.mapMetric,mapName:name});
    if(revision!==S.revision)return;
    setMaps({...result,mapProfile:S.mapProfile});paint();
    $('compute-timing').textContent=`Map ${name} from the last evaluation · ${result.mapMilliseconds.toFixed(1)} ms`;
  }catch(error){S.mapName=name;S.pendingMap=undefined;S.map=null;paint();schedule();}
}
$('swap').onclick=()=>{[S.reference,S.distorted]=[S.distorted,S.reference];S.generated=false;const a=$('name-reference'),b=$('name-distorted');[a.textContent,b.textContent]=[b.textContent,a.textContent];S.map=null;S.revision++;paint();schedule();scheduleLoop();};
$('quality').oninput=()=>{$('quality-value').textContent=$('quality').value;};
$('quality').onchange=()=>{if(S.generated)generate();};
$('codec').onchange=()=>{S.codec=$('codec').value;$('generate').textContent=`Encode with ${codecInfo[S.codec].label}`;if(S.generated)generate();};
async function generate() {
  if(!S.ready||!S.reference)return;
  const name=$('codec').value,info=codecInfo[name],quality=Number($('quality').value);
  const revision=++S.revision;S.map=null;status(`Encoding with ${info.label}…`);
  try{
    const r=S.reference;const result=await rpc({type:'generate',codec:name,reference:r.data,width:r.width,height:r.height,quality},300);
    if(revision!==S.revision)return;
    S.encoded=result.encoded;S.encodedCodec=name;S.generated=true;$('download-jpeg').disabled=false;
    S.distorted={width:result.width,height:result.height,data:new Uint8ClampedArray(result.rgba)};
    $('name-distorted').textContent=`${info.label} · quality ${quality} · ${(result.bytes/1024).toFixed(1)} KiB`;
    paint();schedule();
  }catch(error){status(error.message,true);}
}
$('generate').onclick=generate;
function drawScene(width,height){
  const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;const g=canvas.getContext('2d');
  const sky=g.createLinearGradient(0,0,0,320);sky.addColorStop(0,'#7fa7d6');sky.addColorStop(1,'#e9d9bf');g.fillStyle=sky;g.fillRect(0,0,768,512);
  g.fillStyle='#3f5d3a';g.beginPath();g.moveTo(0,320);for(let x=0;x<=768;x+=8)g.lineTo(x,320-38*Math.sin(x/70)-18*Math.sin(x/23));g.lineTo(768,512);g.lineTo(0,512);g.fill();
  for(let y=340;y<512;y+=3){g.strokeStyle=`rgba(30,40,20,${.25+.2*Math.sin(y)})`;g.beginPath();g.moveTo(0,y);for(let x=0;x<=768;x+=12)g.lineTo(x,y+1.5*Math.sin(x/9+y));g.stroke();}
  g.fillStyle='#f4efe6';g.fillRect(60,70,255,100);g.fillStyle='#222';g.font='23px sans-serif';g.fillText('Ringing near edges',75,113);g.font='16px serif';g.fillText('Fine detail and flat areas',75,147);
  g.fillStyle='#c8452b';g.beginPath();g.arc(550,125,50,0,Math.PI*2);g.fill();
  for(let i=0;i<6;i++){g.strokeStyle='#fff';g.lineWidth=2;g.beginPath();g.moveTo(425+i*9,40);g.lineTo(476+i*9,230);g.stroke();}
  return g.getImageData(0,0,width,height).data;
}
async function demoPair(){
  S.reference={width:768,height:512,data:drawScene(768,512)};S.generated=true;$('name-reference').textContent='Demo scene · 768×512';fit();paint();await generate();scheduleLoop();
}
$('demo').onclick=demoPair;
const palettes={viridis:[[0,[68,1,84]],[.25,[59,82,139]],[.5,[33,145,140]],[.75,[94,201,98]],[1,[253,231,37]]],turbo:[[0,[48,18,59]],[.2,[62,155,254]],[.4,[70,248,132]],[.6,[225,220,55]],[.8,[240,92,18]],[1,[122,4,3]]]};
// Visualized error: 1 − v for similarity maps; VSI's primary map is a
// quality contribution; feature maps show their raw (non-negative) values.
function errorValue(v){if(!Number.isFinite(v))return 0;if(S.mapMetric==='vsi'&&S.mapName==='quality')return Math.max(0,v);return S.mapPolarity==='similarity'?Math.max(0,1-v):Math.max(0,v);}
function mapIndex(x,y){
  const g=S.geometry;
  if(g){const col=Math.floor((x-g.originX)/g.cellWidth),row=Math.floor((y-g.originY)/g.cellHeight);return col<0||row<0||col>=g.cellsX||row>=g.cellsY?-1:row*g.cellsX+col;}
  return Math.min(S.mapHeight-1,Math.floor(y*S.mapHeight/S.reference.height))*S.mapWidth+Math.min(S.mapWidth-1,Math.floor(x*S.mapWidth/S.reference.width));
}
const ramp=new Uint8ClampedArray(768);
function updatePalette(){const stops=palettes[$('palette').value];
for(let i=0;i<256;i++){let k=1;const x=i/255;while(k<stops.length-1&&x>stops[k][0])k++;const [x0,a]=stops[k-1],[x1,b]=stops[k];for(let c=0;c<3;c++)ramp[i*3+c]=a[c]+(b[c]-a[c])*(x-x0)/(x1-x0);}
for(const node of document.querySelectorAll('.ramp'))node.style.background=`linear-gradient(90deg,${stops.map(([_,c])=>`rgb(${c.join(',')})`).join(',')})`;}

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
    const index=S.map?mapIndex(x,Math.floor(i/r.width)):-1;
    const sample=index>=0&&Number.isFinite(S.map[index])?index:-1;
    if(S.mode==='map'&&S.map&&sample<0){const grey=(Math.floor(x/4)+Math.floor(i/r.width/4))%2?165:190;for(let c=0;c<3;c++)output.data[j+c]=grey;}
    if((S.mode==='overlay'||S.mode==='map')&&S.map&&sample>=0){
      let t=top>0?Math.min(1,Math.max(0,errorValue(S.map[sample])/top)):0;if($('norm').value==='sqrt')t=Math.sqrt(t);const k=Math.round(t*255);
      const weight=S.mode==='map'?1:alpha*Math.sqrt(t),grey=.2126*source.data[j]+.7152*source.data[j+1]+.0722*source.data[j+2];
      for(let c=0;c<3;c++)output.data[j+c]=($('grey').checked?grey:source.data[j+c])*(1-weight)+ramp[k*3+c]*weight;
    }
    output.data[j+3]=255;
  }
  canvas.getContext('2d').putImageData(output,0,0);canvas.style.width=`${r.width*S.zoom}px`;canvas.style.height=`${r.height*S.zoom}px`;
  $('zoom').textContent=`${Math.round(S.zoom*100)}%`;$('dimensions').textContent=`${r.width}×${r.height}`;
  $('scale').disabled=$('norm').value!=='fixed';
  const feature=S.mapPolarity==='feature',vsi=S.mapMetric==='vsi'&&S.mapName==='quality';
  const units=vsi?'quality contribution':feature?'native value':S.mapPolarity==='similarity'?'1 − similarity':S.mapMetric==='cvvdp'?'masked error':S.mapMetric==='butteraugli'?'Butteraugli error':'error';
  $('legend-high').textContent=vsi?'High quality contribution':feature?'High value':'High error';
  $('map-info').textContent=S.map?`${S.mapMetric} · ${S.mapName} · ${S.mapProfile?`model ${S.mapProfile} · `:''}${S.mapWidth}×${S.mapHeight} native samples · 0–${top.toPrecision(4)} ${units}${$('norm').value==='fixed'?' · fixed':' · per-image scale'}`:mapRoute($('metric').value)?'No map yet':mapUnavailable($('metric').value);
}
function fit(){if(S.reference)S.zoom=Math.min(1,Math.max(.05,($('viewport').clientWidth||768)/S.reference.width));paint();}
for(const id of ['opacity','scale','norm','background','grey','split'])$(id).oninput=paint;
$('palette').onchange=()=>{updatePalette();paint();};updatePalette();
$('views').onclick=event=>{if(!event.target.dataset.mode)return;S.mode=event.target.dataset.mode;for(const b of $('views').children)b.classList.toggle('on',b===event.target);paint();};
setInterval(()=>{if(S.mode==='blink'){S.blink=!S.blink;paint();}},500);
$('fit').onclick=fit;$('actual').onclick=()=>{S.zoom=1;paint();};
$('zoom-in').onclick=()=>{S.zoom=Math.min(8,S.zoom*1.25);paint();};$('zoom-out').onclick=()=>{S.zoom=Math.max(.05,S.zoom/1.25);paint();};
$('view').onpointermove=event=>{if(!S.reference)return;const box=$('view').getBoundingClientRect(),x=Math.floor((event.clientX-box.left)/S.zoom),y=Math.floor((event.clientY-box.top)/S.zoom),i=y*S.reference.width+x;
  if(x<0||y<0||x>=S.reference.width||y>=S.reference.height)return;const v=S.map?.[mapIndex(x,y)];$('pixel').textContent=`${x}, ${y} · RGB ${Array.from(S.reference.data.subarray(i*4,i*4+3)).join(', ')}${v===undefined?(S.map?' · no native map sample':''):!Number.isFinite(v)?' · undefined native map value':` · ${S.mapMetric} ${S.mapName} ${v.toPrecision(6)}`}`;};
let drag;
$('viewport').onpointerdown=event=>{drag={x:event.clientX,y:event.clientY,left:$('viewport').scrollLeft,top:$('viewport').scrollTop};$('viewport').setPointerCapture(event.pointerId);};
$('viewport').addEventListener('pointermove',event=>{if(drag){$('viewport').scrollLeft=drag.left+drag.x-event.clientX;$('viewport').scrollTop=drag.top+drag.y-event.clientY;}});
$('viewport').onpointerup=()=>{drag=null;};$('viewport').onpointercancel=()=>{drag=null;};
function save(bytes,type,name){const url=URL.createObjectURL(new Blob([bytes],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('download').onclick=()=>{if(S.map)save(S.map,'application/octet-stream',`${S.mapMetric}-${S.mapName}-${S.mapWidth}x${S.mapHeight}-f32le.bin`);};
$('theme').onclick=()=>{document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark';};
if(!crossOriginIsolated||typeof SharedArrayBuffer==='undefined')status('Threaded WASM requires HTTPS with COOP: same-origin and COEP: require-corp.',true);
else startWorker();

function snippet(){const metric=$('metric').value;
  $('zensim-model-row').hidden=metric!=='zensim';
  $('zensim-model-note').hidden=$('zensim-model-row').hidden||$('zensim-model').value!=='c';
  $('cli-snippet').textContent=`zenmetrics${metric==='zensim'?` --zensim-profile ${$('zensim-model').value}`:''} score --metric ${metric} --reference reference.png --distorted distorted.${codecInfo[S.encodedCodec||'jpeg'].ext}${metric==='cvvdp'?` --display-model ${$('display').value}`:''}`;
}
$('copy-cli').onclick=async()=>{try{await navigator.clipboard.writeText($('cli-snippet').textContent);status('CLI command copied.');}catch(error){status(error.message,true);}};
$('download-jpeg').onclick=()=>{if(S.encoded){const info=codecInfo[S.encodedCodec];save(S.encoded,info.mime,`distorted.${info.ext}`);}};

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

// Encoder loop player: jxl-encoder's perceptual quantization loop on the
// reference image, encoded live in the worker or read from a saved .ltrace.
// `epoch` counts what the player shows; a trace file opened during an
// encode supersedes it, so the encode's frames and result are dropped.
const L={trace:null,frame:0,timer:null,running:false,queued:false,bytes:null,live:[],epoch:0};
// The loop steers with the metric and map selected above, put on
// butteraugli's scale (scale · error^exponent); fitted defaults per map.
// An empty map means the metric's default map, which the fit lists first.
function loopNorm(metric,map){return S.loopDefaults.find(d=>d.metric===metric&&(!map||d.map===map));}
// The map the Map menu shows for the metric: the user's choice, before and
// after the metric evaluates it.
function steerMap(route){const [metric,map]=($('map-name').value||'').split('|');return metric===route?map||'':'';}
function syncLoopSteer(reset=false){
  const builtin=$('loop-steer').value==='builtin',metric=$('metric').value,route=mapRoute(metric);
  const map=route?steerMap(route):'';
  for(const id of ['loop-scale','loop-exponent','loop-default'])$(id).disabled=builtin||!route;
  const fitted=route&&loopNorm(route,map);
  if(reset||L.steerKey!==`${route}/${map}`){
    L.steerKey=`${route}/${map}`;
    $('loop-scale').value=fitted?fitted.scale.toPrecision(4):1;$('loop-exponent').value=fitted?fitted.exponent.toPrecision(4):1;
  }
  $('loop-steer-note').textContent=builtin?'Scored by jxl-encoder\'s own butteraugli, as in a normal encode.'
    :!route?`${metric} has no spatial map, so it cannot steer the loop. Choose a metric with a map, or the encoder's own butteraugli.`
    :S.mapPolarity==='feature'&&route===S.mapMetric&&map===S.mapName?`${route} · ${map} is a feature map with no error direction; choose an error or similarity map.`
    :`Scored by ${route} · ${map||'its default map'} as error, then scale · error^exponent to reach butteraugli's scale. `+
      (fitted?`Fitted default: scale ${fitted.scale.toPrecision(4)}, exponent ${fitted.exponent.toPrecision(4)} (block distances matched to butteraugli's, log r ${fitted.r.toFixed(3)}).`+
        (fitted.r<0.5?' This map tracks butteraugli\'s block distances poorly, so the loop follows the metric\'s own judgment.':''):'No fitted default for this map; starting at 1, 1.');
  $('loop-run').disabled=!S.ready||L.running||!builtin&&!route;
}
$('loop-steer').onchange=()=>{
  const value=$('loop-steer').value;S.steerBuiltin=value==='builtin';
  if(!S.steerBuiltin&&value!==$('map-name').value)chooseMap(value);
  buildMenus();syncLoopSteer(true);scheduleLoop();
};
$('loop-default').onclick=()=>{syncLoopSteer(true);scheduleLoop();};
for(const id of ['loop-scale','loop-exponent','loop-effort','loop-iters'])$(id).addEventListener('change',()=>scheduleLoop());
// The distance slider is logarithmic over the number field's range.
const DMIN=0.1,DMAX=25;
const distanceToSlider=d=>Math.round(1000*Math.log(d/DMIN)/Math.log(DMAX/DMIN));
const sliderToDistance=p=>Number((DMIN*Math.pow(DMAX/DMIN,p/1000)).toPrecision(2));
$('loop-distance-range').value=distanceToSlider(Number($('loop-distance').value));
$('loop-distance-range').oninput=()=>{$('loop-distance').value=sliderToDistance(Number($('loop-distance-range').value));};
$('loop-distance-range').onchange=()=>{$('loop-distance-range').oninput();scheduleLoop();};
$('loop-distance').onchange=()=>{
  const d=Math.min(DMAX,Math.max(DMIN,Number($('loop-distance').value)||1));$('loop-distance').value=d;
  $('loop-distance-range').value=distanceToSlider(d);scheduleLoop();
};
// The loop re-encodes when what it encodes changes: the metric or map
// steering it, the distance, effort, iterations or scale, or the reference.
// A change while it runs encodes once more when it finishes.
const AUTO_PIXELS=4e6;
function scheduleLoop(){
  if(!$('loop-auto').checked||!S.ready||!S.reference)return;
  if(S.reference.width*S.reference.height>AUTO_PIXELS){
    $('loop-info').textContent=`The reference has ${(S.reference.width*S.reference.height/1e6).toFixed(1)} megapixels; press Encode and trace to run the loop on it.`;return;
  }
  if(L.running){L.queued=true;return;}
  runLoop();
}
$('loop-auto').onchange=()=>{if($('loop-auto').checked)scheduleLoop();};
async function runLoop(){
  if(!S.ready||!S.reference||L.running)return;
  const builtin=$('loop-steer').value==='builtin',metric=$('metric').value,route=mapRoute(metric);
  if(!builtin&&!route){syncLoopSteer();return;}
  const map=steerMap(route),scale=Number($('loop-scale').value),exponent=Number($('loop-exponent').value);
  const steer=builtin?null:{metric:route,map,scale,exponent,display:$('display').value};
  const steering=builtin?'butteraugli (built-in)':`${route} · ${map||'default map'} · scale ${scale} · exponent ${exponent}`;
  const r=S.reference,distance=Number($('loop-distance').value),effort=Number($('loop-effort').value),iters=Number($('loop-iters').value);
  L.running=true;L.queued=false;L.liveTarget=distance;L.live=[];L.trace=null;loopStop();const epoch=L.runEpoch=++L.epoch;
  for(const id of ['loop-prev','loop-play','loop-next','loop-frame','loop-save'])$(id).disabled=true;
  $('loop-run').disabled=true;$('loop-info').className='note loopline';
  $('loop-info').textContent=`Encoding ${r.width}×${r.height} at distance ${distance}, effort ${effort}, steered by ${steering}…`;
  showReadout(null);drawChart();
  let ok=false;
  try{
    const result=await rpc({type:'loop',reference:r.data,width:r.width,height:r.height,distance,effort,iters,steer,steering},600);
    if(L.queued||epoch!==L.epoch)return;
    const t=result.trace;L.bytes=t;showTrace(parseTrace(t.buffer.slice(t.byteOffset,t.byteOffset+t.byteLength)),`live encode · ${(result.milliseconds/1000).toFixed(1)} s`);
    $('loop-save').disabled=false;ok=true;
  }catch(error){if(epoch===L.epoch){$('loop-info').textContent=`Loop encode failed: ${error.message}`;$('loop-info').className='note loopline error';}}
  finally{
    L.running=false;syncLoopSteer();
    if(L.queued)runLoop();else if(ok)loopPlay();
  }
}
$('loop-run').onclick=runLoop;
$('loop-save').onclick=()=>{if(L.bytes)save(L.bytes,'application/octet-stream','loop.ltrace');};
function parseTrace(buffer){
  const bytes=new Uint8Array(buffer),magic='JXLTRACE1\n';
  for(let i=0;i<magic.length;i++)if(bytes[i]!==magic.charCodeAt(i))throw new Error('Not a jxl-encoder loop trace');
  const n=new DataView(buffer).getUint32(10,true),head=JSON.parse(new TextDecoder().decode(bytes.subarray(14,14+n)));
  const {width:w,height:h,xsize_blocks:xb,ysize_blocks:yb}=head;let off=14+n;
  const take=k=>{if(off+k>bytes.length)throw new Error('Truncated loop trace');const out=bytes.subarray(off,off+k);off+=k;return out;};
  const f32=k=>{const b=take(k*4);return new Float32Array(buffer.slice(b.byteOffset,b.byteOffset+k*4));};
  const source=take(w*h*3);
  const frames=head.frames.map(f=>({...f,qf:f32(xb*yb),next:f.has_next?f32(xb*yb):null,tile:f32(xb*yb),diffmap:f32(w*h),recon:take(w*h*3)}));
  if(off!==bytes.length)throw new Error('Unexpected bytes after the last loop frame');
  if(!frames.length)throw new Error('The trace holds no loop iterations');
  // One scale per layer across all frames, so steps compare directly.
  const sample=[];let qfMin=Infinity,qfMax=-Infinity,dqfTop=0;
  for(const [k,f] of frames.entries()){
    const step=Math.max(1,Math.floor(f.diffmap.length/50000));for(let i=0;i<f.diffmap.length;i+=step)if(f.diffmap[i]>0)sample.push(f.diffmap[i]);
    // Per-frame block summary: above target, the worst block, the change
    // the loop made for the next iteration.
    f.above=0;f.finer=0;f.coarser=0;f.worst=0;
    for(let b=0;b<f.qf.length;b++){
      qfMin=Math.min(qfMin,f.qf[b]);qfMax=Math.max(qfMax,f.qf[b]);
      if(f.tile[b]>f.target)f.above++;if(f.tile[b]>f.tile[f.worst])f.worst=b;
      if(f.next){dqfTop=Math.max(dqfTop,Math.abs(Math.log2(f.next[b]/f.qf[b])));if(f.next[b]>f.qf[b])f.finer++;else if(f.next[b]<f.qf[b])f.coarser++;}
    }
    f.converged=!!f.next&&f.finer===0&&f.coarser===0;
    const before=frames[k-1];f.repeat=!!before&&before.seed===f.seed&&before.qf.every((q,b)=>q===f.qf[b]);
    f.last=k+1===frames.length||frames[k+1].seed!==f.seed;
  }
  sample.sort((a,b)=>a-b);const diffTop=sample.length?sample[Math.min(sample.length-1,Math.floor(sample.length*.995))]:1;
  return {...head,source,frames,diffTop,qfMin,qfMax,dqfTop};
}
// The steering map over a reconstruction, on the loop's own scale: the
// palette runs linearly from 0 to twice the target, so its middle is the
// target. The blend weight rises linearly to 0.85 at 1.5× the target, so
// the map shows everywhere while the reconstruction stays readable under it.
function overlayPixel(out,i,src,j,v,target){
  const x=Math.max(0,v)/target,k=Math.round(Math.min(1,x/2)*255)*3,a=.85*Math.min(1,x/1.5);
  for(let c=0;c<3;c++)out[i*4+c]=src[j+c]*(1-a)+ramp[k+c]*a;out[i*4+3]=255;
}
function paintLive(f){
  if(!L.running||L.runEpoch!==L.epoch)return;const {width:w,height:h}=f,canvas=$('loop-view');canvas.width=w;canvas.height=h;
  const g=canvas.getContext('2d'),img=g.createImageData(w,h);
  for(let i=0;i<w*h;i++)overlayPixel(img.data,i,f.recon,i*4,f.map[i],L.liveTarget);
  g.putImageData(img,0,0);$('loop-legend').hidden=false;$('loop-legend-high').textContent='2× target (middle: the target)';
  L.live.push({score:f.score,target:L.liveTarget});
  $('loop-info').className='note loopline';
  $('loop-info').textContent=`Encoding live · reconstruction ${f.index+1} scored · map over the reconstruction; blocks and the full readout appear when the encode finishes`;
  showReadout(null,f);drawChart();
}
const fmt=v=>Number.isFinite(v)?v.toFixed(v>=100?1:3):'–';
// The readout keeps one line per field, so playing never moves the page.
function showReadout(f,live){
  const t=L.trace,set=(id,text)=>{$(id).textContent=text;$(id).title=text;};
  if(!f){
    set('ls-iter',live?`live · reconstruction ${live.index+1}`:'–');
    set('ls-score',live?`${fmt(live.score)} / ${fmt(L.liveTarget)} (${Math.round(100*live.score/L.liveTarget)}%)`:'–');
    gauge(live?live.score/L.liveTarget:0);
    for(const id of ['ls-blocks','ls-worst','ls-next'])set(id,live?'when the encode finishes':'–');
    set('ls-encode','–');return;
  }
  const seedFrames=t.frames.filter(x=>x.seed===f.seed),scheduled=t.iters;
  set('ls-iter',`${f.iter} of ${seedFrames.length-1}${seedFrames.length-1<scheduled?` (${scheduled} scheduled)`:''} · seed ${f.seed}`);
  set('ls-score',`${fmt(f.score)} / ${fmt(f.target)} (${Math.round(100*f.score/f.target)}%)`);
  gauge(f.score/f.target);
  set('ls-blocks',`${f.above} of ${f.tile.length} (${(100*f.above/f.tile.length).toFixed(1)}%)`);
  const bx=f.worst%t.xsize_blocks,by=Math.floor(f.worst/t.xsize_blocks);
  set('ls-worst',`block ${bx}, ${by} · ${fmt(f.tile[f.worst])} (${Math.round(100*f.tile[f.worst]/f.target)}%)`);
  set('ls-next',!f.next?'none: final pass, score only':f.converged?'unchanged: converged, later iterations repeat this one':`${f.finer} finer · ${f.coarser} coarser`);
  set('ls-encode',`${t.width}×${t.height} · d${t.distance} e${t.effort} · ${t.bytes} B`);
}
function gauge(ratio){const g=$('ls-gauge');g.style.width=`${Math.min(100,50*Math.max(0,ratio))}%`;g.classList.toggle('over',ratio>1);}
// Score per iteration (dots, a line per seed) against the target (dashed),
// with the share of blocks above the target as bars; the current frame is
// shaded. Live encodes plot scores as they arrive.
function chartFrames(){return L.trace?L.trace.frames:L.live;}
function drawChart(){
  const c=$('loop-chart'),dpr=devicePixelRatio||1,W=c.clientWidth||600,H=c.clientHeight||130;
  c.width=Math.round(W*dpr);c.height=Math.round(H*dpr);const g=c.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,W,H);
  const frames=chartFrames(),css=getComputedStyle(document.documentElement),color=name=>css.getPropertyValue(name).trim();
  g.font='11px system-ui, sans-serif';g.fillStyle=color('--muted');
  if(!frames.length){g.fillText('Score per iteration appears here.',10,H/2);return;}
  const pad={l:46,r:10,t:10,b:20},cw=(W-pad.l-pad.r)/frames.length;
  const top=1.15*Math.max(...frames.map(f=>Math.max(f.score,f.target)));
  const x=k=>pad.l+(k+.5)*cw,y=v=>pad.t+(1-v/top)*(H-pad.t-pad.b);
  if(L.trace){g.fillStyle=color('--line');g.fillRect(pad.l+L.frame*cw,pad.t,cw,H-pad.t-pad.b);}
  // Blocks above target, on their own 0–100% axis.
  g.fillStyle=color('--bar2');g.globalAlpha=.35;
  for(const [k,f] of frames.entries())if(f.above!==undefined){const h=(H-pad.t-pad.b)*f.above/f.tile.length;g.fillRect(x(k)-cw*.3,H-pad.b-h,cw*.6,h);}
  g.globalAlpha=1;
  // Target, dashed.
  g.strokeStyle=color('--ink');g.setLineDash([4,3]);g.lineWidth=1;g.beginPath();
  for(const [k,f] of frames.entries()){const xa=pad.l+k*cw,xb=xa+cw;if(k===0||frames[k-1].target!==f.target)g.moveTo(xa,y(f.target));g.lineTo(xb,y(f.target));}
  g.stroke();g.setLineDash([]);
  // Score: a line per seed, dots coloured by whether they pass the target.
  g.strokeStyle=color('--muted');g.beginPath();
  for(const [k,f] of frames.entries()){if(k===0||(f.seed!==undefined&&frames[k-1].seed!==f.seed))g.moveTo(x(k),y(f.score));else g.lineTo(x(k),y(f.score));}
  g.stroke();
  for(const [k,f] of frames.entries()){g.fillStyle=f.score>f.target?color('--accent'):color('--good');g.beginPath();g.arc(x(k),y(f.score),k===L.frame&&L.trace?4.5:3,0,2*Math.PI);g.fill();}
  g.fillStyle=color('--muted');g.textAlign='right';
  g.fillText(fmt(top),pad.l-6,pad.t+8);g.fillText('0',pad.l-6,H-pad.b);
  const target=frames[frames.length-1].target;g.fillStyle=color('--ink');g.fillText(`target ${fmt(target)}`,W-pad.r-2,y(target)-4);
  g.textAlign='center';g.fillStyle=color('--muted');
  const every=Math.max(1,Math.ceil(frames.length/24));
  for(const [k,f] of frames.entries())if(k%every===0)g.fillText(f.iter!==undefined?String(f.iter):String(k),x(k),H-6);
  g.textAlign='left';
}
$('loop-chart').onclick=event=>{
  if(!L.trace)return;const box=$('loop-chart').getBoundingClientRect(),cw=(box.width-56)/L.trace.frames.length;
  const k=Math.floor((event.clientX-box.left-46)/cw);if(k<0||k>=L.trace.frames.length)return;
  loopStop();L.frame=k;paintLoop();
};
addEventListener('resize',()=>drawChart());
function paintLoop(){
  const t=L.trace;if(!t)return;const f=t.frames[L.frame],{width:w,height:h}=t,canvas=$('loop-view');canvas.width=w;canvas.height=h;
  const g=canvas.getContext('2d'),img=g.createImageData(w,h),out=img.data,layer=$('loop-layer').value;
  const put=(i,r,gg,b)=>{out[i*4]=r;out[i*4+1]=gg;out[i*4+2]=b;out[i*4+3]=255;};
  const heat=(i,x)=>{const k=Math.max(0,Math.min(255,Math.round(x*255)))*3;put(i,ramp[k],ramp[k+1],ramp[k+2]);};
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const i=y*w+x,b=(y>>3)*t.xsize_blocks+(x>>3);
    if(layer==='overlay')overlayPixel(out,i,f.recon,i*3,f.diffmap[i],f.target);
    else if(layer==='composite'){
      // The map on the trace's own scale (as the diffmap layer), so it shows
      // even far below the target; the target is in the block outlines.
      const v=Math.min(1,Math.sqrt(Math.max(0,f.diffmap[i])/t.diffTop)),k=Math.round(v*255)*3,a=.75*v;
      for(let c=0;c<3;c++)out[i*4+c]=f.recon[i*3+c]*(1-a)+ramp[k+c]*a;out[i*4+3]=255;
      // Block outlines: the outer ring of each block above the target in
      // magenta; the worst block also gets a white inner ring.
      const bx=x&7,by=y&7,outer=bx===0||by===0||bx===7||by===7||x===w-1||y===h-1;
      const inner=!outer&&(bx===1||by===1||bx===6||by===6||x===w-2||y===h-2);
      if(outer&&f.tile[b]>f.target)put(i,255,0,255);
      else if(outer&&b===f.worst)put(i,0,0,0);
      else if(inner&&b===f.worst)put(i,255,255,255);
    }
    else if(layer==='recon'||layer==='source'){const s=layer==='recon'?f.recon:t.source;put(i,s[i*3],s[i*3+1],s[i*3+2]);}
    else if(layer==='diffmap')heat(i,Math.sqrt(Math.max(0,f.diffmap[i])/t.diffTop));
    else if(layer==='tile')heat(i,f.tile[b]/(2*f.target));
    else if(layer==='qf')heat(i,(f.qf[b]-t.qfMin)/((t.qfMax-t.qfMin)||1));
    else{const d=f.next?Math.log2(f.next[b]/f.qf[b])/(t.dqfTop||1):0,c=Math.round(Math.min(1,Math.abs(d))*255);put(i,d<0?255-c:255,255-c,d>0?255-c:255);}
  }
  g.putImageData(img,0,0);
  $('loop-legend-high').textContent=layer==='overlay'?'2× target (middle: the target)':layer==='composite'?'99.5th percentile of all frames (square root)':'';
  $('loop-legend').hidden=layer!=='overlay'&&layer!=='composite';
  const text=`Frame ${L.frame+1} of ${t.frames.length} · ${f.metric} · ${L.origin||'trace'}`+(t.steering?` · steered by ${t.steering}`:'')+
    (f.repeat?' · same quant field and image as the previous iteration':'');
  $('loop-info').className='note loopline';$('loop-info').textContent=text;$('loop-info').title=text;
  $('loop-frame').value=L.frame;showReadout(f);drawChart();
}
$('loop-view').onpointermove=event=>{
  const t=L.trace;if(!t)return;const f=t.frames[L.frame],box=$('loop-view').getBoundingClientRect();
  const x=Math.floor((event.clientX-box.left)*t.width/box.width),y=Math.floor((event.clientY-box.top)*t.height/box.height);
  if(x<0||y<0||x>=t.width||y>=t.height)return;
  const bx=x>>3,by=y>>3,b=by*t.xsize_blocks+bx;
  $('loop-hover').textContent=`pixel ${x}, ${y} · map ${fmt(f.diffmap[y*t.width+x])} · block ${bx}, ${by} · distance ${fmt(f.tile[b])} (${Math.round(100*f.tile[b]/f.target)}% of target) · quant field ${fmt(f.qf[b])}`+
    (f.next?` → ${fmt(f.next[b])}`:'');
};
function loopStep(k){if(!L.trace)return;const n=L.trace.frames.length;L.frame=((L.frame+k)%n+n)%n;paintLoop();}
function loopStop(){clearTimeout(L.timer);L.timer=null;$('loop-play').textContent='▶ Play';}
// Playback holds the last frame of the trace a little longer before looping.
function loopPlay(){
  if(!L.trace)return;loopStop();$('loop-play').textContent='⏸ Pause';
  const wait=()=>L.frame===L.trace.frames.length-1?2000:900;
  const tick=()=>{loopStep(1);L.timer=setTimeout(tick,wait());};
  L.timer=setTimeout(tick,wait());
}
function showTrace(trace,origin){
  loopStop();L.trace=trace;L.frame=0;L.origin=origin;
  for(const id of ['loop-prev','loop-play','loop-next','loop-frame'])$(id).disabled=false;
  $('loop-frame').max=L.trace.frames.length-1;paintLoop();
}
async function loadTrace(file){
  if(!file)return;loopStop();++L.epoch;L.queued=false;
  try{const buffer=await file.arrayBuffer();showTrace(parseTrace(buffer),file.name);L.bytes=new Uint8Array(buffer);$('name-trace').textContent=file.name;$('loop-save').disabled=false;loopPlay();}
  catch(error){L.trace=null;$('loop-info').textContent=error.message;$('loop-info').className='note loopline error';}
}
$('trace').onchange=()=>loadTrace($('trace').files[0]);
$('drop-trace').ondragover=e=>{e.preventDefault();$('drop-trace').classList.add('drag');};
$('drop-trace').ondragleave=()=>$('drop-trace').classList.remove('drag');
$('drop-trace').ondrop=e=>{e.preventDefault();$('drop-trace').classList.remove('drag');loadTrace(e.dataTransfer.files[0]);};
$('loop-prev').onclick=()=>{loopStop();loopStep(-1);};
$('loop-next').onclick=()=>{loopStop();loopStep(1);};
$('loop-frame').oninput=()=>{loopStop();L.frame=Number($('loop-frame').value);paintLoop();};
$('loop-layer').onchange=paintLoop;
$('loop-play').onclick=()=>{if(L.timer)loopStop();else loopPlay();};
$('palette').addEventListener('change',paintLoop);
drawChart();
