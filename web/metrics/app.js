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
    if(data.type==='loop-progress'){$('loop-info').textContent=data.message||`Encoding: scored ${data.scored} reconstructions (last ${data.score.toFixed(4)})…`;return;}
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
    if(side==='reference'&&S.generated)await generate();else schedule();
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
  S.mapName=result.mapName||'';S.mapNames=result.mapNames||[];S.mapPolarity=result.mapPolarity;
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
  const map=route===S.mapMetric?S.mapName:(S.requestedMap||defaultMapOf(route)||'');
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
  showCachedMap(map);
}
$('metric').onchange=()=>{S.map=null;S.mapName='';S.mapMetric=null;S.requestedMap=undefined;syncMetric();S.revision++;paint();schedule();};
$('zensim-model').onchange=()=>{snippet();S.map=null;S.revision++;paint();schedule();};
$('display').onchange=()=>{snippet();S.revision++;schedule();};
$('live').onchange=()=>{if($('live').checked)compare();};
$('map-name').onchange=()=>chooseMap($('map-name').value);
async function showCachedMap(name){
  const revision=++S.revision;
  try{
    const result=await rpc({type:'map',mapMetric:S.mapMetric,mapName:name});
    if(revision!==S.revision)return;
    setMaps({...result,mapProfile:S.mapProfile});paint();
    $('compute-timing').textContent=`Map ${name} from the last evaluation · ${result.mapMilliseconds.toFixed(1)} ms`;
  }catch(error){S.mapName=name;S.map=null;paint();schedule();}
}
$('swap').onclick=()=>{[S.reference,S.distorted]=[S.distorted,S.reference];S.generated=false;const a=$('name-reference'),b=$('name-distorted');[a.textContent,b.textContent]=[b.textContent,a.textContent];S.map=null;S.revision++;paint();schedule();};
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
  S.reference={width:768,height:512,data:drawScene(768,512)};S.generated=true;$('name-reference').textContent='Demo scene · 768×512';fit();paint();await generate();
  if(!L.trace&&!L.running)runLoop();
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
const L={trace:null,frame:0,timer:null,running:false,bytes:null};
// The loop steers with the metric and map selected above, put on
// butteraugli's scale (scale · error^exponent); fitted defaults per map.
// An empty map means the metric's default map, which the fit lists first.
function loopNorm(metric,map){return S.loopDefaults.find(d=>d.metric===metric&&(!map||d.map===map));}
function syncLoopSteer(reset=false){
  const builtin=$('loop-steer').value==='builtin',metric=$('metric').value,route=mapRoute(metric);
  const map=route&&route===S.mapMetric?S.mapName:'';
  for(const id of ['loop-scale','loop-exponent','loop-default'])$(id).disabled=builtin||!route;
  const fitted=route&&loopNorm(route,map);
  if(reset||L.steerKey!==`${route}/${map}`){
    L.steerKey=`${route}/${map}`;
    $('loop-scale').value=fitted?fitted.scale.toPrecision(4):1;$('loop-exponent').value=fitted?fitted.exponent.toPrecision(4):1;
  }
  $('loop-steer-note').textContent=builtin?'Scored by jxl-encoder\'s own butteraugli, as in a normal encode.'
    :!route?`${metric} has no spatial map, so it cannot steer the loop. Choose a metric with a map, or the encoder's own butteraugli.`
    :S.mapPolarity==='feature'&&route===S.mapMetric?`${route} · ${map} is a feature map with no error direction; choose an error or similarity map.`
    :`Scored by ${route} · ${map||'its default map'} as error, then scale · error^exponent to reach butteraugli's scale. `+
      (fitted?`Fitted default: scale ${fitted.scale.toPrecision(4)}, exponent ${fitted.exponent.toPrecision(4)} (block distances matched to butteraugli's, log r ${fitted.r.toFixed(3)}).`+
        (fitted.r<0.5?' This map tracks butteraugli\'s block distances poorly, so the loop follows the metric\'s own judgment.':''):'No fitted default for this map; starting at 1, 1.');
  $('loop-run').disabled=!S.ready||L.running||!builtin&&!route;
}
$('loop-steer').onchange=()=>{
  const value=$('loop-steer').value;S.steerBuiltin=value==='builtin';
  if(!S.steerBuiltin&&value!==$('map-name').value)chooseMap(value);
  buildMenus();syncLoopSteer(true);
};
$('loop-default').onclick=()=>syncLoopSteer(true);
async function runLoop(){
  if(!S.ready||!S.reference||L.running)return;
  const builtin=$('loop-steer').value==='builtin',metric=$('metric').value,route=mapRoute(metric);
  if(!builtin&&!route){syncLoopSteer();return;}
  const map=route===S.mapMetric?S.mapName:'',scale=Number($('loop-scale').value),exponent=Number($('loop-exponent').value);
  const steer=builtin?null:{metric:route,map,scale,exponent,display:$('display').value};
  const steering=builtin?'butteraugli (built-in)':`${route} · ${map||'default map'} · scale ${scale} · exponent ${exponent}`;
  const r=S.reference,distance=Number($('loop-distance').value),effort=Number($('loop-effort').value),iters=Number($('loop-iters').value);
  L.running=true;L.liveTarget=distance;loopStop();$('loop-run').disabled=true;$('loop-info').className='note';$('loop-info').textContent=`Encoding ${r.width}×${r.height} at distance ${distance}, effort ${effort}, steered by ${steering}…`;
  try{
    const result=await rpc({type:'loop',reference:r.data,width:r.width,height:r.height,distance,effort,iters,steer,steering},600);
    const t=result.trace;L.bytes=t;showTrace(parseTrace(t.buffer.slice(t.byteOffset,t.byteOffset+t.byteLength)),`live encode · ${(result.milliseconds/1000).toFixed(1)} s`);
    $('loop-save').disabled=false;
  }catch(error){$('loop-info').textContent=`Loop encode failed: ${error.message}`;$('loop-info').className='note error';}
  finally{L.running=false;syncLoopSteer();}
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
  for(const f of frames){
    const step=Math.max(1,Math.floor(f.diffmap.length/50000));for(let i=0;i<f.diffmap.length;i+=step)if(f.diffmap[i]>0)sample.push(f.diffmap[i]);
    for(let b=0;b<f.qf.length;b++){qfMin=Math.min(qfMin,f.qf[b]);qfMax=Math.max(qfMax,f.qf[b]);if(f.next)dqfTop=Math.max(dqfTop,Math.abs(Math.log2(f.next[b]/f.qf[b])));}
  }
  sample.sort((a,b)=>a-b);const diffTop=sample.length?sample[Math.min(sample.length-1,Math.floor(sample.length*.995))]:1;
  return {...head,source,frames,diffTop,qfMin,qfMax,dqfTop};
}
// The steering map over a reconstruction, on the loop's own scale: the
// palette's middle is the target distance, so a pixel at or past the middle
// is where the map exceeds what the loop aims for.
function overlayPixel(out,i,src,j,v,target){
  // The blend weight rises steeply (t³) so pixels well below the target stay
  // readable and those near or past it stand out.
  const t=Math.sqrt(Math.min(1,Math.max(0,v)/(2*target))),k=Math.round(t*255)*3,a=.9*t*t*t;
  for(let c=0;c<3;c++)out[i*4+c]=src[j+c]*(1-a)+ramp[k+c]*a;out[i*4+3]=255;
}
function paintLive(f){
  if(!L.running)return;const {width:w,height:h}=f,canvas=$('loop-view');canvas.width=w;canvas.height=h;
  const g=canvas.getContext('2d'),img=g.createImageData(w,h);
  for(let i=0;i<w*h;i++)overlayPixel(img.data,i,f.recon,i*4,f.map[i],L.liveTarget);
  g.putImageData(img,0,0);
  $('loop-info').className='note';
  $('loop-info').textContent=`Encoding live · reconstruction ${f.index+1} scored ${f.score.toFixed(4)} against target ${L.liveTarget} · map over the reconstruction, the palette's middle is the target`;
}
function paintLoop(){
  const t=L.trace;if(!t)return;const f=t.frames[L.frame],{width:w,height:h}=t,canvas=$('loop-view');canvas.width=w;canvas.height=h;
  const g=canvas.getContext('2d'),img=g.createImageData(w,h),out=img.data,layer=$('loop-layer').value;
  const put=(i,r,gg,b)=>{out[i*4]=r;out[i*4+1]=gg;out[i*4+2]=b;out[i*4+3]=255;};
  const heat=(i,x)=>{const k=Math.max(0,Math.min(255,Math.round(x*255)))*3;put(i,ramp[k],ramp[k+1],ramp[k+2]);};
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const i=y*w+x,b=(y>>3)*t.xsize_blocks+(x>>3);
    if(layer==='overlay')overlayPixel(out,i,f.recon,i*3,f.diffmap[i],f.target);
    else if(layer==='recon'||layer==='source'){const s=layer==='recon'?f.recon:t.source;put(i,s[i*3],s[i*3+1],s[i*3+2]);}
    else if(layer==='diffmap')heat(i,Math.sqrt(Math.max(0,f.diffmap[i])/t.diffTop));
    else if(layer==='tile')heat(i,f.tile[b]/(2*f.target));
    else if(layer==='qf')heat(i,(f.qf[b]-t.qfMin)/((t.qfMax-t.qfMin)||1));
    else{const d=f.next?Math.log2(f.next[b]/f.qf[b])/(t.dqfTop||1):0,c=Math.round(Math.min(1,Math.abs(d))*255);put(i,d<0?255-c:255,255-c,d>0?255-c:255);}
  }
  g.putImageData(img,0,0);
  let above=0,finer=0,coarser=0;for(let b=0;b<f.tile.length;b++){if(f.tile[b]>f.target)above++;if(f.next){if(f.next[b]>f.qf[b])finer++;else if(f.next[b]<f.qf[b])coarser++;}}
  const last=t.frames.filter(x=>x.seed===f.seed).length-1,before=t.frames[L.frame-1];
  const same=before&&before.seed===f.seed&&before.qf.every((q,b)=>q===f.qf[b]);
  $('loop-info').textContent=`Frame ${L.frame+1} of ${t.frames.length} · seed ${f.seed} · iteration ${f.iter} of ${last} · ${f.metric} ${f.score.toFixed(4)} (target ${f.target.toFixed(4)}) · ${above} of ${f.tile.length} blocks above target · `+
    (f.next?`next: ${finer} blocks finer, ${coarser} coarser`:'score-only final pass')+` · ${t.width}×${t.height}, d${t.distance} e${t.effort}, final ${t.bytes} bytes`+(t.steering?` · steered by ${t.steering}`:'')+(L.origin?` · ${L.origin}`:'')+
    (same?' · same quant field and image as the previous iteration: from iteration 2 the loop only refines blocks above the target':'');
  $('loop-frame').value=L.frame;
}
function loopStep(k){if(!L.trace)return;const n=L.trace.frames.length;L.frame=((L.frame+k)%n+n)%n;paintLoop();}
function loopStop(){clearInterval(L.timer);L.timer=null;$('loop-play').textContent='▶ Play';}
function showTrace(trace,origin){
  loopStop();L.trace=trace;L.frame=0;L.origin=origin;
  for(const id of ['loop-prev','loop-play','loop-next','loop-frame'])$(id).disabled=false;
  $('loop-frame').max=L.trace.frames.length-1;$('loop-info').className='note';paintLoop();
}
async function loadTrace(file){
  if(!file)return;loopStop();
  try{const buffer=await file.arrayBuffer();showTrace(parseTrace(buffer),file.name);L.bytes=new Uint8Array(buffer);$('name-trace').textContent=file.name;$('loop-save').disabled=false;}
  catch(error){L.trace=null;$('loop-info').textContent=error.message;$('loop-info').className='note error';}
}
$('trace').onchange=()=>loadTrace($('trace').files[0]);
$('drop-trace').ondragover=e=>{e.preventDefault();$('drop-trace').classList.add('drag');};
$('drop-trace').ondragleave=()=>$('drop-trace').classList.remove('drag');
$('drop-trace').ondrop=e=>{e.preventDefault();$('drop-trace').classList.remove('drag');loadTrace(e.dataTransfer.files[0]);};
$('loop-prev').onclick=()=>{loopStop();loopStep(-1);};
$('loop-next').onclick=()=>{loopStop();loopStep(1);};
$('loop-frame').oninput=()=>{loopStop();L.frame=Number($('loop-frame').value);paintLoop();};
$('loop-layer').onchange=paintLoop;
$('loop-play').onclick=()=>{if(L.timer){loopStop();return;}$('loop-play').textContent='⏸ Pause';L.timer=setInterval(()=>loopStep(1),800);};
$('palette').addEventListener('change',paintLoop);
