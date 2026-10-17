import {escape as esc} from './plots.js';
const unique=a=>[...new Set(a)];
export const size=r=>`${r.width}×${r.height}`;
export const device=r=>String(r.backend).toLowerCase()==='cpu'?'CPU':'GPU';
// Historical workload names remain exact: warm, strip, latency and throughput
// measurements cannot be promoted to the same score-only contract.
export const mode=r=>r.mode==='score'?'score-only':r.mode;
const dimensions=['os','backend','phase','engine','content','dtype','tf32_conv'];
export const workload=r=>dimensions.filter(k=>r[k]!=null).map(k=>`${k}=${r[k]}`).join(' · ')||'not recorded';
export const measured=rows=>rows.filter(r=>Number.isFinite(r.ms)&&r.ms>0&&r.width>0&&r.height>0);
export const family=m=>/lpips|dists|dream|arniqa|musiq|hyper|maniqa|paq|liqe|deep|ckdn|afine|wadi|pieapp|topiq|ahiq|wadiqam|wd$/.test(m)?'Learned':/ssim|psnr|gmsd|vif|fsim|haar|vsi/.test(m)?'Structural':'Perceptual / other';
const colors={'Learned':'#a97bcc','Structural':'#3c9bc0','Perceptual / other':'#36a582'};
export function chooseDefault(rows,sources){
 const groups=new Map();
 for(const r of measured(rows).filter(r=>r.implementation==='zenmetrics')){const key=JSON.stringify([r.host,device(r),String(r.threads),mode(r),workload(r),r.source,size(r)]);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(r)}
 return [...groups.values()].sort((a,b)=>unique(b.map(r=>r.metric)).length-unique(a.map(r=>r.metric)).length||String(sources[b[0].source]?.date).localeCompare(String(sources[a[0].source]?.date)))[0]?.[0];
}
export function selectRanking(rows,p){return measured(rows).filter(r=>r.host===p.get('host')&&device(r)===p.get('device')&&String(r.threads)===p.get('threads')&&mode(r)===p.get('mode')&&workload(r)===p.get('workload')&&(p.get('impl')==='competitors'||r.implementation==='zenmetrics')&&(!p.get('source')||r.source===p.get('source')))}
const value=(r,unit)=>r.ms/(unit==='ms/MP'?r.width*r.height/1e6:1);
export function rankingBars(rows,unit,sources){
 const sorted=[...rows].sort((a,b)=>value(a,unit)-value(b,unit));if(!sorted.length)return '<p>No measurements in this exact cell.</p>';
 const vals=sorted.flatMap(r=>[value(r,unit),r.p10_ms?value({...r,ms:r.p10_ms},unit):value(r,unit),r.p90_ms?value({...r,ms:r.p90_ms},unit):value(r,unit)]);
 const low=Math.min(...vals)/1.3,high=Math.max(...vals)*1.3, x=v=>310+430*(Math.log(v)-Math.log(low))/(Math.log(high)-Math.log(low));
 const sourceIds=unique(sorted.map(r=>r.source));
 return `<svg viewBox="0 0 850 ${65+sorted.length*32}" style="min-width:700px" role="img" aria-label="Metric speed ranking, logarithmic ${esc(unit)} axis"><text x="310" y="15">${esc(unit)} · log scale → slower</text>${[0,.25,.5,.75,1].map(t=>{const v=Math.exp(Math.log(low)+t*(Math.log(high)-Math.log(low)));return `<path d="M${x(v)} 30V${40+sorted.length*32}" stroke="currentColor" opacity=".15"/><text x="${x(v)}" y="30">${v.toPrecision(2)}</text>`}).join('')}${sorted.map((r,i)=>{const y=45+i*32,s=sources[r.source]||{},competitor=r.implementation!=='zenmetrics';return `<g class="rank-bar" data-metric="${esc(r.metric)}" data-ms="${r.ms}"><title>${esc(JSON.stringify({...r,...s},null,2))}</title><text x="0" y="${y+13}">${competitor?'◆ ':''}${esc(r.metric)} · ${esc(r.implementation)}</text><rect x="310" y="${y}" width="${Math.max(1,x(value(r,unit))-310)}" height="19" fill="${colors[family(r.metric)]}" ${competitor?'stroke="currentColor" stroke-dasharray="3 2"':''}/>${r.p10_ms>0&&r.p90_ms>0?`<path class="whisker" d="M${x(value({...r,ms:r.p10_ms},unit))} ${y+9}H${x(value({...r,ms:r.p90_ms},unit))}m0 -5v10" stroke="currentColor" fill="none"/>`:''}<text x="${x(value(r,unit))+5}" y="${y+14}">${value(r,unit).toPrecision(3)} [${sourceIds.indexOf(r.source)+1}]</text></g>`}).join('')}</svg><p class="source-legend">${sourceIds.map((s,i)=>`[${i+1}] ${esc(s)} · ${esc(sources[s]?.date)} · ${esc(sources[s]?.commit)}`).join('<br>')}</p>`;
}
export function slope(rows,a,b){
 const ranks=s=>{const seen=new Map();[...rows].filter(r=>size(r)===s).sort((a,b)=>a.ms-b.ms).forEach(r=>{const key=r.metric+' · '+r.implementation;if(!seen.has(key))seen.set(key,seen.size+1)});return seen};
 const left=ranks(a),right=ranks(b),keys=[...left.keys()].filter(k=>right.has(k)),height=80+Math.max(left.size,right.size)*24;
 return `<p>Ranks use each metric / implementation’s fastest selected observation at each size; absent endpoints are omitted. ${keys.length} shared entries.</p><svg viewBox="0 0 850 ${height}" style="min-width:700px" role="img" aria-label="Rank change between resolutions"><text x="260" y="20">${esc(a)}</text><text x="530" y="20">${esc(b)}</text>${keys.map(k=>`<g><title>${esc(k)}: ${left.get(k)} → ${right.get(k)}</title><path d="M300 ${40+left.get(k)*24}L550 ${40+right.get(k)*24}" stroke="${colors[family(k)]}"/><text x="0" y="${44+left.get(k)*24}">${esc(k)} #${left.get(k)}</text><text x="560" y="${44+right.get(k)*24}">#${right.get(k)} ${esc(k)}</text></g>`).join('')}</svg>`;
}
export function rankingUI(rows,p,sources,onChange,catalog=[]){
 const all=measured(rows),d=chooseDefault(rows,sources);
 const defaults={host:d?.host,device:d&&device(d),threads:String(d?.threads),mode:d&&mode(d),workload:d&&workload(d),source:d?.source,size:d&&size(d),size2:size([...all].sort((a,b)=>b.width*b.height-a.width*a.height)[0]),unit:'ms',layout:'single',impl:'zenmetrics'};
 for(const [k,v] of Object.entries(defaults))if(!p.has(k)&&v)p.set(k,v);
 // Source may be explicitly empty to request a marked, mixed-source comparison.
 const fields={host:unique(all.map(r=>r.host)),device:['CPU','GPU'],threads:unique(all.filter(r=>r.host===p.get('host')).map(r=>String(r.threads))),mode:unique(all.map(mode)),workload:unique(all.filter(r=>r.host===p.get('host')&&device(r)===p.get('device')&&String(r.threads)===p.get('threads')&&mode(r)===p.get('mode')).map(workload)),size:unique(all.sort((a,b)=>a.width*a.height-b.width*b.height).map(size)),impl:['zenmetrics','competitors'],source:['',...unique(all.filter(r=>r.host===p.get('host')).map(r=>r.source))],unit:['ms','ms/MP'],layout:['single','all sizes','rank change']};
 fields.size2=fields.size;
 const labels={mode:'MODE (exact historical modes retained)',workload:'WORKLOAD / BACKEND (exact)',impl:'IMPLEMENTATIONS',source:'SOURCE',size:'EXACT RESOLUTION',size2:'SECOND RESOLUTION',threads:'THREADS (exact reported count)'};
 const html=Object.entries(fields).map(([k,vs])=>`<label>${labels[k]||k.toUpperCase()}<select data-ranking="${k}">${vs.map(v=>`<option value="${esc(v)}" ${p.get(k)===v?'selected':''}>${esc(v===''?'All sources · marked comparison':k==='impl'?(v==='zenmetrics'?'zenmetrics only':'Include competitors ◆'):k==='threads'&&v==='1'?'1 · single thread':k==='threads'&&v==='20'&&p.get('host')==='i265'?'20 · all host threads':k==='size'?(v+(' · '+unique(all.filter(r=>size(r)===v).map(r=>r.size).filter(Boolean)).join(', '))):v)}</option>`).join('')}</select></label>`).join('');
 document.querySelector('#filters').innerHTML=html;
 document.querySelectorAll('[data-ranking]').forEach(e=>e.onchange=()=>{p.set(e.dataset.ranking,e.value);if(e.dataset.ranking==='impl'||e.dataset.ranking==='layout')p.set('source','');if(e.dataset.ranking==='layout'&&e.value==='rank change'){const available=unique(selectRanking(rows,p).map(size)).filter(s=>s!==p.get('size'));if(available.length)p.set('size2',available.at(-1))}if(['host','device','threads','mode'].includes(e.dataset.ranking)){p.delete('workload');p.set('source','');const candidates=all.filter(r=>r.host===p.get('host')&&device(r)===p.get('device')&&String(r.threads)===p.get('threads')&&mode(r)===p.get('mode'));if(candidates.length)p.set('workload',workload(candidates[0]));else p.set('workload','unmeasured')}onChange(p)});
 history.replaceState(null,'','#ranking?'+p);
 const selected=selectRanking(rows,p),cell=selected.filter(r=>size(r)===p.get('size')),sizes=fields.size;
 const shown=p.get('layout')==='single'?cell:selected;
 const signatures=unique(shown.map(r=>JSON.stringify([r.source,sources[r.source]?.date,sources[r.source]?.commit])));
 const missing=unique([...rows,...catalog].map(r=>r.metric)).filter(m=>!cell.some(r=>r.metric===m)).sort();
 const banner=signatures.length>1?'<p class="warning" role="status">Multiple sources, dates or commits are selected. Source numbers identify each bar; timing conditions may differ. No measurements are averaged across sources.</p>':'';
 const chart=p.get('layout')==='all sizes'?'<div class="rank-multiples">'+sizes.map(s=>`<section><h3>${esc(s)}</h3>${rankingBars(selected.filter(r=>size(r)===s),p.get('unit'),sources)}</section>`).join('')+'</div>':p.get('layout')==='rank change'?slope(selected,p.get('size'),p.get('size2')||sizes.at(-1)):rankingBars(cell,p.get('unit'),sources);
 document.querySelector('#chart').innerHTML=banner+`<p>Family colours: <span style="color:#a97bcc">Learned</span> · <span style="color:#3c9bc0">Structural</span> · <span style="color:#36a582">Perceptual / other</span>. ◆ competitor. Whiskers: reported p10–p90. Hover bars for provenance and sample count.</p>`+chart+`<details open id="missing-ranking"><summary>Not measured at this size and selected conditions (${missing.length})</summary><p>${missing.map(esc).join(', ')||'None'}</p></details>`;
 return {current:[...shown].sort((a,b)=>a.ms-b.ms),params:p};
}
