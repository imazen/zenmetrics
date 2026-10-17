export const finite = x => typeof x === 'number' && Number.isFinite(x);
export const fmt = x => finite(x) ? x.toPrecision(5) : 'missing';
export const escape = v => String(v ?? 'missing').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const palette=['#12775f','#bf5825','#675bc4','#2179ac','#b34e83','#66822c','#775840','#248687'];
const color = value => palette[[...String(value)].reduce((s,c)=>s+c.charCodeAt(0),0)%palette.length];
const median = values => {const a=values.toSorted((a,b)=>a-b),m=Math.floor(a.length/2);return a.length%2?a[m]:(a[m-1]+a[m])/2};
const key = (row,fields) => fields.map(k=>row[k]??'missing').join(' / ');
function mark(x,y,row,shade){
 const title='<title>'+escape(Object.entries(row).map(([k,v])=>`${k}: ${v}`).join('\n'))+'</title>';
 const warm=String(row.phase||row.mode||row.series).includes('warm');
 const hollow=String(row.series).includes('published');
 const attrs=`fill="${hollow?'none':shade}" stroke="${shade}" stroke-width="1.4" opacity=".7"`;
 return warm?`<rect x="${x-3}" y="${y-3}" width="6" height="6" ${attrs}>${title}</rect>`:`<circle cx="${x}" cy="${y}" r="3.5" ${attrs}>${title}</circle>`;
}
export function plot(rows,xKey,yKey,{logX=false,logY=false,categorical=false,colorBy='threads'}={}){
 const valid=rows.filter(r=>(categorical?r[xKey]!=null:finite(r[xKey])&&(!logX||r[xKey]>0))&&finite(r[yKey])&&(!logY||r[yKey]>0));
 const zeroRows=logY?rows.filter(r=>r[yKey]===0):[];
 if(!valid.length&&!zeroRows.length)return '<p class="chart-caption">Missing: no finite nonzero observations for this plot selection.</p>';
 const categories=[...new Set(rows.map(r=>String(r[xKey])))];
 const xv=r=>categorical?categories.indexOf(String(r[xKey])):logX?Math.log10(r[xKey]):r[xKey];
 const yv=r=>logY?Math.log10(r[yKey]):r[yKey];
 const xs=(valid.length?valid:zeroRows).map(xv),ys=valid.map(yv);let minX=Math.min(...xs),maxX=Math.max(...xs),minY=ys.length?(logY?Math.min(...ys):Math.min(0,...ys)):0,maxY=ys.length?Math.max(...ys):1;
 if(minX===maxX){minX-=.5;maxX+=.5}if(minY===maxY)maxY+=1;
 const x=v=>80+(v-minX)/(maxX-minX)*750,y=v=>280-(v-minY)/(maxY-minY)*240;
 const xTicks=categorical?categories.filter((_,i)=>i%Math.max(1,Math.ceil(categories.length/6))===0).map(label=>({v:categories.indexOf(label),label})):Array.from({length:5},(_,i)=>({v:minX+(maxX-minX)*i/4,label:fmt(logX?10**(minX+(maxX-minX)*i/4):minX+(maxX-minX)*i/4)}));
 const svg=`<svg viewBox="0 0 900 365" role="img" aria-label="${escape(yKey)} versus ${escape(xKey)}"><path class="axis" fill="none" d="M80 35V280H840"/>
 ${(valid.length?[0,.25,.5,.75,1]:[]).map(t=>`<text x="0" y="${284-240*t}">${fmt(logY?10**(minY+(maxY-minY)*t):minY+(maxY-minY)*t)}</text>`).join('')}
 ${xTicks.map(t=>`<text x="${x(t.v)}" y="302">${escape(t.label)}</text>`).join('')}
 <text x="80" y="18">${escape(yKey)} · ${valid.length?(logY?'logarithmic':'linear'):'exact zeros only; no nonzero range'} y scale, local to this group</text><text x="320" y="355">${escape(xKey)}${logX?' (log pixels)':''} →</text>
 ${valid.map(r=>mark(x(xv(r)),y(yv(r)),r,color(r[colorBy]))).join('')}
 ${zeroRows.length?`<text x="0" y="330">Exact zero</text>`+zeroRows.map(r=>mark(x(xv(r)),326,r,color(r[colorBy]))).join(''):''}</svg>`;
 const values=[...new Set(valid.map(r=>String(r[colorBy]??'missing')))];
 return svg+`<p class="chart-caption">Color: ${escape(colorBy)} · ${values.map(v=>`<span style="color:${color(v)}">${escape(v)}</span>`).join(' · ')}. Circle: cold/unspecified; square: warm. Published comparisons use hollow marks. ${zeroRows.length} exact zeros occupy the labelled separate lane; ${rows.filter(r=>!finite(r[yKey])).length} missing values are not plotted.</p>`;
}
export function speed(rows,group){
 const groups=new Map();for(const r of rows){const k=String(r[group]??'missing');if(!groups.has(k))groups.set(k,[]);groups.get(k).push(r)}
 return `<p class="chart-caption">Grouped by ${escape(group)}; each group has its own labelled y scale. Thread count determines color; cold and warm observations have distinct marks. Expand any group.</p>`+[...groups].map(([k,rs],i)=>`<details ${i<3?'open':''}><summary>${escape(k)} · ${rs.length} observations</summary>${plot(rs.map(r=>({...r,pixels:r.width*r.height,ms_per_MP:finite(r.ms)?r.ms/(r.width*r.height/1e6):null})),'pixels','ms_per_MP',{logX:true})}</details>`).join('');
}
export function agreement(rows){
 const unconfirmed=[...new Set(rows.filter(r=>r.reproduction==='reproduction unconfirmed').map(r=>r.metric))];
 const precision=[...new Set(rows.map(r=>r.oracle_precision).filter(Boolean))];
 return (unconfirmed.length?`<p class="warning">Published reproduction unconfirmed: ${unconfirmed.map(escape).join(', ')}.</p>`:'')+
 `<p class="chart-caption">Oracle print precision: ${precision.map(escape).join(' · ')||'not recorded'}. A delta from a rounded oracle is limited by its printed precision; small residuals do not establish algorithmic agreement.</p>`+
 plot(rows,'pair','abs_delta',{logY:true,categorical:true,colorBy:'metric'})+plot(rows,'pair','relative_delta',{logY:true,categorical:true,colorBy:'metric'});
}
const workFields=['metric','implementation','implementation_version','host','os','backend','engine','mode','phase','content','dtype','tf32_conv'];
export function threadScaling(rows){
 const groups=new Map();for(const row of rows.filter(r=>finite(r.ms)&&r.ms>0&&Number(r.threads)>0)){
  const group=key(row,[...workFields,'width','height']);if(!groups.has(group))groups.set(group,[]);groups.get(group).push(row)
 }
 const result=[];
 for(const [group,rs] of groups){const times=new Map();for(const r of rs){const t=Number(r.threads);if(!times.has(t))times.set(t,[]);times.get(t).push(r.ms)}
  if(!times.has(1)||times.size<2)continue;
  const baseline=median(times.get(1));for(const [threads,values] of times)result.push({group,threads,speedup:baseline/median(values),samples:values.length,source:[...new Set(rs.map(r=>r.source))].join(';')});
 }
 return result;
}
export function regression(rows){
 if(rows.length<2)return null;
 const xs=rows.map(r=>r.pixels),ys=rows.map(r=>r.ms),n=xs.length,mx=xs.reduce((a,b)=>a+b,0)/n,my=ys.reduce((a,b)=>a+b,0)/n;
 const d=xs.reduce((s,x)=>s+(x-mx)**2,0);if(!d)return null;
 const beta=xs.reduce((s,x,i)=>s+(x-mx)*(ys[i]-my),0)/d,alpha=my-beta*mx;
 const residual=xs.reduce((sum,x,i)=>sum+(ys[i]-alpha-beta*x)**2,0),total=ys.reduce((sum,y)=>sum+(y-my)**2,0);
 return {alpha,beta,n,r2:total?1-residual/total:null};
}
export function costFits(rows){
 const groups=new Map();for(const row of rows.filter(r=>finite(r.ms)&&r.width>0&&r.height>0)){
  const group=key(row,[...workFields,'threads']);if(!groups.has(group))groups.set(group,[]);groups.get(group).push(row)
 }
 return [...groups].map(([group,rs])=>{const sizes=new Map();for(const r of rs){const pixels=r.width*r.height;if(!sizes.has(pixels))sizes.set(pixels,[]);sizes.get(pixels).push(r.ms)}
  const medians=[...sizes].map(([pixels,values])=>({pixels,ms:median(values),repeats:values.length}));
  return {group,small_n:medians.filter(r=>r.pixels<=512**2).length,large_n:medians.filter(r=>r.pixels>=1024**2).length,small:regression(medians.filter(r=>r.pixels<=512**2)),large:regression(medians.filter(r=>r.pixels>=1024**2)),observations:rs.length,distinct_sizes:medians.length,sources:[...new Set(rs.map(r=>r.source))]};
 });
}
export function fitPanels(rows){
 return '<h3>Setup intercept & pixel cost</h3><p class="chart-caption">Median per measured size across sources, within an identical workload. Each size has one OLS vote. n = distinct sizes; R² refers to each separate fitted range. No interpolated points are drawn.</p>'+costFits(rows).map(f=>`<details><summary>${escape(f.group)} · ${f.observations} observations / ${f.distinct_sizes} sizes</summary><div class="fit">α ≤512²: ${f.small?fmt(f.small.alpha)+' ms; n='+f.small.n+'; R²='+fmt(f.small.r2)+(f.small.alpha<0?' — NEGATIVE α: model limitation':''):'missing: n='+f.small_n+'; need 2 distinct small sizes'}<br>β ≥1024²: ${f.large?fmt(f.large.beta*1e6)+' ms/MP; n='+f.large.n+'; R²='+fmt(f.large.r2):'missing: n='+f.large_n+'; need 2 distinct large sizes'}<br>${f.sources.map(escape).join('<br>')}</div></details>`).join('');
}
