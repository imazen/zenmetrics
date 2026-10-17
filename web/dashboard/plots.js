export const finite = x => typeof x === 'number' && Number.isFinite(x);
export const fmt = x => finite(x) ? x.toPrecision(5) : 'missing';
export const escape = v => String(v ?? 'missing').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const palette=['#12775f','#bf5825','#675bc4','#2179ac','#b34e83','#66822c','#775840','#248687'];
const color = value => palette[[...String(value)].reduce((s,c)=>s+c.charCodeAt(0),0)%palette.length];
const median = values => {const a=values.toSorted((a,b)=>a-b),m=Math.floor(a.length/2);return a.length%2?a[m]:(a[m-1]+a[m])/2};
const key = (row,fields) => fields.map(k=>row[k]??'missing').join(' / ');
export function bars(items,{unit='',log=true,whiskers=false,label='Value',caption=''}={}){
 const valid=items.filter(i=>finite(i.value)&&(!log||i.value>0)).sort((a,b)=>a.value-b.value);
 const zeros=log?items.filter(i=>i.value===0):[],missing=items.filter(i=>!finite(i.value)).length;
 if(!valid.length&&!zeros.length)return '<p class="chart-caption">Missing: no finite observations for this selection.</p>';
 const spread=i=>whiskers&&finite(i.lo)&&finite(i.hi)&&i.hi>i.lo&&(!log||i.lo>0);
 const vals=valid.flatMap(i=>spread(i)?[i.value,i.lo,i.hi]:[i.value]);
 let low=log?Math.min(...vals)/1.3:Math.min(0,...vals),high=Math.max(...vals,low+1e-300)*(log?1.3:1.05);if(high===low)high=low+1;
 const x=v=>310+430*(log?(Math.log(v)-Math.log(low))/(Math.log(high)-Math.log(low)):(v-low)/(high-low));
 const rowH=26,h=45+(valid.length+(zeros.length?1:0))*rowH+zeros.length*rowH;
 const ticks=[0,.25,.5,.75,1].map(t=>log?Math.exp(Math.log(low)+t*(Math.log(high)-Math.log(low))):low+t*(high-low));
 const body=valid.map((i,k)=>{const y=40+k*rowH;return `<g class="bar" data-label="${escape(i.label)}" data-value="${i.value}"><title>${escape(i.title||i.label)}</title><text x="0" y="${y+13}">${escape(i.label.length>46?i.label.slice(0,45)+'…':i.label)}</text><rect x="310" y="${y}" width="${Math.max(1,x(i.value)-310)}" height="17" fill="${i.color||color(i.label)}"/>${spread(i)?`<path class="whisker" d="M${x(i.lo)} ${y+8}H${x(i.hi)}m0 -5v10M${x(i.lo)} ${y+3}v10" stroke="currentColor" fill="none"/>`:''}<text x="${Math.min(800,x(spread(i)?i.hi:i.value)+5)}" y="${y+13}">${i.value.toPrecision(3)}${i.n>1?' · n='+i.n:''}</text></g>`}).join('');
 const zeroLane=zeros.length?`<text x="0" y="${52+valid.length*rowH}">Exact zero (${zeros.length})</text>`+zeros.map((i,k)=>`<g class="bar zero"><title>${escape(i.title||i.label)}</title><text x="310" y="${52+(valid.length+1+k)*rowH}">${escape(i.label)} = 0</text></g>`).join(''):'';
 return `<svg viewBox="0 0 850 ${h}" style="min-width:700px" role="img" aria-label="${escape(label)} bar chart${log?', logarithmic axis':''}"><text x="310" y="14">${escape(label)}${unit?' ('+escape(unit)+')':''} · ${log?'log scale':'linear scale'} →</text>${ticks.map(v=>`<path d="M${x(v)} 22V${40+valid.length*rowH}" stroke="currentColor" opacity=".15"/><text x="${x(v)}" y="32">${v.toPrecision(2)}</text>`).join('')}${body}${zeroLane}</svg><p class="chart-caption">${caption} ${whiskers?'Whiskers: observed minimum to maximum across repeats and sources within the exact condition.':'Bars only; switch to “Bars + whiskers” to show the spread.'} ${missing} missing values are not drawn.</p>`;
}
const quantiles=values=>{const a=values.toSorted((a,b)=>a-b);return {median:median(a),lo:a[0],hi:a.at(-1),n:a.length}};
export function aggregate(rows,keyFields,valueOf){
 const groups=new Map();for(const r of rows){const v=valueOf(r);if(!finite(v))continue;const k=keyFields.map(f=>r[f]).filter(v=>v!=null&&v!=='missing'&&v!=='').join(' · ')||'not recorded';if(!groups.has(k))groups.set(k,{rows:[],values:[]});const g=groups.get(k);g.rows.push(r);g.values.push(v)}
 return [...groups].map(([k,g])=>{const q=quantiles(g.values);return {label:k,value:q.median,lo:q.lo,hi:q.hi,n:q.n,rows:g.rows,title:k+'\nmedian '+q.median.toPrecision(4)+' · min '+q.lo.toPrecision(4)+' · max '+q.hi.toPrecision(4)+' · n='+q.n+'\nsources: '+[...new Set(g.rows.map(r=>r.source))].join(', ')}});
}
export const dims=r=>finite(Number(r.width))&&finite(Number(r.height))?`${Number(r.width)}×${Number(r.height)}`:'size not recorded';
export function speed(rows,group,{whiskers=false,unit='ms/MP'}={}){
 const groups=new Map();for(const r of rows){const k=String(r[group]??'missing');if(!groups.has(k))groups.set(k,[]);groups.get(k).push(r)}
 const value=r=>finite(r.ms)&&r.width>0&&r.height>0?(unit==='ms/MP'?r.ms/(r.width*r.height/1e6):r.ms):null;
 return `<p class="chart-caption">Grouped by ${escape(group)}. Each bar is the median of one exact condition (metric · implementation · resolution · threads · mode · phase · host). Timings are provisional; final numbers are pending cleaner controlled runs.</p>`+[...groups].map(([k,rs],i)=>`<details ${i<3?'open':''}><summary>${escape(k)} · ${rs.length} observations</summary>${bars(aggregate(rs.map(r=>({...r,size_dims:dims(r)})),['metric','implementation','size_dims','threads','mode','phase','host'],value),{unit,whiskers,label:'Time'})}</details>`).join('');
}
export function agreement(rows,{whiskers=false}={}){
 const unconfirmed=[...new Set(rows.filter(r=>r.reproduction==='reproduction unconfirmed').map(r=>r.metric))];
 const precision=[...new Set(rows.map(r=>r.oracle_precision).filter(Boolean))];
 const items=field=>aggregate(rows,['metric','series'],r=>r[field]).map(i=>({...i,label:i.label+' · median over '+i.n+' pairs'}));
 return (unconfirmed.length?`<p class="warning">Published reproduction unconfirmed: ${unconfirmed.map(escape).join(', ')}.</p>`:'')+
 `<p class="chart-caption">Oracle print precision: ${precision.map(escape).join(' · ')||'not recorded'}. A delta from a rounded oracle is limited by its printed precision; small residuals do not establish algorithmic agreement. Bars show the median delta per metric and series; exact zeros are listed in a separate lane.</p>`+
 '<h3>Absolute delta</h3>'+bars(items('abs_delta'),{whiskers,label:'|ours − reference|'})+'<h3>Relative delta</h3>'+bars(items('relative_delta'),{whiskers,label:'relative delta'});
}
export function scalingBars(rows,{whiskers=false}={}){
 return bars(threadScaling(rows).map(r=>({label:r.group.split(' / ').filter(x=>x!=='missing').join(' · ')+' · '+r.threads+'T',value:r.speedup,n:r.samples,title:r.group+'\n'+r.threads+' threads: speedup '+r.speedup.toPrecision(4)+' vs 1 thread (medians)\n'+r.source})).filter(i=>!i.label.endsWith(' 1T')),{log:false,whiskers:false,label:'Speedup vs one thread',caption:'Median time at one thread divided by median time at N threads, within one exact workload and size.'});
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
