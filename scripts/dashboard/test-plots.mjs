import assert from 'node:assert/strict';
import {costFits,threadScaling,agreement} from '../../web/dashboard/plots.js';
const base={metric:'m',implementation:'ours',host:'i265',threads:1,backend:'CPU',engine:'native',mode:'score',phase:'cold',os:'Linux',content:'photo'};
const rows=[64,128,256,1024,2048].flatMap(size=>[
 {...base,width:size,height:size,ms:2+size*size/1e6,source:'first.tsv'},
 {...base,width:size,height:size,ms:2+size*size/1e6,source:'second.tsv'},
 {...base,width:size,height:size,ms:10000,source:'outlier.tsv'}]);
const fit=costFits(rows);assert.equal(fit.length,1);assert.equal(fit[0].small.n,3);assert.equal(fit[0].large.n,2);
assert.ok(Math.abs(fit[0].small.alpha-2)<1e-12);assert.ok(Math.abs(fit[0].large.beta-1e-6)<1e-16);assert.equal(fit[0].large.r2,1);
const scaling=threadScaling([{...base,width:64,height:64,ms:10},{...base,width:64,height:64,threads:8,ms:2},{...base,width:128,height:128,ms:100},{...base,width:128,height:128,threads:8,ms:50}]);
assert.deepEqual(scaling.filter(r=>r.threads===8).map(r=>r.speedup).sort((a,b)=>a-b),[2,5]);
const panel=agreement([{metric:'m',pair:'real-pair',series:'oracle',abs_delta:0,relative_delta:0,reproduction:'reproduction unconfirmed',oracle_precision:'7 digits'}]);
assert.ok(panel.includes('unconfirmed'));assert.ok(panel.includes('7 digits'));assert.ok(panel.includes('Exact zero'));assert.ok(!panel.includes('Missing:'));
console.log('Cross-source size medians, coefficients, R², per-size scaling and agreement annotations pass');

const {selectRanking,chooseDefault,rankingBars}=await import('../../web/dashboard/ranking.js');
const timing=(metric,ms,extra={})=>({metric,ms,width:512,height:512,host:'i265',backend:'CPU',threads:1,mode:'score',implementation:'zenmetrics',source:'one',...extra});
const rankRows=[timing('a',4),timing('b',2),timing('a',1,{threads:8}),timing('c',.5,{width:1024}),timing('d',0),timing('e',5,{implementation:'libjxl v0.12',source:'two',p10_ms:4,p90_ms:6})];
const params=new URLSearchParams({host:'i265',device:'CPU',threads:'1',mode:'score-only',workload:'backend=CPU',impl:'competitors',source:''});
assert.equal(selectRanking(rankRows,params).length,4);
assert.equal(chooseDefault(rankRows,{one:{date:'2026-10-17'}}).width,512);
const bars=rankingBars(selectRanking(rankRows,params).filter(r=>r.width===512),'ms',{});
assert(bars.indexOf('data-metric="b"')<bars.indexOf('data-metric="a"'));
assert(bars.includes('whisker')&&bars.includes('◆')&&bars.includes('log scale'));
assert(!bars.includes('data-metric="c"')&&!bars.includes('data-metric="d"'));
console.log('Exact-cell ranking, sorting, competitor marks and whiskers pass');

const pairedRows=[timing('butteraugli',3,{comparison_group:'Shared RGB8 pairs',os:'Linux-7',backend:'cpu',engine:'native'}),timing('butteraugli',4,{comparison_group:'Shared RGB8 pairs',implementation:'libjxl v0.12',os:'Linux',backend:'CPU'})];
assert.equal(selectRanking(pairedRows,new URLSearchParams({...Object.fromEntries(params),workload:'Shared RGB8 pairs'})).length,2);

// Build versions must not share scaling baselines, fits or rank identities.
const versioned=[timing('versioned',10,{implementation_version:'build-a'}),timing('versioned',2,{threads:8,implementation_version:'build-b'})];
assert.equal(threadScaling(versioned).length,0);
assert.equal(costFits(versioned).length,2);
const versionBars=rankingBars(versioned,'ms',{});
assert(versionBars.includes('build-a')&&versionBars.includes('build-b'));
const {slope}=await import('../../web/dashboard/ranking.js');
assert(slope([versioned[0],{...versioned[1],width:1024}], '512×512','1024×512').includes('0 shared entries'));
console.log('Implementation versions remain distinct in scaling, fits, labels and rank changes');

const {bars:barChart,speed:speedChart}=await import('../../web/dashboard/plots.js');
const barSvg=barChart([{label:'slow',value:10,lo:8,hi:12,n:3},{label:'fast',value:1,lo:1,hi:1,n:1},{label:'zero',value:0}],{whiskers:true,label:'Time'});
assert(barSvg.indexOf('data-label="fast"')<barSvg.indexOf('data-label="slow"'));
assert.equal((barSvg.match(/class="whisker"/g)||[]).length,1);
assert(barSvg.includes('Exact zero (1)')&&!barSvg.includes('<circle'));
assert(!barChart([{label:'a',value:2,lo:1,hi:3}],{whiskers:false}).includes('whisker"'));
const sp=speedChart([{metric:'m',implementation:'z',width:1000,height:1000,threads:1,mode:'lat',phase:'na',host:'i265',ms:2,source:'s'},{metric:'m',implementation:'z',width:1000,height:1000,threads:1,mode:'lat',phase:'na',host:'i265',ms:4,source:'s'}],'metric',{whiskers:true});
assert(sp.includes('1000×1000')&&sp.includes('3.00')&&sp.includes('n=2')&&!sp.includes('<circle'));
console.log('Bars sort ascending, whiskers only on spread, exact zeros labelled, speed bars use real dimensions');
