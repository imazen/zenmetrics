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
