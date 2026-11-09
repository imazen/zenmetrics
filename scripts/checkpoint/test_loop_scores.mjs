import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {scoreTraceIterations} from '../../web/metrics/loop-scores.js';

const trace = () => ({width: 2, height: 1, source: new Uint8Array([1,2,3,4,5,6]),
  frames: [{recon: new Uint8Array([7,8,9,10,11,12])}, {recon: new Uint8Array([13,14,15,16,17,18])}]});
const coverage = JSON.parse(await readFile(new URL('../../web/metrics/coverage.json', import.meta.url)));
const runnable = coverage.metrics.filter(m => m.status !== 'unavailable');
for (const {name} of runnable) {
  const calls = [], original = trace(), rows = [];
  const result = await scoreTraceIterations(original, name, async (metric, reference, candidate, w, h) => {
    assert.equal(metric, name); assert.equal(w, 2); assert.equal(h, 1);
    assert.deepEqual([...reference], [1,2,3,255,4,5,6,255]);
    assert.equal(candidate[3], 255); assert.equal(candidate[7], 255);
    calls.push({reference, candidate});
    return {metric, values: [{column:'candidate',value:candidate[0]}, {column:'reference',value:reference[0]}]};
  }, {onResult: row => rows.push(row)});
  assert.equal(calls.length, 2); assert.equal(rows.length, 2);
  assert.equal(result.stopped, false);
  assert.equal(calls[0].reference, calls[1].reference); // Convert the source once.
  assert.equal(rows[0].score.values[0].value, 7); assert.equal(rows[1].score.values[0].value, 13);
  assert.deepEqual([...original.source], [1,2,3,4,5,6]);
}
{
  let calls = 0;
  const result = await scoreTraceIterations(trace(), 'nr-control', async () => {
    if (++calls === 1) throw new Error('unsupported geometry');
    return {values:[{column:'nr', value:17}]};
  });
  assert.equal(result.rows[0].error, 'unsupported geometry');
  assert.equal(result.rows[0].score, undefined);
  assert.equal(result.rows[1].score.values[0].value, 17);
}
{
  let cancelled = false, calls = 0;
  const result = await scoreTraceIterations(trace(), 'async-control', async () => {
    calls++; cancelled = true; return {score:123};
  }, {isCancelled: () => cancelled, onResult: () => assert.fail('stale result published')});
  assert.equal(result.stopped, true); assert.equal(result.rows.length, 0); assert.equal(calls, 1);
}
{
  const input = trace(); input.frames[0].recon = new Uint8Array(2);
  let calls = 0;
  const result = await scoreTraceIterations(input, 'control', async () => { calls++; return {score:1}; });
  assert.match(result.rows[0].error, /geometry/); assert.equal(calls, 1);
  await assert.rejects(scoreTraceIterations({...trace(), width:0}, 'control', () => assert.fail()), /geometry/);
}
console.log(`PASS loop trace routing for ${runnable.length} runnable catalog entries; native multi-score retention, failures, cancellation and byte geometry`);
