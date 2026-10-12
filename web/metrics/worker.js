import init, { initThreadPool, Reference, thread_count } from './pkg/zenmetrics_browser.js';
let ready = false, cached, previousReference;
self.onmessage = async ({data}) => {
  try {
    if (data.type === 'init') {
      if (!self.crossOriginIsolated || typeof SharedArrayBuffer === 'undefined') {
        throw new Error('Threaded WASM requires cross-origin isolation and SharedArrayBuffer');
      }
      postMessage({type:'progress', message:'Instantiating WASM…'});
      const relaxedProbe = new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,15,1,13,0,65,0,253,15,65,0,253,15,253,128,2,11]);
      if (!WebAssembly.validate(relaxedProbe)) throw new Error('Relaxed SIMD128 is required');
      await init(); // Compilation rejects unsupported SIMD/relaxed SIMD; no alternate module.
      postMessage({type:'progress', message:'Starting Rayon workers…'});
      let timer;
      try {
        await Promise.race([initThreadPool(data.threads), new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error('Rayon worker initialization deadline expired')), 15000);
        })]);
      } finally { clearTimeout(timer); }
      ready = true;
      if (thread_count() !== data.threads) throw new Error('Rayon pool size differs from the request');
      postMessage({type: 'ready', threads: thread_count()});
      return;
    }
    if (!ready) throw new Error('WASM worker has not initialized');
    const start = performance.now();
    const {reference, distorted, width, height, metric, id} = data;
    if (metric !== 'zensim') throw new Error('Browser module supports zensim; full registry is in native checkpoint');
    if (!previousReference || previousReference.width !== width || previousReference.height !== height ||
        previousReference.bytes.length !== reference.length || !reference.every((v,i) => v === previousReference.bytes[i])) {
      const prepared = new Reference(reference, width, height, width * 4);
      if (cached) cached.free();
      cached = prepared;
      previousReference = {width, height, bytes: reference.slice()};
    }
    const compared = cached.compare(distorted, width * 4);
    const score = {metric:'zensim', profile:'codec_target', backend:'native-cpu-wasm', score: compared.score};
    const map = compared.map; compared.free();
    postMessage({type: 'result', id, score, map, milliseconds: performance.now() - start}, [map.buffer]);
  } catch (error) {
    postMessage({type: 'error', id: data.id, message: String(error)});
  }
};
