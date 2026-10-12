const $ = id => document.getElementById(id);
const worker = new Worker(new URL('./worker.js', import.meta.url), {type:'module'});
let reference, distorted, lastMap, nextId = 0, currentId = 0, busy = false, initialized = false;
function status(message, error = false) { $('status').textContent = message; $('status').className = error ? 'error' : ''; }
async function pixels(file) {
  const bitmap = await createImageBitmap(file, {colorSpaceConversion:'default'});
  const canvas = document.createElement('canvas'); canvas.width = bitmap.width; canvas.height = bitmap.height;
  const context = canvas.getContext('2d', {colorSpace:'srgb', willReadFrequently:true});
  context.drawImage(bitmap, 0, 0); bitmap.close();
  const image = context.getImageData(0, 0, canvas.width, canvas.height);
  if (image.data.some((v,i) => i % 4 === 3 && v !== 255)) throw new Error('Transparent input requires an explicit composite before comparison');
  return image;
}
function paint() {
  if (!reference || !lastMap) return;
  const scale = Number($('scale').value), opacity = Number($('opacity').value);
  if (!(scale > 0 && Number.isFinite(scale))) { status('Map maximum must be positive', true); return; }
  const canvas = $('map'); canvas.width = reference.width; canvas.height = reference.height;
  const out = new ImageData(new Uint8ClampedArray(reference.data), reference.width, reference.height);
  for (let p = 0; p < lastMap.length; p++) {
    const value = Math.max(0, Math.min(1, lastMap[p] / scale));
    for (let c = 0; c < 3; c++) out.data[p * 4 + c] = out.data[p * 4 + c] * (1-opacity) + (c < 2 ? value * 255 : 0) * opacity;
  }
  canvas.getContext('2d').putImageData(out, 0, 0);
}
for (const name of ['reference','distorted']) $(name).onchange = async () => {
  currentId = ++nextId; lastMap = null;
  try {
    const image = await pixels($(name).files[0]);
    if (name === 'reference') {
      reference = image; $('source').width = image.width; $('source').height = image.height;
      $('source').getContext('2d').putImageData(image, 0, 0);
    } else distorted = image;
    status('Image loaded. Compare when both images are ready.');
  } catch(error) { status(String(error), true); }
};
$('compare').onclick = () => {
  if (busy) return; // Single in-flight comparison; controls never enqueue unbounded work.
  if (!reference || !distorted || reference.width !== distorted.width || reference.height !== distorted.height) {status('Load two images with identical dimensions', true); return;}
  busy = true; $('compare').disabled = true; currentId = ++nextId;
  status('Computing in the worker…');
  worker.postMessage({id:currentId, metric:$('metric').value, reference:reference.data, distorted:distorted.data, width:reference.width,height:reference.height});
};
worker.onmessage = ({data}) => {
  if (data.type === 'progress') { status(data.message); return; }
  if (data.type === 'ready') { initialized = true; $('compare').disabled = false; status(`Ready: ${data.threads} Rayon workers, SIMD128 + relaxed SIMD`); return; }
  busy = false; $('compare').disabled = !initialized;
  if (data.id !== undefined && data.id !== currentId) return;
  if (data.type === 'error') { status(data.message,true); return; }
  lastMap = data.map; paint(); status(`${JSON.stringify(data.score,null,2)}\nWorker compute: ${data.milliseconds.toFixed(1)} ms`);
};
worker.onerror = error => {busy = false; $('compare').disabled = true; status(`Worker failed: ${error.message}`,true);};
$('scale').oninput = paint; $('opacity').oninput = paint;
if (!crossOriginIsolated || typeof SharedArrayBuffer === 'undefined') {
  status('Serve over HTTPS with COOP: same-origin and COEP: require-corp. Threaded WASM cannot start.', true);
} else worker.postMessage({type:'init',threads:Math.max(1,Math.min(8,navigator.hardwareConcurrency || 1))});
