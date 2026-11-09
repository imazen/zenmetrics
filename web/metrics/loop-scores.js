// Score the trace's stored sRGB8 images. These observations do not change the
// encoder's steering map, target, quantizer, native float pixels or file bytes.
function rgba(rgb, pixels) {
  if (!(rgb instanceof Uint8Array) || rgb.length !== pixels * 3)
    throw new Error('Trace RGB storage does not match its geometry');
  const out = new Uint8Array(pixels * 4);
  for (let i = 0; i < pixels; i++) {
    out.set(rgb.subarray(i * 3, i * 3 + 3), i * 4);
    out[i * 4 + 3] = 255;
  }
  return out;
}

export async function scoreTraceIterations(trace, metric, score, {
  onResult = () => {}, isCancelled = () => false,
} = {}) {
  const {width, height, source, frames} = trace;
  const pixels = width * height;
  if (!Number.isSafeInteger(pixels) || !Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1)
    throw new Error('Invalid trace geometry');
  const reference = rgba(source, pixels), rows = [];
  for (const [index, frame] of frames.entries()) {
    if (isCancelled()) return {rows, stopped: true};
    let row;
    try {
      const result = await score(metric, reference, rgba(frame.recon, pixels), width, height);
      row = {index, metric, score: result};
    } catch (error) {
      row = {index, metric, error: String(error?.message ?? error)};
    }
    // A trace/metric change must not publish a stale asynchronous result.
    if (isCancelled()) return {rows, stopped: true};
    rows.push(row);
    onResult(row);
  }
  return {rows, stopped: false};
}
