// Perf-only view: one row per metric for the selected CPU, image size, thread count and phase.
// Data: window.PERF (brief_data.py canonical_perf) and window.KINDS (metric -> [kind, origin]).
// Row glyphs: box = 95 % interval of the mean, whiskers = spread (warm: p10-p90 of the rounds;
// cold: min-max of the fresh-process runs), vertical tick = median, tiny dots = single cold runs.
(function () {
  const P = window.PERF, K = window.KINDS || {};
  const root = document.getElementById('perfchart');
  if (!P || !root) return;
  const F = Object.fromEntries(P.fields.map((f, i) => [f, i]));
  const T975 = [0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131];
  const st = { host: Object.keys(P.hosts).includes('i265') ? 'i265' : Object.keys(P.hosts)[0], size: 'aic-small', phase: 'warm' };
  if (!P.sizes.some(s => s.name === st.size)) st.size = P.sizes[0].name;
  const threads = () => document.documentElement.classList.contains('t8') ? 8 : 1;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const f3 = v => v >= 1000 ? Math.round(v).toLocaleString('en') : v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toPrecision(2);
  const tick = v => v >= 1e6 ? v / 1e6 + 'M' : v >= 1e3 ? v / 1e3 + 'k' : String(+v.toPrecision(3));

  function stats(r) {
    if (st.phase === 'warm') {
      const med = r[F.median_ms], mean = r[F.mean_ms] ?? med, p10 = r[F.p10_ms], p90 = r[F.p90_ms], n = r[F.n] || 0;
      if (med == null) return null;
      let ci = null;
      if (p10 != null && p90 != null && n > 1) {           // spread inferred from p10-p90 (normal approximation)
        const sd = (p90 - p10) / 2.5631, h = (T975[n - 1] || 1.96) * sd / Math.sqrt(n);
        ci = [mean - h, mean + h];
      }
      return { c: med, mean, ci, lo: p10 ?? med, hi: p90 ?? med, n, dots: [], what: 'p10–p90', approx: true };
    }
    const v = (r[F.cold_ms] || []).slice().sort((a, b) => a - b);
    if (!v.length) return null;
    const n = v.length, mean = v.reduce((a, b) => a + b, 0) / n, med = n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
    let ci = null;
    if (n > 1) {
      const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)), h = (T975[n - 1] || 1.96) * sd / Math.sqrt(n);
      ci = [Math.max(mean - h, v[0] / 2), mean + h];
    }
    return { c: med, mean, ci, lo: v[0], hi: v[n - 1], n, dots: v, what: 'range', approx: false };
  }

  function kmeans(xs, k) {                                     // exact 1-D k-means on sorted xs
    const n = xs.length, pre = [0], pre2 = [0];
    xs.forEach(x => { pre.push(pre.at(-1) + x); pre2.push(pre2.at(-1) + x * x); });
    const cost = (i, j) => (pre2[j] - pre2[i]) - (pre[j] - pre[i]) ** 2 / (j - i);
    const D = [...Array(k + 1)].map(() => Array(n + 1).fill(Infinity)), A = [...Array(k + 1)].map(() => Array(n + 1).fill(0));
    D[0][0] = 0;
    for (let c = 1; c <= k; c++) for (let j = c; j <= n; j++) for (let i = c - 1; i < j; i++) {
      const v = D[c - 1][i] + cost(i, j); if (v < D[c][j]) { D[c][j] = v; A[c][j] = i; }
    }
    const lab = Array(n).fill(0); let j = n;
    for (let c = k; c > 0; c--) { const i = A[c][j]; for (let t = i; t < j; t++) lab[t] = c - 1; j = i; }
    return [lab, D[k][n]];
  }

  function niceStep(span, target) {
    const raw = span / target, mag = 10 ** Math.floor(Math.log10(raw));
    for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * mag) return m * mag;
    return 10 * mag;
  }

  function render() {
    const th = threads();
    const items = P.rows.filter(r => r[F.host] === st.host && r[F.size] === st.size && r[F.threads] === th && r[F.phase] === st.phase)
      .map(r => ({ m: r[F.metric], s: stats(r) })).filter(x => x.s && x.s.c > 0).sort((a, b) => a.s.c - b.s.c);
    const info = document.getElementById('perfinfo');
    if (!items.length) { root.innerHTML = '<p class="note">No qualified observations for this selection yet.</p>'; info.textContent = ''; return; }
    const xs = items.map(x => Math.log10(x.s.c));
    const total = kmeans(xs, 1)[1] || 1;
    let k = 1, lab = xs.map(() => 0), frac = 1;
    for (let kk = 2; kk <= Math.min(7, items.length); kk++) { const [l, ss] = kmeans(xs, kk); k = kk; lab = l; frac = ss / total; if (ss <= 0.06 * total) break; }
    const groups = [...Array(k)].map((_, c) => items.filter((_, i) => lab[i] === c));
    const W = 1310, NAME = 262, TXT = 272, LX0 = 530, LX1 = 890, RX0 = 915, RX1 = 1295, rowh = 19, head = 32, foot = 28;
    const allLo = Math.min(...items.map(x => Math.min(x.s.lo, x.s.ci ? x.s.ci[0] : x.s.c))), allHi = Math.max(...items.map(x => Math.max(x.s.hi, x.s.ci ? x.s.ci[1] : x.s.c)));
    const lo = 10 ** Math.floor(Math.log10(allLo)), hi = 10 ** Math.ceil(Math.log10(allHi));
    const X = v => LX0 + (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * (LX1 - LX0);
    const decades = []; for (let e = Math.log10(lo); e <= Math.log10(hi) + 1e-9; e++) decades.push(10 ** e);
    const minors = decades.flatMap(d => [2 * d, 5 * d]).filter(v => v > lo && v < hi);
    const H = 26 + groups.reduce((a, g) => a + head + rowh * g.length + foot + 10, 0);
    const p = [`<text class="pl-title" x="${TXT}" y="16">n · 95 % CI · ${st.phase === 'warm' ? 'p10–p90' : 'range'} (ms)</text><text class="pl-title" x="${LX0}" y="16">all metrics, log axis (ms per call)</text><text class="pl-title" x="${RX0}" y="16">within cluster, linear axis (ms)</text>`];
    let y = 26, overlap = 0;
    groups.forEach((g, gi) => {
      const gmax = Math.max(...g.map(x => Math.max(x.s.hi, x.s.ci ? x.s.ci[1] : 0, x.s.c))) * 1.05;
      const step = niceStep(gmax, 6), RX = v => RX0 + v / gmax * (RX1 - RX0);
      const y0 = y + head, y1 = y0 + rowh * g.length;
      p.push(`<rect class="pl-cluster" x="4" y="${y}" width="${W - 8}" height="${y1 + foot - y}" rx="4"/><text class="pl-row" x="12" y="${y + 21}" font-weight="600">Cluster ${gi + 1}: ${f3(g[0].s.c)}–${f3(g.at(-1).s.c)} ms, ${g.length} metrics</text>`);
      minors.forEach(v => p.push(`<line class="pl-vgrid-minor" x1="${X(v)}" y1="${y0 - 4}" x2="${X(v)}" y2="${y1}"/>`));
      decades.forEach(v => p.push(`<line class="pl-vgrid" x1="${X(v)}" y1="${y0 - 4}" x2="${X(v)}" y2="${y1}"/><text class="pl-tick" x="${X(v)}" y="${y1 + 16}" text-anchor="middle">${tick(v)}</text>`));
      for (let v = 0; v <= gmax + 1e-12; v += step) p.push(`<line class="pl-vgrid" x1="${RX(v)}" y1="${y0 - 4}" x2="${RX(v)}" y2="${y1}"/><text class="pl-tick" x="${RX(v)}" y="${y1 + 16}" text-anchor="middle">${tick(+v.toPrecision(6))}</text>`);
      p.push(`<line class="pl-axisline" x1="${LX0}" y1="${y1}" x2="${LX1}" y2="${y1}"/><line class="pl-axisline" x1="${RX0}" y1="${y1}" x2="${RX1}" y2="${y1}"/>`);
      let prev = null;
      g.forEach((x, i) => {
        const yy = y0 + rowh * i + rowh / 2, s = x.s, kind = K[x.m] || ['classic', 'public'];
        const cls = kind[0].startsWith('nn') ? 'pt-nn' : 'pt-cl';
        const ci = s.ci ? `${f3(s.ci[0])}–${f3(s.ci[1])}${s.approx ? '≈' : ''}` : '–';
        const tip = `${x.m}: median ${f3(s.c)} ms, mean ${f3(s.mean)} ms, n=${s.n}, 95 % CI ${ci}, ${s.what} ${f3(s.lo)}–${f3(s.hi)} ms`;
        p.push(`<text class="perf-lab" x="${NAME}" y="${yy + 4}" text-anchor="end">${esc(x.m)}</text><text class="perf-ci" x="${TXT}" y="${yy + 4}">n=${s.n} · ${ci} · ${f3(s.lo)}–${f3(s.hi)}</text>`);
        for (const XX of [X, RX]) {
          const a = XX(s.lo), b = XX(s.hi), c = XX(s.c);
          p.push(`<g><title>${esc(tip)}</title><line class="bw-whisk" x1="${a}" y1="${yy}" x2="${b}" y2="${yy}"/><line class="bw-whisk" x1="${a}" y1="${yy - 5}" x2="${a}" y2="${yy + 5}"/><line class="bw-whisk" x1="${b}" y1="${yy - 5}" x2="${b}" y2="${yy + 5}"/>`);
          if (s.ci) { const q = XX(Math.max(s.ci[0], XX === X ? lo : 0)), r = XX(s.ci[1]); p.push(`<rect class="bw-box ${cls}" x="${Math.min(q, r)}" y="${yy - 5.5}" width="${Math.max(Math.abs(r - q), 1.5)}" height="11"/>`); }
          s.dots.forEach(v => p.push(`<circle class="bw-dot" cx="${XX(v)}" cy="${yy}" r="1.9"/>`));
          if (kind[1] === 'private') p.push(`<rect class="bw-med" x="${c - 3}" y="${yy - 7}" width="6" height="14"/>`);
          else if (kind[1] === 'dvifmish') p.push(`<path class="bw-med" d="M${c},${yy - 8} L${c + 4},${yy} L${c},${yy + 8} L${c - 4},${yy} Z"/>`);
          else p.push(`<line class="bw-med" x1="${c}" y1="${yy - 7}" x2="${c}" y2="${yy + 7}"/>`);
          p.push('</g>');
        }
        if (s.ci && prev && prev[1] >= s.ci[0]) overlap++;
        prev = s.ci;
      });
      y = y1 + foot + 10;
    });
    root.innerHTML = `<svg class="chart perf" viewBox="0 0 ${W} ${H}" role="img" aria-label="CPU time per metric" xmlns="http://www.w3.org/2000/svg">${p.join('')}</svg>`;
    const cov = P.coverage[st.host];
    info.innerHTML = `${esc(P.hosts[st.host])}, ${esc(sizeLabel(st.size))}, ${th} thread${th > 1 ? 's' : ''}, ${st.phase}: ${items.length} metrics in ${k} clusters (${(frac * 100).toFixed(1)} % of log-time variance within clusters); ${overlap} adjacent pairs have overlapping intervals.` +
      (cov ? ` Campaign coverage on this host so far: ${cov.ok.toLocaleString('en')} of ${cov.cells.toLocaleString('en')} cells.` : '') +
      (st.phase === 'warm' ? ' Warm intervals (≈) are approximate: the spread is inferred from the published p10–p90 assuming a normal distribution.' : ' Cold: each dot is one fresh process (constructor plus first evaluation).');
  }

  function sizeLabel(name) { const s = P.sizes.find(s => s.name === name); return s ? `${s.w}×${s.h} (${name})` : name; }

  function group(id, options, key) {
    const el = document.getElementById(id);
    el.innerHTML = options.map(([v, label]) => `<button type="button" data-v="${esc(v)}" aria-pressed="${v === st[key]}">${esc(label)}</button>`).join('');
    el.addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      st[key] = b.dataset.v; el.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); render();
    });
  }
  group('perf-host', Object.entries(P.hosts).map(([h, cpu]) => [h, `${h} · ${cpu}`]), 'host');
  group('perf-size', P.sizes.map(s => [s.name, `${s.w}×${s.h}`]), 'size');
  group('perf-phase', [['warm', 'warm (10 rounds)'], ['cold', 'cold (3 fresh processes)']], 'phase');
  document.addEventListener('threadschange', render);
  render();
})();
