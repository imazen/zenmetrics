// Perf-only view: per metric, a 1-thread row and an 8-thread row for the selected CPU, image size and phase.
// Data: window.PERF (brief_data.py canonical_perf) and window.KINDS (metric -> [kind, origin]).
// Glyphs: whiskers = spread (warm: p10-p90 of the rounds; cold: min-max of the runs), box = 95 % interval
// of the mean, tick = median, 1 px dots = every individual warm round (zenbench batch mean) or cold run.
// Layout: one log-axis block for every metric, then one linear-axis block per cluster, stacked.
(function () {
  const P = window.PERF, K = window.KINDS || {};
  const root = document.getElementById('perfchart');
  if (!P || !root) return;
  const F = Object.fromEntries(P.fields.map((f, i) => [f, i]));
  const T975 = [0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086];
  const st = { host: Object.keys(P.hosts).includes('i265') ? 'i265' : Object.keys(P.hosts)[0], size: 'aic-small', phase: 'warm', view: 'dist' };
  if (!P.sizes.some(s => s.name === st.size)) st.size = P.sizes[0].name;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const f3 = v => v >= 1000 ? Math.round(v).toLocaleString('en') : v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toPrecision(2);
  const tick = v => v >= 1e6 ? v / 1e6 + 'M' : v >= 1e3 ? v / 1e3 + 'k' : String(+v.toPrecision(3));
  const tq = n => T975[n - 1] || 1.96;

  function stats(r) {
    if (!r) return null;
    if (st.phase === 'warm') {
      const med = r[F.median_ms]; if (med == null) return null;
      const rounds = (r[F.rounds_ms] || []).slice();
      const n = rounds.length || r[F.n] || 0;
      let mean = r[F.mean_ms] ?? med, ci = null, approx = true;
      if (rounds.length > 1) {                                   // exact: from the published rounds
        mean = rounds.reduce((a, b) => a + b, 0) / n;
        const sd = Math.sqrt(rounds.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)), h = tq(n) * sd / Math.sqrt(n);
        ci = [mean - h, mean + h]; approx = false;
      } else if (r[F.p10_ms] != null && r[F.p90_ms] != null && n > 1) {   // approximate: spread from p10-p90
        const sd = (r[F.p90_ms] - r[F.p10_ms]) / 2.5631, h = tq(n) * sd / Math.sqrt(n);
        ci = [mean - h, mean + h];
      }
      const fast = rounds.length ? Math.min(...rounds) : (r[F.p10_ms] ?? med), fastExact = rounds.length > 0;
      return { c: med, mean, ci, approx, fast, fastExact, lo: r[F.p10_ms] ?? med, hi: r[F.p90_ms] ?? med, n, dots: rounds, what: 'p10–p90',
               clean: r[F.clean], flagged: r[F.flagged], rss: r[F.rss_gib], simd: r[F.simd], place: r[F.placement], pc: r[F.placement_cpus] };
    }
    const v = (r[F.cold_ms] || []).slice().sort((a, b) => a - b);
    if (!v.length) return null;
    const n = v.length, mean = v.reduce((a, b) => a + b, 0) / n, med = n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
    let ci = null;
    if (n > 1) { const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)), h = tq(n) * sd / Math.sqrt(n); ci = [Math.max(mean - h, v[0] / 2), mean + h]; }
    return { c: med, mean, ci, approx: false, fast: v[0], fastExact: true, lo: v[0], hi: v[n - 1], n, dots: v, what: 'range', rss: r[F.rss_gib] };
  }

  function kmeans(xs, k) {
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

  // geometry (viewBox units)
  const W = 1500, NAME = 330, TXT = 344, AX0 = 720, AX1 = 1480, SUB = 24, HEAD = 40, FOOT = 34, GAP = 14;

  function glyph(p, s, XX, yy, cls, kind, lo) {
    if (st.view === 'fast') {
      const f = XX(Math.max(s.fast, lo));
      if (kind[1] === 'private') p.push(`<rect class="${cls} fast-pt" x="${f - 5}" y="${yy - 5}" width="10" height="10"/>`);
      else if (kind[1] === 'dvifmish') p.push(`<path class="${cls} fast-pt" d="M${f},${yy - 7} L${f + 7},${yy} L${f},${yy + 7} L${f - 7},${yy} Z"/>`);
      else p.push(`<circle class="${cls} fast-pt${s.fastExact ? '' : ' fast-approx'}" cx="${f}" cy="${yy}" r="5"/>`);
      return;
    }
    const a = XX(Math.max(s.lo, lo)), b = XX(s.hi), c = XX(s.c);
    p.push(`<line class="bw-whisk" x1="${a}" y1="${yy}" x2="${b}" y2="${yy}"/><line class="bw-whisk" x1="${a}" y1="${yy - 6}" x2="${a}" y2="${yy + 6}"/><line class="bw-whisk" x1="${b}" y1="${yy - 6}" x2="${b}" y2="${yy + 6}"/>`);
    if (s.ci) { const q = XX(Math.max(s.ci[0], lo)), r = XX(s.ci[1]); p.push(`<rect class="bw-box ${cls}" x="${Math.min(q, r)}" y="${yy - 6.5}" width="${Math.max(Math.abs(r - q), 1.5)}" height="13"/>`); }
    s.dots.forEach(v => p.push(`<rect class="bw-dot" x="${XX(Math.max(v, lo)) - 0.5}" y="${yy - 9.5}" width="1" height="1"/>`));
    if (kind[1] === 'private') p.push(`<rect class="bw-med" x="${c - 3.5}" y="${yy - 8}" width="7" height="16"/>`);
    else if (kind[1] === 'dvifmish') p.push(`<path class="bw-med" d="M${c},${yy - 9} L${c + 5},${yy} L${c},${yy + 9} L${c - 5},${yy} Z"/>`);
    else p.push(`<line class="bw-med" x1="${c}" y1="${yy - 8}" x2="${c}" y2="${yy + 8}"/>`);
  }

  function block(p, y, title, items, XX, ticks, minorTicks, lo, tickFmt) {
    // items: [{m, s1, s8}]; two sub-rows per metric, alternate shading per metric
    const y0 = y + HEAD, y1 = y0 + items.length * 2 * SUB;
    p.push(`<rect class="pl-cluster" x="2" y="${y}" width="${W - 4}" height="${y1 + FOOT - y}" rx="5"/><text class="perf-head" x="14" y="${y + 27}">${esc(title)}</text>`);
    items.forEach((it, i) => { if (i % 2) p.push(`<rect class="perf-zebra" x="4" y="${y0 + i * 2 * SUB}" width="${W - 8}" height="${2 * SUB}"/>`); });
    (minorTicks || []).forEach(v => p.push(`<line class="pl-vgrid-minor" x1="${XX(v)}" y1="${y0 - 4}" x2="${XX(v)}" y2="${y1}"/>`));
    ticks.forEach(v => p.push(`<line class="pl-vgrid" x1="${XX(v)}" y1="${y0 - 4}" x2="${XX(v)}" y2="${y1}"/><text class="perf-tick" x="${XX(v)}" y="${y1 + 24}" text-anchor="middle">${tickFmt(v)}</text>`));
    p.push(`<line class="pl-axisline" x1="${AX0}" y1="${y1}" x2="${AX1}" y2="${y1}"/>`);
    items.forEach((it, i) => {
      const kind = K[it.m] || ['classic', 'public'], cls = kind[0].startsWith('nn') ? 'pt-nn' : 'pt-cl';
      [[1, it.s1], [8, it.s8]].forEach(([th, s], j) => {
        const yy = y0 + (2 * i + j) * SUB + SUB / 2;
        if (j === 0) p.push(`<text class="perf-lab" x="${NAME}" y="${yy + 6}" text-anchor="end">${esc(it.m)}</text>`);
        if (!s) { p.push(`<text class="perf-ci" x="${TXT}" y="${yy + 5}">${th}t · not measured yet</text>`); return; }
        const ci = s.ci ? `${f3(s.ci[0])}–${f3(s.ci[1])}${s.approx ? '≈' : ''}` : '–';
        const gate = s.clean != null ? ` · ${s.clean}/${s.n} clean` : '';
        p.push(st.view === 'fast'
          ? `<text class="perf-ci" x="${TXT}" y="${yy + 5}">${th}t · fastest ${f3(s.fast)}${s.fastExact ? '' : ' (p10)'} of n=${s.n} · median ${f3(s.c)}</text>`
          : `<text class="perf-ci" x="${TXT}" y="${yy + 5}">${th}t · n=${s.n} · ${ci} · ${f3(s.lo)}–${f3(s.hi)}${gate}</text>`);
        const tip = `${it.m}, ${th} thread${th > 1 ? 's' : ''}: median ${f3(s.c)} ms, mean ${f3(s.mean)} ms, n=${s.n}, 95 % CI ${ci}, ${s.what} ${f3(s.lo)}–${f3(s.hi)} ms` +
          (s.clean != null ? `, ${s.clean} gate-clean / ${s.flagged || 0} flagged rounds` : '') + (s.rss ? `, peak RSS ${s.rss} GiB` : '') +
          (s.simd ? `, SIMD ${s.simd}` : '') + (s.place ? `, placement ${s.place}${s.pc ? ' (' + s.pc + ' CPUs)' : ''}` : '');
        p.push(`<g><title>${esc(tip)}</title>`);
        glyph(p, s, XX, yy, cls, kind, lo);
        p.push('</g>');
      });
    });
    return y1 + FOOT + GAP;
  }

  function render() {
    const pick = th => P.rows.filter(r => r[F.host] === st.host && r[F.size] === st.size && r[F.threads] === th && r[F.phase] === st.phase);
    const by1 = Object.fromEntries(pick(1).map(r => [r[F.metric], stats(r)])), by8 = Object.fromEntries(pick(8).map(r => [r[F.metric], stats(r)]));
    const names = [...new Set([...Object.keys(by1), ...Object.keys(by8)])].filter(m => by1[m] || by8[m]);
    const val = s => st.view === 'fast' ? s.fast : s.c;
    const key = m => val(by1[m] || by8[m]);
    const items = names.map(m => ({ m, s1: by1[m], s8: by8[m] })).sort((a, b) => key(a.m) - key(b.m));
    const info = document.getElementById('perfinfo');
    if (!items.length) { root.innerHTML = '<p class="note">No qualified observations for this selection yet.</p>'; info.textContent = ''; return; }
    const xs = items.map(it => Math.log10(key(it.m))), total = kmeans(xs, 1)[1] || 1;
    let k = 1, lab = xs.map(() => 0), frac = 1;
    for (let kk = 2; kk <= Math.min(7, items.length); kk++) { const [l, ss] = kmeans(xs, kk); k = kk; lab = l; frac = ss / total; if (ss <= 0.06 * total) break; }
    const groups = [...Array(k)].map((_, c) => items.filter((_, i) => lab[i] === c));
    const all = items.flatMap(it => [it.s1, it.s8]).filter(Boolean);
    const lo = 10 ** Math.floor(Math.log10(Math.min(...all.map(s => Math.min(s.lo, s.ci ? s.ci[0] : s.c, ...s.dots))))),
          hi = 10 ** Math.ceil(Math.log10(Math.max(...all.map(s => Math.max(s.hi, s.ci ? s.ci[1] : s.c, ...s.dots)))));
    const LX = v => AX0 + (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * (AX1 - AX0);
    const decades = []; for (let e = Math.log10(lo); e <= Math.log10(hi) + 1e-9; e++) decades.push(10 ** e);
    const minors = decades.flatMap(d => [2 * d, 5 * d]).filter(v => v > lo && v < hi);
    const p = [];
    let y = 8;
    p.push(`<text class="perf-head" x="14" y="${y + 22}">All metrics, log axis (ms per call)</text><text class="perf-ci" x="${TXT}" y="${y + 22}">${st.view === 'fast' ? 'threads · fastest measurement · median (ms)' : `threads · n · 95 % CI · ${st.phase === 'warm' ? 'p10–p90' : 'range'} (ms)${st.phase === 'warm' ? ' · gate-clean rounds' : ''}`}</text>`);
    y += 34;
    groups.forEach((g, gi) => { y = block(p, y, `Cluster ${gi + 1}: ${f3(key(g[0].m))}–${f3(key(g.at(-1).m))} ms, ${g.length} metrics`, g, LX, decades, minors, lo, tick); });
    y += 16;
    p.push(`<text class="perf-head" x="14" y="${y + 22}">Within each cluster, linear axis (ms per call)</text>`);
    y += 34;
    let overlap = 0;
    groups.forEach((g, gi) => {
      const gs = g.flatMap(it => [it.s1, it.s8]).filter(Boolean);
      const gmax = Math.max(...gs.map(s => st.view === 'fast' ? s.fast : Math.max(s.hi, s.ci ? s.ci[1] : 0, s.c, ...s.dots))) * 1.04;
      const step = niceStep(gmax, 8), RX = v => AX0 + Math.max(v, 0) / gmax * (AX1 - AX0);
      const ticks = []; for (let v = 0; v <= gmax + 1e-12; v += step) ticks.push(+v.toPrecision(6));
      y = block(p, y, `Cluster ${gi + 1}, linear`, g, RX, ticks, null, 0, tick);
      g.forEach((it, i) => { const a = it.s1 && it.s1.ci, b = g[i + 1] && g[i + 1].s1 && g[i + 1].s1.ci; if (a && b && a[1] >= b[0]) overlap++; });
    });
    root.innerHTML = `<svg class="chart perf" viewBox="0 0 ${W} ${y}" role="img" aria-label="CPU time per metric, 1 and 8 threads" xmlns="http://www.w3.org/2000/svg">${p.join('')}</svg>`;
    const cov = P.coverage[st.host], anyRounds = all.some(s => s.dots.length && st.phase === 'warm');
    info.innerHTML = `${esc(P.hosts[st.host])}, ${esc(sizeLabel(st.size))}, ${st.phase}: ${items.length} metrics in ${k} clusters by 1-thread median (${(frac * 100).toFixed(1)} % of log-time variance within clusters); ${overlap} adjacent 1-thread pairs have overlapping intervals.` +
      (cov ? ` Campaign coverage on this host so far: ${cov.ok.toLocaleString('en')} of ${cov.cells.toLocaleString('en')} cells.` : '') +
      (st.view === 'fast' ? ' Fastest view: each mark is the metric\'s single fastest measurement (warm: fastest round; cold: fastest fresh process); hollow marks fall back to p10 where rounds are not published yet.' : '') +
      (st.phase === 'warm' ? (anyRounds ? ' Dots are individual zenbench rounds (batch means); intervals from the rounds.' : ' Per-round values are not published for this selection yet; intervals marked ≈ are inferred from p10–p90 assuming a normal distribution.') : ' Cold: each dot is one fresh process (constructor plus first evaluation).');
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
  group('perf-view', [['dist', 'distribution (box and whiskers)'], ['fast', 'fastest measurement']], 'view');
  render();
})();
