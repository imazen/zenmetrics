// IQA metric-cost dashboard. Data: window.DATA (brief_page.py). Every chart is drawn here, sized to its
// panel, and redrawn when a control or the window changes. State lives in the URL hash so any view can
// be linked. ?print=1 lays every tab out as a page for brief.sh's PDF.
(function () {
  'use strict';
  const D = window.DATA, P = D.perf, M = D.meta;
  const F = Object.fromEntries(P.fields.map((f, i) => [f, i]));
  const HOSTS = Object.keys(P.hosts);
  const SIZES = P.sizes.slice().sort((a, b) => a.w * a.h - b.w * b.h);
  const SIZE = Object.fromEntries(SIZES.map(s => [s.name, s]));
  const FAMS = [['classic', 'Classical'], ['neural', 'Neural'], ['dvifmish', 'DVIFM'], ['private', 'Private']];
  const T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086];
  const PRINT = /[?&]print=1/.test(location.search);
  const CELL = new Map();
  for (const r of P.rows) CELL.set([r[F.host], r[F.metric], r[F.size], r[F.threads], r[F.phase]].join('|'), r);
  const NAMES = [...new Set(P.rows.map(r => r[F.metric]))].sort();
  const TABLE = Object.fromEntries(D.tables.map(t => [t.id, t]));
  const SHORT_T = { flops: 'flops', pedzisz_2026_draft: 'Pedzisz draft', i265_P_rthroughput: 'i265 thr.', i265_P_latency: 'i265 lat.',
    llvm_tti_arrowlake_throughput: 'LLVM ARL thr.', llvm_tti_arrowlake_latency: 'LLVM ARL lat.', llvm_mca_arrowlake_rthroughput: 'llvm-mca',
    llvm_tti_znver5_throughput: 'LLVM Zen 5' };

  // ------------------------------------------------------------------ helpers
  const $ = (s, r = document) => r.querySelector(s);
  const rem = () => parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const h = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const fmt = v => v == null || !isFinite(v) ? '–' : v >= 1000 ? Math.round(v).toLocaleString('en') : v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toPrecision(2);
  const tickFmt = v => v >= 1e6 ? +(v / 1e6).toPrecision(3) + 'M' : v >= 1e3 ? +(v / 1e3).toPrecision(3) + 'k' : String(+v.toPrecision(3));
  const hostName = k => (P.hosts[k] || k).replace(/^(Intel|AMD|Apple) /, '');
  const sizeName = n => { const s = SIZE[n]; return s ? `${s.w}×${s.h}` + (n === 'aic-small' ? ' · AIC crop' : '') : n; };
  const thName = t => t === 'both' ? '1 and 8' : t === '8' || t === 8 ? '8 threads' : '1 thread';
  const tq = n => T975[Math.min(n - 2, 19)] || 1.96;
  const meta = m => M[m] || { fam: 'classic', status: 'unchecked', why: [], est: {} };
  const famOn = m => S.fam.split(',').includes(meta(m).fam);
  const qOn = m => !S.q || m.toLowerCase().includes(S.q.toLowerCase());
  const sum = a => a.reduce((x, y) => x + y, 0);

  function cellStats(r) {
    if (!r) return null;
    if (r[F.phase] === 'warm') {
      const med = r[F.median_ms]; if (med == null) return null;
      const rounds = r[F.rounds_ms] || [], n = rounds.length || r[F.n] || 0;
      let mean = r[F.mean_ms] ?? med, ci = null;
      if (rounds.length > 1) {
        mean = sum(rounds) / n;
        const sd = Math.sqrt(sum(rounds.map(v => (v - mean) ** 2)) / (n - 1)), hw = tq(n) * sd / Math.sqrt(n); ci = [mean - hw, mean + hw];
      } else if (r[F.p10_ms] != null && r[F.p90_ms] != null && n > 1) {
        const sd = (r[F.p90_ms] - r[F.p10_ms]) / 2.5631, hw = tq(n) * sd / Math.sqrt(n); ci = [mean - hw, mean + hw];
      }
      return { median: med, mean, ci, lo: r[F.p10_ms] ?? med, hi: r[F.p90_ms] ?? med, fast: rounds.length ? Math.min(...rounds) : (r[F.p10_ms] ?? med),
        n, dots: rounds, flags: new Set(r[F.flagged_rounds] || []), clean: r[F.clean], flagged: r[F.flagged], rss: r[F.rss_gib],
        simd: r[F.simd], place: r[F.placement], src: r[F.source], spread: 'p10–p90 of rounds' };
    }
    const v = (r[F.cold_ms] || []).slice().sort((a, b) => a - b);
    if (!v.length) return null;
    const n = v.length, mean = sum(v) / n, med = n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
    let ci = null;
    if (n > 1) { const sd = Math.sqrt(sum(v.map(x => (x - mean) ** 2)) / (n - 1)), hw = tq(n) * sd / Math.sqrt(n); ci = [Math.max(mean - hw, v[0] / 2), mean + hw]; }
    return { median: med, mean, ci, lo: v[0], hi: v[n - 1], fast: v[0], n, dots: v, flags: new Set(), rss: r[F.rss_gib], src: r[F.source], spread: 'min–max of fresh processes' };
  }
  const cell = (host, m, size, th, ph) => cellStats(CELL.get([host, m, size, +th, ph].join('|')));
  const perPx = size => 1e6 / (SIZE[size].w * SIZE[size].h);
  const unitK = size => S.unit === 'img' ? 1 : perPx(size);
  const unitName = () => S.unit === 'img' ? 'ms per image' : 'ns per pixel';
  const unitShort = () => S.unit === 'img' ? 'ms' : 'ns/px';
  const pick = s => S.show === 'fast' ? s.fast : s.median;

  function niceStep(span, target) {
    const raw = span / Math.max(target, 1), mag = 10 ** Math.floor(Math.log10(raw));
    for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * mag) return m * mag;
    return 10 * mag;
  }
  function logTicks(a, b, room) {
    const out = [], decades = Math.log10(b / a), mult = decades > room / 2 ? [1] : decades > room / 5 ? [1, 3] : [1, 2, 5];
    for (let e = Math.floor(Math.log10(a)) - 1; e <= Math.ceil(Math.log10(b)); e++)
      for (const m of [1, 2, 3, 5]) { const v = m * 10 ** e; if (v >= a * 0.999 && v <= b * 1.001) out.push({ v, label: mult.includes(m), major: m === 1 }); }
    return out;
  }
  function linTicks(a, b, target) {
    const st = niceStep(b - a, target), out = [];
    for (let v = Math.ceil(a / st) * st; v <= b + st * 1e-9; v += st) out.push({ v: +v.toPrecision(12), label: true, major: true });
    return out;
  }
  function kmeans(xs, k) {  // exact 1-D k-means by dynamic programming; xs sorted
    const n = xs.length, pre = [0], pre2 = [0];
    xs.forEach(x => { pre.push(pre.at(-1) + x); pre2.push(pre2.at(-1) + x * x); });
    const cost = (i, j) => (pre2[j] - pre2[i]) - (pre[j] - pre[i]) ** 2 / (j - i);
    const Dm = [...Array(k + 1)].map(() => Array(n + 1).fill(Infinity)), A = [...Array(k + 1)].map(() => Array(n + 1).fill(0));
    Dm[0][0] = 0;
    for (let c = 1; c <= k; c++) for (let j = c; j <= n; j++) for (let i = c - 1; i < j; i++) {
      const v = Dm[c - 1][i] + cost(i, j); if (v < Dm[c][j]) { Dm[c][j] = v; A[c][j] = i; }
    }
    const lab = Array(n).fill(0); let j = n;
    for (let c = k; c > 0; c--) { const i = A[c][j]; for (let t = i; t < j; t++) lab[t] = c - 1; j = i; }
    return [lab, Dm[k][n]];
  }
  function flagTspans(m) {
    const mm = meta(m);
    return (mm.status === 'deviation' ? '<tspan class="g-dev">*</tspan>' : mm.status === 'unchecked' ? '<tspan class="g-lab-dim">°</tspan>' : '')
      + (mm.tuned ? '<tspan class="g-lab-dim">†</tspan>' : '');
  }
  const short = m => m.length > 26 ? m.slice(0, 25) + '…' : m;
  const flagText = m => { const mm = meta(m); return (mm.status === 'deviation' ? '*' : mm.status === 'unchecked' ? '°' : '') + (mm.tuned ? '†' : ''); };
  const MARKS_LEGEND = '<span><b class="mark-dev">*</b> deviation recorded</span><span><b class="mark-un">°</b> no parity check recorded</span><span><b class="mark-un">†</b> DVIFM release retuned on AIC2026 data</span>';

  // ------------------------------------------------------------------ state (URL hash)
  function defaultHost() {
    let best = HOSTS[0], bn = -1;
    for (const host of HOSTS) { const n = NAMES.filter(m => cell(host, m, 'aic-small', 1, 'warm')).length; if (n > bn) { bn = n; best = host; } }
    return best;
  }
  const PIN_DEFAULT = ['psnr-y', 'ssim2', 'butteraugli', 'vmaf', 'lpips-alex', 'dvifmish-port-lap-v0.2'].filter(m => NAMES.includes(m)).map((m, i) => `${m}:${i + 1}`).join(',');
  const DEF = { v: 'overview', host: defaultHost(), size: SIZE['aic-small'] ? 'aic-small' : SIZES[0].name, th: 'both', ph: 'warm', show: 'spread', ax: 'log',
    unit: 'px', fam: FAMS.map(f => f[0]).join(','), q: '', tb: D.best_table, cth: '1', cax: 'log', pins: PIN_DEFAULT, rf: 'all', jf: 'all', sort: '' };
  let S = { ...DEF };
  function readHash() {
    const [v, ...kv] = location.hash.slice(1).split('&');
    S = { ...DEF };
    if (VIEWS.some(x => x.id === v)) S.v = v;
    for (const p of kv) { const i = p.indexOf('='); const k = p.slice(0, i); if (k in DEF) S[k] = decodeURIComponent(p.slice(i + 1)); }
    if (S.host !== 'all' && !HOSTS.includes(S.host)) S.host = DEF.host;
    if (!SIZE[S.size]) S.size = DEF.size;
    if (!TABLE[S.tb]) S.tb = DEF.tb;
  }
  function writeHash() {
    const kv = Object.keys(DEF).filter(k => k !== 'v' && S[k] !== DEF[k]).map(k => `${k}=${encodeURIComponent(S[k])}`);
    history.replaceState(null, '', '#' + [S.v, ...kv].join('&'));
  }
  function set(patch) {
    const viewChanged = patch.v && patch.v !== S.v;
    Object.assign(S, patch);
    if (viewChanged) S.sort = '';
    writeHash();
    if (viewChanged) mount(); else { syncControls(); draw(); }
  }

  // ------------------------------------------------------------------ controls
  function seg(label, key, opts, o = {}) {
    const box = h('div', 'ctl'); box.append(h('div', 'ctl-label', label));
    const row = h('div', 'seg'); box.append(row);
    for (const [val, text, title, key2] of opts) {
      const b = h('button'); b.type = 'button'; b.dataset.k = key; b.dataset.val = val; if (o.multi) b.dataset.multi = '1';
      if (key2) { const k = h('span', 'k'); k.style.background = key2; b.append(k); }
      b.append(document.createTextNode(text)); if (title) b.title = title;
      if (o.disabled) b._dis = () => o.disabled(val);
      if (o.pressed) b._on = () => o.pressed(val);
      b.onclick = () => {
        if (o.multi) {
          const on = new Set(S[key].split(',').filter(Boolean));
          on.has(val) ? on.delete(val) : on.add(val);
          if (!on.size) on.add(val);
          set({ [key]: opts.map(x => x[0]).filter(x => on.has(x)).join(',') });
        } else set({ [key]: val, ...(o.also ? o.also(val) : {}) });
      };
      row.append(b);
    }
    return box;
  }
  function syncControls() {
    document.querySelectorAll('button[data-k]').forEach(b => {
      const cur = S[b.dataset.k];
      b.setAttribute('aria-pressed', String(b._on ? b._on() : b.dataset.multi ? cur.split(',').includes(b.dataset.val) : cur === b.dataset.val));
      b.disabled = b._dis ? b._dis(b.dataset.val) : false;
    });
    document.querySelectorAll('input[data-k]').forEach(i => { if (document.activeElement !== i) i.value = S[i.dataset.k]; });
    document.querySelectorAll('#tabs button').forEach(b => b.dataset.v === S.v ? b.setAttribute('aria-current', 'page') : b.removeAttribute('aria-current'));
  }
  function search(label = 'Find metric') {
    const box = h('div', 'ctl'); box.append(h('div', 'ctl-label', label));
    const i = h('input'); i.type = 'search'; i.placeholder = 'e.g. ssim, lpips, psnr'; i.dataset.k = 'q'; i.value = S.q;
    i.oninput = () => set({ q: i.value.trim() });
    box.append(i); return box;
  }
  const hostSeg = (all = true) => seg('CPU', 'host', [...HOSTS.map(k => [k, hostName(k), P.hosts[k]]), ...(all ? [['all', 'All three', 'one mark per CPU']] : [])],
    { also: v => v === 'all' && S.th === 'both' ? { th: '1' } : {} });
  const sizeSeg = () => seg('Image size', 'size', SIZES.map(s => [s.name, sizeName(s.name), `${s.name}: ${(s.w * s.h / 1e6).toFixed(2)} MP`]));
  const famSeg = (shapes) => seg('Families', 'fam', FAMS.map(([k, t]) => [k, t]), { multi: true });
  const phaseSeg = () => seg('Phase', 'ph', [['warm', 'Warm', 'scorer constructed and warmed; ≥10 rounds of ~100 ms batches'], ['cold', 'Cold', 'three fresh processes: construction + first call']]);
  const showSeg = () => seg('Show', 'show', [['spread', 'Median + spread'], ['fast', 'Fastest only']]);

  // ------------------------------------------------------------------ tooltip and drawer
  const TIP = $('#tip'), DRAWER = $('#drawer');
  function showTip(ev, build) {
    TIP.replaceChildren(); build(TIP); TIP.hidden = false;
    const r = TIP.getBoundingClientRect();
    let x = ev.clientX + 18, y = ev.clientY + 14;
    if (x + r.width > innerWidth - 8) x = Math.max(8, ev.clientX - r.width - 18);
    if (y + r.height > innerHeight - 8) y = Math.max(8, innerHeight - r.height - 8);
    TIP.style.left = x + 'px'; TIP.style.top = y + 'px';
  }
  const hideTip = () => { TIP.hidden = true; };
  function tipRow(tip, color, value, label, sub) {
    const row = h('div', 'row'); const k = h('i'); k.style.background = color || 'transparent';
    row.append(k, h('span', 'v', value), h('span', null, label));
    if (sub) row.append(h('span'), h('span'), h('span', 's', sub));
    tip.append(row);
  }
  function tipHead(tip, m) {
    const mm = meta(m), t = h('h4', null, m + ' ');
    t.append(h('span', 'tag', FAMS.find(f => f[0] === mm.fam)?.[1] || mm.fam));
    tip.append(t);
    if (mm.status === 'deviation') tip.append(h('div', 'mark-dev', '* ' + mm.why.join('; ')));
    else if (mm.status === 'unchecked') tip.append(h('div', 'note', '° no parity check against the reference recorded yet'));
    if (mm.tuned) tip.append(h('div', 'note', '† released DVIFM variant, retuned on AIC2026 data'));
  }
  function openChecks() {
    hideTip();
    const d = DRAWER; d.replaceChildren();
    const close = h('button', 'icon close', '✕'); close.onclick = () => { d.hidden = true; };
    d.append(close, h('h3', null, 'Data checks at build time'),
      h('div', 'hint', 'Run on every rebuild by scripts/complexity/brief_data.py and scripts/aic2026/repro_check.py: hashes of committed data, routes against the column map, the status doc against the full-set reproduction, code changed since a reproduction, timing sync age. Errors also fail `just aic-repro-check`.'));
    const ul = h('ul');
    for (const c of (D.checks || []).slice().sort((a, b) => ({ error: 0, warn: 1, info: 2 }[a.level] - { error: 0, warn: 1, info: 2 }[b.level]))) {
      const li = h('li'); li.append(h('b', c.level === 'error' ? 'mark-dev' : c.level === 'warn' ? null : 'mark-un', `${c.level}: ${c.check}. `), document.createTextNode(c.detail)); ul.append(li);
    }
    d.append(ul); d.hidden = false;
  }
  function openDrawer(m) {
    hideTip();
    const mm = meta(m), d = DRAWER;
    d.replaceChildren();
    const close = h('button', 'icon close', '✕'); close.title = 'Close (Esc)'; close.onclick = () => { d.hidden = true; };
    const title = h('h3', null, m + ' '); title.append(h('span', 'tag', FAMS.find(f => f[0] === mm.fam)?.[1] || mm.fam));
    d.append(close, title);
    const dl = h('dl');
    const row = (k, v) => { dl.append(h('dt', null, k), h('dd', null, v ?? '–')); };
    row('Reference implementation', mm.ref || 'not recorded');
    row('Reference parity (ledger)', mm.parity === 'Not documented' ? 'none recorded' : mm.parity);
    row('AIC2026 column', mm.col);
    row('AIC2026 match', mm.aic);
    row('Primary-oracle check', mm.spot === 'missing' ? 'not run' : mm.spot);
    row('Evidence', mm.status === 'deviation' ? 'deviation: ' + mm.why.join('; ') : mm.status === 'checked' ? 'agrees with at least one reference record' : 'no parity record yet');
    if (mm.tuned) row('Note', 'released DVIFM variant; the author retuned it on AIC2026 data');
    if (mm.alias) row('Same implementation as', mm.alias);
    if (mm.est && mm.est.flops) row('Counted flops per pixel', fmt(mm.est.flops));
    if (mm.c1) row(`${D.fit_host} ${D.fit_shape}, CPU ns/px (perf stat)`, `${fmt(mm.c1)} at 1 thread, ${fmt(mm.c8)} at 8 threads`);
    d.append(dl);
    if (mm.july.length || mm.canon.length || mm.july_cols.length) {
      d.append(h('h4', null, 'Full-set diff (9,618 crops)'));
      if (!mm.july.length && mm.july_cols.length) d.append(h('div', 'hint', `Maps to ${mm.july_cols.join(', ')}, but the full-set run scored that column with another route or has not reached it.`));
      const t = h('table'), tr = h('tr');
      ['set', 'against', 'n', 'median |Δ|', 'max |Δ|', 'median |Δ JND|', 'max |Δ JND|', 'Spearman', 'result'].forEach(x => tr.append(h('th', null, x)));
      const th = h('thead'); th.append(tr); t.append(th); const tb = h('tbody');
      for (const r of mm.july) { const q = h('tr'); [r.set, `published ${r.column}`, r.n, r.median_d, r.max_d, r.median_dJND, r.max_dJND, r.spearman, r.match].forEach(x => q.append(h('td', null, x))); tb.append(q); }
      for (const r of mm.canon) { const q = h('tr'); [r.set, r.reference, r.n, r.median_d, r.max_d, `rel ${r.median_rel}`, `rel ${r.max_rel}`, r.spearman, `${r.match} (gate ${r.gate})`].forEach(x => q.append(h('td', null, x))); tb.append(q); }
      t.append(tb); const w = h('div', 'scroll'); w.append(t); d.append(w);
      const st = (D.repro.sets || [])[0];
      if (mm.code_changed.length) d.append(h('div', 'mark-dev', `Code for this route changed after the reproduction's source commit ${st ? st.commit : ''}: ${mm.code_changed.join(', ')}. Rescore, or confirm the change is bit-exact.`));
    }
    d.append(h('div', 'hint', `Wall time per call, ${S.ph} phase, median ns per pixel [95 % interval of the mean]; empty cells are not measured yet.`));
    const tbl = h('table'), thead = h('thead'), tr = h('tr');
    tr.append(h('th', null, 'size'));
    for (const host of HOSTS) for (const t of [1, 8]) tr.append(h('th', 'num', `${hostName(host)} ${t}t`));
    thead.append(tr); tbl.append(thead);
    const tb = h('tbody');
    for (const s of SIZES) {
      const r = h('tr'); r.append(h('td', null, sizeName(s.name)));
      for (const host of HOSTS) for (const t of [1, 8]) {
        const c = cell(host, m, s.name, t, S.ph), k = perPx(s.name), td = h('td', 'num');
        if (c) { td.append(h('b', null, fmt(c.median * k))); if (c.ci) td.append(h('div', 'ci', `[${fmt(c.ci[0] * k)}, ${fmt(c.ci[1] * k)}]`)); }
        else td.textContent = '–';
        r.append(td);
      }
      tb.append(r);
    }
    tbl.append(tb); const wrap = h('div', 'scroll'); wrap.append(tbl); d.append(wrap);
    d.hidden = false;
  }

  // ------------------------------------------------------------------ pan and zoom
  // Wheel zooms around the pointer, drag pans, double-click resets. The view lives in ZOOM[key], where the key
  // includes every control that changes the axes, so switching tables, threads or axes starts from the default.
  const ZOOM = {};
  let DRAG = null;
  function zoomable(body, key, axes) {
    const tf = (ax, v) => ax.log ? Math.log10(v) : v, itf = (ax, t) => ax.log ? 10 ** t : t;
    const shift = (ax, frac0, scale, dt) => {
      const t0 = tf(ax, ax.dom[0]), t1 = tf(ax, ax.dom[1]), tc = t0 + frac0 * (t1 - t0);
      return [itf(ax, tc - (tc - t0) * scale + dt * (t1 - t0)), itf(ax, tc + (t1 - tc) * scale + dt * (t1 - t0))];
    };
    body.style.touchAction = 'none';
    body.onwheel = ev => {
      ev.preventDefault();
      const b = body.getBoundingClientRect(), f = Math.exp(Math.max(-60, Math.min(60, ev.deltaY)) * 0.004), z = {};
      for (const [k, ax] of Object.entries(axes))
        z[k] = shift(ax, ((k === 'x' ? ev.clientX - b.left : ev.clientY - b.top) - ax.p0) / (ax.p1 - ax.p0), f, 0);
      ZOOM[key] = { ...ZOOM[key], ...z }; hideTip(); draw();
    };
    body.onpointerdown = ev => { if (ev.button === 0) DRAG = { key, x: ev.clientX, y: ev.clientY, axes, id: ev.pointerId, moved: false }; };
    body.onpointermove = ev => {
      if (!DRAG || DRAG.key !== key) return;
      const dx = ev.clientX - DRAG.x, dy = ev.clientY - DRAG.y;
      if (!DRAG.moved && Math.abs(dx) + Math.abs(dy) < 4) return;
      if (!DRAG.moved) { DRAG.moved = true; body.setPointerCapture?.(DRAG.id); }
      const z = {};
      for (const [k, ax] of Object.entries(DRAG.axes)) z[k] = shift(ax, 0, 1, -(k === 'x' ? dx : dy) / (ax.p1 - ax.p0));
      ZOOM[key] = { ...ZOOM[key], ...z }; hideTip(); draw();
    };
    body.onpointerup = () => { if (DRAG && DRAG.moved) body._dragged = true; DRAG = null; };
    body.ondblclick = () => { delete ZOOM[key]; draw(); };
  }
  function zoomButtons() {
    const box = h('div', 'ctl'); box.append(h('div', 'ctl-label', 'View'));
    const row = h('div', 'seg');
    for (const [k, text] of [['all', 'All points'], ['bulk', 'Bulk (90 %)'], ['reset', 'Reset']]) {
      const b = h('button', null, text); b.type = 'button';
      b.onclick = () => { const f = CUR && CUR.body && CUR.body._fit; if (!f) return; if (k === 'reset') delete ZOOM[f.key]; else ZOOM[f.key] = f[k]; draw(); };
      row.append(b);
    }
    box.append(row, h('div', 'hint', 'Scroll to zoom, drag to pan, double-click to reset.'));
    return box;
  }

  // ------------------------------------------------------------------ views
  const VIEWS = [
    { id: 'overview', name: 'Overview', page: mountOverview },
    { id: 'speed', name: 'Speed', rail: railSpeed, draw: drawSpeed, plot: true },
    { id: 'scaling', name: 'Size scaling', rail: railScaling, draw: drawScaling, plot: true, noAll: true },
    { id: 'cost', name: 'Counted cost', rail: railCost, draw: drawCost, plot: true },
    { id: 'repro', name: 'Reproduction', rail: railRepro, draw: drawRepro },
    { id: 'data', name: 'Data', rail: railData, draw: drawData, noAll: true },
    { id: 'method', name: 'Method', page: mountMethod },
  ];
  let CUR = null;  // {view, stage, head, body}

  function mountInto(app, view) {
    app.replaceChildren();
    if (view.noAll && S.host === 'all') S.host = DEF.host;
    if (view.page) { view.page(app); return { view }; }
    const wrap = h('div', 'view'), rail = h('div', 'rail'), stage = h('section', 'stage');
    const head = h('div', 'stage-head'), body = h('div', view.plot ? 'plot' : 'scroll');
    stage.append(head, body); wrap.append(rail, stage); app.append(wrap);
    view.rail(rail);
    return { view, head, body };
  }
  function mount() {
    DRAWER.hidden = true; hideTip();
    CUR = mountInto($('#app'), VIEWS.find(v => v.id === S.v));
    syncControls(); draw();
  }
  function draw() { if (CUR && CUR.view.draw) CUR.view.draw(CUR.head, CUR.body); }
  function setHead(head, title, what, legend) {
    head.innerHTML = `<h2>${esc(title)}</h2><span class="what">${esc(what)}</span>` + (legend ? `<div class="legend">${legend}</div>` : '');
  }
  const empty = (body, text) => { body.innerHTML = `<p class="hint" style="padding:2rem">${esc(text)}</p>`; };

  // Timing coverage: each logical cell once per host. Pending cells on a retired host will never
  // run, so they do not keep the data unfinished; they are reported as not run.
  function coverageSummary() {
    const cov = P.coverage || {}, v = Object.values(cov);
    const ok = sum(v.map(c => c.ok)), cells = sum(v.map(c => c.cells));
    const live = Object.keys(cov).filter(x => !cov[x].retired && (cov[x].pending || 0) > 0);
    const n = x => x.toLocaleString('en');
    const host = x => {
      const c = cov[x]; if (!c) return `${esc(hostName(x))} ?`;
      const parts = [`${n(c.ok)} of ${n(c.cells)} timed`];
      if (c.refused) parts.push(`${n(c.refused)} refused by design`);
      if (c.failed) parts.push(`${n(c.failed)} failed`);
      if (c.pending) parts.push(c.retired ? `${n(c.pending)} not run` : `${n(c.pending)} pending`);
      return `${esc(hostName(x))}${c.retired ? ' (retired ' + esc(String(c.retired.since || '')) + ')' : ''}: ${parts.join(', ')}`;
    };
    return { cov, ok, cells, final: live.length === 0, hosts: HOSTS.map(host) };
  }

  // ---- overview
  function mountOverview(app) {
    const C = coverageSummary(), ok = C.ok, cells = C.cells;
    const timed = NAMES.filter(m => HOSTS.some(host => SIZES.some(s => [1, 8].some(t => cell(host, m, s.name, t, 'warm'))))).length;
    const rounds = sum(P.rows.map(r => (r[F.rounds_ms] || []).length + (r[F.cold_ms] || []).length));
    const host = DEF.host, size = DEF.size, k = perPx(size);
    const at = NAMES.map(m => [m, cell(host, m, size, 1, 'warm')]).filter(x => x[1]).sort((a, b) => a[1].median - b[1].median);
    const med = at.length ? at[Math.floor(at.length / 2)] : null;
    const scal = scalingSummary(host);
    const s1 = D.stats.find(s => s.table === D.best_table && s.threads === 1);
    const cnt = st => Object.values(M).filter(x => !x.alias && x.status === st).length;
    const aicN = D.aic.map(([c, cls, names]) => [c, cls, names.length]);
    const okCols = sum(aicN.filter(a => a[1] === 'ok').map(a => a[2])), devCols = sum(aicN.filter(a => a[1] === 'dev').map(a => a[2]));
    const o = h('div', 'overview');
    o.innerHTML = `
      <h1>What image-quality metrics cost to run</h1>
      <p class="lead">Measured wall time per call for ${timed} metric implementations (zenmetrics, pure Rust) on ${HOSTS.length} CPUs at ${SIZES.length} image sizes,
      and how well counted operations predict CPU time. Every metric reads 8-bit sRGB RGB images already in memory; colour conversion is inside the timed call.</p>
      <div class="banner">${C.final
        ? `<b>Timing campaigns are finished.</b> Every planned cell is timed, refused by its input contract, or failed (${C.hosts.join(' · ')}).`
        : `<b>Unfinished data, updated as it arrives.</b> Timing campaigns are still running (${C.hosts.join(' · ')}).`}
      The page rebuilds every five minutes; timings last synced ${esc((P.synced_utc || '').slice(0, 16).replace('T', ' '))} UTC.
      Agreement with subjective scores is not shown: the preliminary July numbers are superseded and return once the full-dataset reproduction is checked.</div>
      <div class="tiles">
        <div class="tile"><div class="n">${timed}</div><div class="l">metric implementations timed</div></div>
        <div class="tile"><div class="n">${HOSTS.length}</div><div class="l">CPUs: ${HOSTS.map(x => esc(hostName(x))).join(', ')}</div></div>
        <div class="tile"><div class="n">${rounds.toLocaleString('en')}</div><div class="l">timed rounds and fresh-process runs</div></div>
        <div class="tile"><div class="n">${cells ? Math.round(100 * ok / cells) : 0} %</div><div class="l">of planned cells timed (${ok.toLocaleString('en')} of ${cells.toLocaleString('en')})</div></div>
      </div>
      <div class="cards" id="cards"></div>`;
    const cards = [
      ['speed', 'Speed', 'Every metric ranked by time per pixel, per CPU, size and thread count.',
        at.length ? `${sizeName(size)}, ${hostName(host)}, 1 thread: fastest ${at[0][0]} ${fmt(at[0][1].median * k)} ns/px; median metric ${med[0]} ${fmt(med[1].median * k)}; slowest ${at.at(-1)[0]} ${fmt(at.at(-1)[1].median * k)}.` : ''],
      ['scaling', 'Size scaling', 'Time per pixel across the six image sizes.',
        scal ? `${hostName(host)}, 1 thread: ${scal.flat} of ${scal.n} metrics stay within 1.5× per pixel between ${sizeName(scal.a)} and ${sizeName(scal.b)}.` : ''],
      ['cost', 'Counted cost', 'Operation counts against measured CPU time, with fits and intervals.',
        s1 ? `${TABLE[D.best_table].label}: Spearman ρ ${s1.spearman.toFixed(2)} [${s1.spearman_lo.toFixed(2)}, ${s1.spearman_hi.toFixed(2)}], log–log slope ${s1.slope.toFixed(2)} [${s1.slope_lo.toFixed(2)}, ${s1.slope_hi.toFixed(2)}], n ${s1.n}.` : ''],
      ['repro', 'Reproduction', 'Each implementation against its reference and the AIC2026 columns.',
        `${okCols} AIC2026 columns reproduced or matched, ${devCols} close but not exact. Full set: ${(D.repro.misses || []).length} columns miss the gate (${(D.repro.misses || []).map(m => m.column).join(', ') || 'none'}). Metrics: ${cnt('checked')} checked, ${cnt('deviation')} with a recorded deviation, ${cnt('unchecked')} without a parity record yet.`],
      ['data', 'Data', 'Time per pixel by metric and CPU, sortable, as CSV.', 'Download: timings.csv (every cell), complexity.csv (counted costs).'],
      ['method', 'Method', 'What is timed, how zenbench protects it, and the limits.', ''],
    ];
    const cbox = $('#cards', o);
    for (const [id, title, text, key] of cards) {
      const c = h('button', 'vcard'); c.type = 'button'; c.onclick = () => set({ v: id });
      c.append(h('h3', null, title), h('p', null, text)); if (key) c.append(h('p', 'key', key));
      cbox.append(c);
    }
    app.append(o);
  }
  function scalingSummary(host) {
    const a = SIZES[0].name, b = SIZES.at(-1).name;
    const pts = NAMES.map(m => [cell(host, m, a, 1, 'warm'), cell(host, m, b, 1, 'warm')]).filter(x => x[0] && x[1])
      .map(([x, y]) => (y.median * perPx(b)) / (x.median * perPx(a)));
    return pts.length ? { n: pts.length, flat: pts.filter(r => r >= 1 / 1.5 && r <= 1.5).length, a, b } : null;
  }

  // ---- speed
  function railSpeed(rail) {
    rail.append(hostSeg(), sizeSeg(),
      seg('Threads', 'th', [['1', '1 thread', null, null], ['8', '8 threads'], ['both', '1 and 8', 'two marks per metric']], { disabled: v => v === 'both' && S.host === 'all' }),
      phaseSeg(), showSeg(),
      seg('Axis', 'ax', [['log', 'Log'], ['lin', 'Linear, by cluster', 'exact 1-D k-means on log time; each cluster gets its own linear axis']]),
      seg('Unit', 'unit', [['px', 'ns per pixel'], ['img', 'ms per image']]),
      famSeg(), search(),
      h('div', 'hint', 'Hover a row for every number; click it for all sizes and CPUs. Keys: 1–7 tabs, ← → previous/next, F full screen.'));
  }
  function speedSeries() {
    const hosts = S.host === 'all' ? HOSTS : [S.host];
    const ths = S.th === 'both' && S.host !== 'all' ? [1, 8] : [S.th === '8' ? 8 : 1];
    const out = [];
    for (const host of hosts) for (const t of ths)
      out.push({ host, t, color: hosts.length > 1 ? `var(--s${HOSTS.indexOf(host) + 1})` : ths.length > 1 ? `var(--s${t === 1 ? 1 : 2})` : 'var(--s1)',
        label: `${hostName(host)} · ${thName(t)}` });
    return out;
  }
  function drawSpeed(head, body) {
    const series = speedSeries(), k = unitK(S.size);
    const rows = [];
    for (const m of NAMES) {
      if (!famOn(m) || !qOn(m)) continue;
      const cs = series.map(se => { const s = cell(se.host, m, S.size, se.t, S.ph); return s && { ...s, k }; });
      if (cs.some(Boolean)) rows.push({ m, cs, key: pick(cs.find(Boolean)) * k });
    }
    rows.sort((a, b) => a.key - b.key);
    const glyphs = S.show === 'fast' ? 'dot = fastest round or run' : 'dot = median · line = spread · band = 95 % interval of the mean · tiny marks = individual rounds (red: flagged busy)';
    setHead(head, `Time per call, ${unitName()}`, `${sizeName(S.size)} · ${S.host === 'all' ? 'all CPUs' : hostName(S.host)} · ${thName(S.th)} · ${S.ph} · ${rows.length} metrics, fastest first`,
      (series.length > 1 ? series.map(se => `<span><i style="background:${se.color}"></i>${esc(se.label)}</span>`).join('') : '') + `<span>${esc(glyphs)}</span>` + MARKS_LEGEND);
    if (!rows.length) return empty(body, 'No measurements match these settings yet.');
    ranked(body, rows, series);
  }
  function ranked(body, rows, series) {
    const W = body.clientWidth, H = body.clientHeight, fs = rem(), ns = series.length;
    if (W < 50 || H < 50) return;
    const v = (c) => c ? pick(c) * c.k : null;
    let groups;
    if (S.ax === 'log' || rows.length < 4) groups = [{ rows, title: null }];
    else {
      const xs = rows.map(r => Math.log10(r.key)), tot = kmeans(xs, 1)[1];
      let lab = xs.map(() => 0);
      for (let kk = 2; kk <= Math.min(5, rows.length); kk++) { const [l, c] = kmeans(xs, kk); lab = l; if (c <= 0.06 * tot) break; }
      groups = [];
      lab.forEach((g, i) => { (groups[g] ||= { rows: [] }).rows.push(rows[i]); });
      groups = groups.filter(Boolean);
      groups.forEach((g, i) => { g.title = `cluster ${i + 1}: ${fmt(g.rows[0].key)}–${fmt(g.rows.at(-1).key)} ${unitShort()}`; });
    }
    const domain = rs => {  // each column gets its own range, fitted to its rows
      const vals = rs.flatMap(r => r.cs.filter(Boolean).flatMap(c => S.show === 'fast' ? [v(c)] : [c.lo * c.k, c.hi * c.k, v(c), ...(c.ci ? [c.ci[0] * c.k, c.ci[1] * c.k] : [])]))
        .filter(x => x > 0 && isFinite(x));
      if (S.ax === 'lin') return { lo: 0, hi: Math.max(...vals) * 1.04 };
      let lo = Math.min(...vals) / 1.08, hi = Math.max(...vals) * 1.08;
      if (hi / lo < 4) { const m = Math.sqrt(hi * lo); lo = m / 2; hi = m * 2; }
      return { lo, hi };
    };
    const headH = groups.some(g => g.title) ? fs * 1.4 : 0, tickH = fs * 1.35, availH = H - 2 * tickH - headH - 4;
    const maxCols = Math.max(groups.length, Math.floor(W / (fs * 17)));
    let minRow = fs * (1.2 + 0.45 * (ns - 1)), cols;
    for (;;) {
      cols = [];
      for (const g of groups) {
        const n = Math.max(1, Math.ceil(g.rows.length * minRow / availH)), per = Math.ceil(g.rows.length / n);
        for (let i = 0; i < n; i++) { const rs = g.rows.slice(i * per, (i + 1) * per); if (rs.length) cols.push({ g, rows: rs, first: i === 0 }); }
      }
      if (cols.length <= maxCols || minRow < fs * 0.6) break;
      minRow *= 0.9;
    }
    const maxRows = Math.max(...cols.map(c => c.rows.length));
    const rowH = Math.min(fs * (1.9 + 0.5 * (ns - 1)), availH / maxRows), labFs = Math.min(fs * 0.86, rowH * 0.8);
    const colW = W / cols.length;
    const longest = Math.max(...rows.map(r => short(r.m).length + 2));
    const labelW = Math.min(colW * 0.48, longest * labFs * 0.57 + fs * 0.6);
    const sub = ns > 1 ? Math.min(rowH / (ns + 0.4), fs * 0.6) : 0;
    const p = [`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="metrics ranked by time">`];
    const hits = [];
    cols.forEach((c, ci) => {
      const cx = ci * colW, x0 = cx + labelW + fs * 0.3, x1 = cx + colW - fs * 1.1, top = tickH + headH;
      const dm = domain(c.rows), log = S.ax === 'log', L0 = log ? Math.log10(dm.lo) : dm.lo, L1 = log ? Math.log10(dm.hi) : dm.hi;
      const X = val => x0 + ((log ? Math.log10(Math.min(Math.max(val, dm.lo), dm.hi)) : Math.min(val, dm.hi)) - L0) / (L1 - L0) * (x1 - x0);
      const bottom = top + rowH * c.rows.length;
      if (c.g.title) p.push(`<text class="g-head" x="${x0}" y="${fs * 1.05}" font-size="${fs * 0.82}">${esc(c.first ? c.g.title : c.g.title + ' (cont.)')}</text>`);
      const ticks = log ? logTicks(dm.lo, dm.hi, (x1 - x0) / (fs * 2.6)) : linTicks(dm.lo, dm.hi, (x1 - x0) / (fs * 4.2));
      for (const t of ticks) {
        const x = X(t.v);
        p.push(`<line class="${t.major || !log ? 'g-grid' : 'g-grid-minor'}" x1="${x}" y1="${top}" x2="${x}" y2="${bottom}"/>`);
        if (t.label) for (const y of [top - fs * 0.35, bottom + fs * 1.05]) p.push(`<text class="g-tick" x="${x}" y="${y}" text-anchor="middle" font-size="${fs * 0.72}">${tickFmt(t.v)}</text>`);
      }
      c.rows.forEach((r, i) => {
        const y0 = top + i * rowH, ym = y0 + rowH / 2;
        if (i % 2) p.push(`<rect class="g-zebra" x="${cx + fs * 0.2}" y="${y0}" width="${colW - fs * 0.4}" height="${rowH}"/>`);
        p.push(`<text class="g-lab" x="${x0 - fs * 0.45}" y="${ym + labFs * 0.34}" text-anchor="end" font-size="${labFs}">${esc(short(r.m))}${flagTspans(r.m)}</text>`);
        r.cs.forEach((s, si) => {
          if (!s) return;
          const y = ym + (si - (ns - 1) / 2) * sub, col = series[si].color, rr = Math.max(2.4, Math.min(rowH * (ns > 1 ? 0.13 : 0.2), fs * 0.3));
          if (S.show === 'fast') { p.push(`<circle class="g-dot" cx="${X(s.fast * s.k)}" cy="${y}" r="${rr * 1.15}" fill="${col}"/>`); return; }
          const a = X(s.lo * s.k), b = X(s.hi * s.k), cap = Math.min(rowH * 0.22, fs * 0.3);
          if (s.ci) { const q0 = X(s.ci[0] * s.k), q1 = X(s.ci[1] * s.k); p.push(`<rect class="g-ci" x="${Math.min(q0, q1)}" y="${y - cap}" width="${Math.max(Math.abs(q1 - q0), 1.5)}" height="${cap * 2}" fill="${col}"/>`); }
          p.push(`<path class="g-whisk" stroke="${col}" d="M${a},${y}H${b}M${a},${y - cap}V${y + cap}M${b},${y - cap}V${y + cap}"/>`);
          for (let di = 0; di < s.dots.length; di++) p.push(`<rect class="${s.flags.has(di) ? 'g-round g-round-flag' : 'g-round'}" fill="var(--ink)" x="${X(s.dots[di] * s.k) - 0.5}" y="${y - cap - 2.5}" width="1" height="1.5"/>`);
          p.push(`<circle class="g-dot" cx="${X(s.median * s.k)}" cy="${y}" r="${rr}" fill="${col}"/>`);
        });
        hits.push({ x0: cx, x1: cx + colW, y0, y1: y0 + rowH, r });
      });
    });
    p.push('<rect id="hl" class="g-hl" x="0" y="0" width="0" height="0" style="pointer-events:none"/></svg>');
    body.innerHTML = p.join('');
    const svg = body.firstChild, hl = $('#hl', svg);
    hl.parentNode.insertBefore(hl, hl.parentNode.firstChild);  // highlight sits under the marks
    const find = ev => { const b = svg.getBoundingClientRect(), x = ev.clientX - b.left, y = ev.clientY - b.top; return hits.find(t => x >= t.x0 && x < t.x1 && y >= t.y0 && y < t.y1); };
    svg.onpointermove = ev => {
      const t = find(ev);
      if (!t) { hl.setAttribute('width', 0); return hideTip(); }
      Object.entries({ x: t.x0, y: t.y0, width: t.x1 - t.x0, height: t.y1 - t.y0 }).forEach(([a, b]) => hl.setAttribute(a, b));
      showTip(ev, tip => {
        tipHead(tip, t.r.m);
        t.r.cs.forEach((s, si) => {
          if (!s) return tipRow(tip, series[si].color, '–', series[si].label, 'not measured yet');
          const u = unitShort(), k = s.k;
          tipRow(tip, series[si].color, `${fmt(pick(s) * k)} ${u}`, series[si].label + (S.show === 'fast' ? ' (fastest)' : ' (median)'),
            `mean ${fmt(s.mean * k)}${s.ci ? ` [${fmt(s.ci[0] * k)}, ${fmt(s.ci[1] * k)}]` : ''} · ${s.spread} ${fmt(s.lo * k)}–${fmt(s.hi * k)} · n ${s.n}`
            + (s.clean != null ? ` (${s.clean} clean, ${s.flagged || 0} flagged)` : '') + (s.rss != null ? ` · peak RSS ${s.rss} GiB` : '')
            + (s.simd ? ` · ${s.simd}` : '') + (s.place ? ` · ${s.place}` : '') + (s.src ? ` · build ${s.src.slice(0, 8)}` : ''));
        });
        tip.append(h('div', 'note', 'Click for all sizes and CPUs.'));
      });
    };
    svg.onpointerleave = () => { hl.setAttribute('width', 0); hideTip(); };
    svg.onclick = ev => { const t = find(ev); if (t) openDrawer(t.r.m); };
  }

  // ---- size scaling
  const pinList = () => S.pins === 'none' || !S.pins ? [] : S.pins.split(',').map(x => { const i = x.lastIndexOf(':'); return { m: x.slice(0, i), slot: +x.slice(i + 1) }; }).filter(x => NAMES.includes(x.m));
  function togglePin(m) {
    let pins = pinList();
    if (pins.some(p => p.m === m)) pins = pins.filter(p => p.m !== m);
    else { if (pins.length >= 8) pins.shift(); const used = new Set(pins.map(p => p.slot)); pins.push({ m, slot: [1, 2, 3, 4, 5, 6, 7, 8].find(s => !used.has(s)) }); }
    set({ pins: pins.length ? pins.map(p => `${p.m}:${p.slot}`).join(',') : 'none' });
  }
  function railScaling(rail) {
    rail.append(hostSeg(false), seg('Threads', 'th', [['1', '1 thread'], ['8', '8 threads']], { pressed: v => (S.th === '8' ? '8' : '1') === v }), phaseSeg(),
      seg('Show', 'show', [['spread', 'Median'], ['fast', 'Fastest']]), famSeg(), search('Find and pin'));
    const pins = h('div', 'ctl'); pins.id = 'pins'; rail.append(pins);
    rail.append(h('div', 'hint', 'Gray: every other metric. Hover any line for its values; click to pin (up to 8) or unpin. Scroll to zoom, drag to pan, double-click to reset.'));
  }
  function drawScaling(head, body) {
    const host = S.host, t = S.th === '8' ? 8 : 1, pins = pinList(), pinned = new Map(pins.map(p => [p.m, p.slot]));
    const pinBox = $('#pins');
    if (pinBox) {
      pinBox.replaceChildren(h('div', 'ctl-label', 'Pinned'));
      const row = h('div', 'seg');
      for (const p of pins) { const b = h('button'); b.type = 'button'; b.title = 'unpin'; const k = h('span', 'k'); k.style.background = `var(--s${p.slot})`; b.append(k, document.createTextNode(p.m + ' ✕')); b.onclick = () => togglePin(p.m); row.append(b); }
      if (pins.length) { const c = h('button', null, 'clear'); c.type = 'button'; c.onclick = () => set({ pins: 'none' }); row.append(c); }
      pinBox.append(row);
    }
    const lines = [];
    for (const m of NAMES) {
      if (!pinned.has(m) && (!famOn(m) || !qOn(m))) continue;
      const pts = SIZES.map((s, i) => { const c = cell(host, m, s.name, t, S.ph); return c && { i, v: pick(c) * perPx(s.name), c }; }).filter(Boolean);
      if (pts.length) lines.push({ m, pts, slot: pinned.get(m) });
    }
    setHead(head, 'Time per pixel against image size', `${hostName(host)} · ${thName(t)} · ${S.ph} · ${S.show === 'fast' ? 'fastest' : 'median'} ns per pixel, log scale · ${lines.length} metrics`,
      pins.map(p => `<span><i style="background:var(--s${p.slot})"></i>${esc(p.m)}</span>`).join('') + '<span>flat = time grows in proportion to pixel count</span>');
    if (!lines.length) return empty(body, 'No measurements match these settings yet.');
    const W = body.clientWidth, H = body.clientHeight, fs = rem();
    const L = fs * 3.6, R = fs * 12, T = fs * 0.8, B = fs * 2.8;
    const all = lines.flatMap(l => l.pts.map(p => p.v));
    const zkey = `scaling|${host}|${t}|${S.ph}|${S.show}|${S.fam}|${S.q}`;
    const [lo, hi] = (ZOOM[zkey] || {}).y || [Math.min(...all) / 1.2, Math.max(...all) * 1.2], l0 = Math.log10(lo), l1 = Math.log10(hi);
    const X = i => L + (i + 0.5) / SIZES.length * (W - L - R), Y = v => T + (1 - (Math.log10(v) - l0) / (l1 - l0)) * (H - T - B);
    const p = [`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="time per pixel against image size"><defs><clipPath id="sc"><rect x="${L - 6}" y="${T}" width="${W - L - R + 12}" height="${H - T - B}"/></clipPath></defs>`];
    for (const tk of logTicks(lo, hi, (H - T - B) / (fs * 1.6))) {
      const y = Y(tk.v);
      p.push(`<line class="${tk.major ? 'g-grid' : 'g-grid-minor'}" x1="${L}" y1="${y}" x2="${W - R}" y2="${y}"/>`);
      if (tk.label) p.push(`<text class="g-tick" x="${L - fs * 0.4}" y="${y + fs * 0.25}" text-anchor="end" font-size="${fs * 0.72}">${tickFmt(tk.v)}</text>`);
    }
    SIZES.forEach((s, i) => {
      p.push(`<line class="g-grid" x1="${X(i)}" y1="${T}" x2="${X(i)}" y2="${H - B}"/>`);
      p.push(`<text class="g-tick" x="${X(i)}" y="${H - B + fs * 1.05}" text-anchor="middle" font-size="${fs * 0.72}">${esc(`${s.w}×${s.h}`)}</text>`);
      p.push(`<text class="g-tick" x="${X(i)}" y="${H - B + fs * 2.05}" text-anchor="middle" font-size="${fs * 0.65}">${(s.w * s.h / 1e6).toFixed(2)} MP</text>`);
    });
    p.push(`<text class="g-tick" transform="translate(${fs * 0.9},${(T + H - B) / 2}) rotate(-90)" text-anchor="middle" font-size="${fs * 0.75}">ns per pixel</text>`);
    const path = l => l.pts.map((q, j) => `${j ? 'L' : 'M'}${X(q.i).toFixed(1)},${Y(q.v).toFixed(1)}`).join('');
    p.push('<g clip-path="url(#sc)">');
    for (const l of lines) if (!l.slot) p.push(`<path class="g-bg-line" d="${path(l)}"/>`);
    p.push('<path id="hot" class="g-bg-line hot" d=""/>');
    const labels = [];
    for (const l of lines.filter(l => l.slot)) {
      p.push(`<path class="g-line" stroke="var(--s${l.slot})" d="${path(l)}"/>`);
      for (const q of l.pts) p.push(`<circle class="g-dot" cx="${X(q.i)}" cy="${Y(q.v)}" r="${fs * 0.22}" fill="var(--s${l.slot})"/>`);
      const last = l.pts.at(-1); if (last.v >= lo && last.v <= hi) labels.push({ y: Y(last.v), x: X(last.i), l });
    }
    p.push('</g>');
    labels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + fs * 0.95);
    for (const lb of labels) p.push(`<text class="g-lab" x="${lb.x + fs * 0.5}" y="${lb.y + fs * 0.3}" font-size="${fs * 0.8}">${esc(lb.l.m)}${flagTspans(lb.l.m)}</text>`);
    p.push('</svg>');
    body.innerHTML = p.join('');
    const svg = body.firstChild, hot = $('#hot', svg);
    const nearest = ev => {
      const b = svg.getBoundingClientRect(), x = ev.clientX - b.left, y = ev.clientY - b.top;
      let best = null, bd = fs * 1.2;
      for (const l of lines) for (let j = 0; j < l.pts.length; j++) {
        const ax = X(l.pts[j].i), ay = Y(l.pts[j].v);
        const [bx, by] = j + 1 < l.pts.length ? [X(l.pts[j + 1].i), Y(l.pts[j + 1].v)] : [ax, ay];
        const dx = bx - ax, dy = by - ay, len = dx * dx + dy * dy, u = len ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len)) : 0;
        const d = Math.hypot(x - ax - u * dx, y - ay - u * dy);
        if (d < bd) { bd = d; best = l; }
      }
      return best;
    };
    svg.onpointermove = ev => {
      const l = nearest(ev);
      if (!l) { hot.setAttribute('d', ''); return hideTip(); }
      hot.setAttribute('d', path(l));
      showTip(ev, tip => {
        tipHead(tip, l.m);
        for (const q of l.pts) tipRow(tip, l.slot ? `var(--s${l.slot})` : 'var(--muted)', `${fmt(q.v)} ns/px`, sizeName(SIZES[q.i].name),
          `n ${q.c.n}${q.c.ci ? ` · 95 % [${fmt(q.c.ci[0] * perPx(SIZES[q.i].name))}, ${fmt(q.c.ci[1] * perPx(SIZES[q.i].name))}]` : ''}`);
        tip.append(h('div', 'note', l.slot ? 'Click to unpin.' : 'Click to pin.'));
      });
    };
    svg.onpointerleave = () => { hot.setAttribute('d', ''); hideTip(); };
    svg.onclick = ev => { if (body._dragged) { body._dragged = false; return; } const l = nearest(ev); if (l) togglePin(l.m); };
    zoomable(body, zkey, { y: { log: true, dom: [lo, hi], p0: H - B, p1: T } });
  }

  // ---- counted cost
  const statOf = (tb, th) => D.stats.find(s => s.table === tb && s.threads === th);
  function railCost(rail) {
    rail.append(seg('Weight table', 'tb', D.tables.filter(t => statOf(t.id, 1)).map(t => [t.id, SHORT_T[t.id] || t.label, t.label])),
      seg('Threads', 'cth', [['1', '1 thread'], ['8', '8 threads', 'CPU time summed over the 8 threads, not wall time']]),
      seg('Axes', 'cax', [['log', 'Log–log'], ['lin', 'Linear']]), famSeg(), zoomButtons());
    const card = h('div', 'card'); card.id = 'coststats'; rail.append(card);
    const rank = h('div', 'card'); rank.id = 'costrank'; rail.append(rank);
    rail.append(h('div', 'hint', 'Counts ignore SIMD width, memory traffic, caches and threading; time is one implementation per metric on one CPU, so a point\'s distance from the line is that implementation\'s efficiency as much as the metric\'s cost.'));
  }
  function drawCost(head, body) {
    const th = +S.cth, s = statOf(S.tb, th), key = th === 1 ? 'c1' : 'c8', lin = S.cax === 'lin';
    const pts = Object.entries(M).filter(([m, mm]) => !mm.alias && mm.est[S.tb] > 0 && mm[key] && famOn(m)).map(([m, mm]) => ({ m, mm, x: mm.est[S.tb], y: mm[key] }));
    const card = $('#coststats'), rank = $('#costrank');
    const p3 = v => v == null ? '–' : v < 1e-3 ? v.toExponential(1) : v.toFixed(3);
    if (card && s) card.innerHTML = `<b>${esc(TABLE[S.tb].label)}, ${th} thread${th > 1 ? 's' : ''}</b>
      <div>n <b>${s.n}</b> metrics (fit uses all families)</div>
      <div>Spearman ρ <b>${s.spearman.toFixed(3)}</b> <span class="ci">[${s.spearman_lo.toFixed(3)}, ${s.spearman_hi.toFixed(3)}]</span>, p ${p3(s.spearman_p)}</div>
      <div>Pearson r, log–log <b>${s.pearson_log.toFixed(3)}</b> <span class="ci">[${s.pearson_log_lo.toFixed(3)}, ${s.pearson_log_hi.toFixed(3)}]</span></div>
      <div>log–log slope <b>${s.slope.toFixed(2)}</b> <span class="ci">[${s.slope_lo.toFixed(2)}, ${s.slope_hi.toFixed(2)}]</span>${s.slope_hi < 1 ? ' (below 1: costly metrics run cheaper per counted operation)' : ''}</div>
      <div>scale <b>${fmt(s.scale)}</b> ns per unit <span class="ci">[${fmt(s.scale_lo)}, ${fmt(s.scale_hi)}]</span></div>
      <div>through-origin linear <b>${fmt(s.scale_linear)}</b> <span class="ci">[${fmt(s.scale_linear_lo)}, ${fmt(s.scale_linear_hi)}]</span></div>
      <div>typical error <b>${s.typical.toFixed(2)}×</b>; within 2×: <b>${s.within2}/${s.n}</b></div>
      <div class="hint">Intervals: 2,000-sample bootstrap over metrics (seed ${D.seed}); Pearson by Fisher z; slope by OLS t-interval.</div>`;
    if (rank) {
      rank.innerHTML = `<b>Rank correlation by weight table, ${th} thread${th > 1 ? 's' : ''}</b>`;
      const tb = h('table'), body2 = h('tbody');
      for (const t of D.tables.map(t => [t, statOf(t.id, th)]).filter(x => x[1]).sort((a, b) => b[1].spearman - a[1].spearman)) {
        const r = h('tr'); if (t[0].id === S.tb) r.style.fontWeight = '650';
        r.append(h('td', null, SHORT_T[t[0].id] || t[0].label));
        const td = h('td', 'num', t[1].spearman.toFixed(3) + ' '); td.append(h('span', 'ci', `[${t[1].spearman_lo.toFixed(2)}, ${t[1].spearman_hi.toFixed(2)}]`)); r.append(td);
        r.onclick = () => set({ tb: t[0].id }); body2.append(r);
      }
      tb.append(body2); rank.append(tb);
    }
    setHead(head, 'Counted cost against measured CPU time',
      `${TABLE[S.tb].label} · ${D.fit_host} · ${D.fit_shape} · CPU time per pixel from perf stat, ${th} thread${th > 1 ? 's' : ''} · ${pts.length} metrics`,
      `<span><i style="background:var(--s1)"></i>classical</span><span><i style="background:var(--s2)"></i>neural network</span>
       <span><svg viewBox="0 0 10 10"><rect x="1" y="1" width="8" height="8" fill="var(--ink2)"/></svg>private port</span>
       <span><svg viewBox="0 0 10 10"><path d="M5,0L10,5L5,10L0,5Z" fill="var(--ink2)"/></svg>DVIFM</span>`
      + (lin ? '<span>dashed: through-origin fit; band: its 95 % interval</span>' : '<span>dashed: single scale (unit slope), band ±2×; solid: least-squares log–log line</span>'));
    if (!pts.length || !s) return empty(body, 'No counted costs for this selection.');
    const W = body.clientWidth, H = body.clientHeight, fs = rem(), L = fs * 4.2, R = fs * 1.2, T = fs * 0.8, B = fs * 2.6;
    // view: zoom/pan state per (table, threads, axes, families); linear axes start on the bulk (90th percentile)
    const xs = pts.map(q => q.x), ys = pts.map(q => q.y), qt = (a, f) => a.slice().sort((u, v) => u - v)[Math.floor(f * (a.length - 1))];
    const all = lin ? { x: [0, Math.max(...xs) * 1.05], y: [0, Math.max(...ys) * 1.05] }
      : { x: [Math.min(...xs) / 1.5, Math.max(...xs) * 1.5], y: [Math.min(...ys) / 1.5, Math.max(...ys) * 1.5] };
    const bulk = lin ? { x: [0, qt(xs, 0.9) * 1.3], y: [0, qt(ys, 0.9) * 1.3] } : all;
    const zkey = `cost|${S.tb}|${th}|${S.cax}|${S.fam}`, view = ZOOM[zkey] || (lin ? bulk : all);
    const xd = view.x, yd = view.y, tf = v => lin ? v : Math.log10(v);
    const X = v => L + (tf(v) - tf(xd[0])) / (tf(xd[1]) - tf(xd[0])) * (W - L - R);
    const Y = v => T + (1 - (tf(v) - tf(yd[0])) / (tf(yd[1]) - tf(yd[0]))) * (H - T - B);
    body._fit = { key: zkey, all, bulk };
    const p = [`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="counted cost against CPU time"><defs><clipPath id="pc"><rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}"/></clipPath></defs>`];
    const xt = lin ? linTicks(xd[0], xd[1], (W - L - R) / (fs * 5)) : logTicks(xd[0], xd[1], (W - L - R) / (fs * 3));
    const yt = lin ? linTicks(yd[0], yd[1], (H - T - B) / (fs * 2.5)) : logTicks(yd[0], yd[1], (H - T - B) / (fs * 1.6));
    for (const t of xt) { const x = X(t.v); p.push(`<line class="${t.major || lin ? 'g-grid' : 'g-grid-minor'}" x1="${x}" y1="${T}" x2="${x}" y2="${H - B}"/>`); if (t.label) p.push(`<text class="g-tick" x="${x}" y="${H - B + fs * 1.05}" text-anchor="middle" font-size="${fs * 0.72}">${tickFmt(t.v)}</text>`); }
    for (const t of yt) { const y = Y(t.v); p.push(`<line class="${t.major || lin ? 'g-grid' : 'g-grid-minor'}" x1="${L}" y1="${y}" x2="${W - R}" y2="${y}"/>`); if (t.label) p.push(`<text class="g-tick" x="${L - fs * 0.4}" y="${y + fs * 0.25}" text-anchor="end" font-size="${fs * 0.72}">${tickFmt(t.v)}</text>`); }
    p.push(`<text class="g-tick" x="${(L + W - R) / 2}" y="${H - fs * 0.3}" text-anchor="middle" font-size="${fs * 0.78}">counted cost per pixel (${esc(SHORT_T[S.tb] || S.tb)})</text>`);
    p.push(`<text class="g-tick" transform="translate(${fs * 0.9},${(T + H - B) / 2}) rotate(-90)" text-anchor="middle" font-size="${fs * 0.78}">CPU ns per pixel</text>`);
    p.push('<g clip-path="url(#pc)">');
    if (lin) {
      const xm = xd[1];
      p.push(`<path class="g-band" d="M${X(0)},${Y(0)}L${X(xm)},${Y(s.scale_linear_hi * xm)}L${X(xm)},${Y(s.scale_linear_lo * xm)}Z"/>`);
      p.push(`<path class="g-fit" d="M${X(0)},${Y(0)}L${X(xm)},${Y(s.scale_linear * xm)}"/>`);
    } else {
      const [a, b] = xd;
      p.push(`<path class="g-band" d="M${X(a)},${Y(2 * s.scale * a)}L${X(b)},${Y(2 * s.scale * b)}L${X(b)},${Y(s.scale * b / 2)}L${X(a)},${Y(s.scale * a / 2)}Z"/>`);
      p.push(`<path class="g-fit" d="M${X(a)},${Y(s.scale * a)}L${X(b)},${Y(s.scale * b)}"/>`);
      const ols = v => 10 ** (s.intercept + s.slope * Math.log10(v));
      p.push(`<path class="g-ols" d="M${X(a)},${Y(ols(a))}L${X(b)},${Y(ols(b))}"/>`);
    }
    p.push('</g>');
    const r0 = fs * 0.3, inView = q => q.x >= xd[0] && q.x <= xd[1] && q.y >= yd[0] && q.y <= yd[1];
    let nOut = 0;
    for (const q of pts) {
      let cx = X(q.x), cy = Y(q.y);
      const col = q.mm.fam === 'neural' || (q.mm.kind || '').startsWith('nn') ? 'var(--s2)' : 'var(--s1)';
      if (!inView(q)) {  // off-view point: a small arrowhead on the plot edge, pointing towards it
        nOut++;
        const ex = Math.min(Math.max(cx, L + 4), W - R - 4), ey = Math.min(Math.max(cy, T + 4), H - B - 4), ang = Math.atan2(cy - ey, cx - ex);
        const k = r0 * 1.6, pt = (a, rr) => `${ex + Math.cos(ang + a) * rr},${ey + Math.sin(ang + a) * rr}`;
        p.push(`<path class="g-out" d="M${pt(0, k)}L${pt(2.5, k * 0.8)}L${pt(-2.5, k * 0.8)}Z" fill="${col}"/>`);
        q.cx = ex; q.cy = ey; q.out = true;
        continue;
      }
      q.cx = cx; q.cy = cy; q.out = false;
      if (q.mm.origin === 'private') p.push(`<rect class="g-pt" x="${cx - r0}" y="${cy - r0}" width="${2 * r0}" height="${2 * r0}" fill="${col}"/>`);
      else if (q.mm.origin === 'dvifmish') p.push(`<path class="g-pt" d="M${cx},${cy - r0 * 1.3}L${cx + r0 * 1.3},${cy}L${cx},${cy + r0 * 1.3}L${cx - r0 * 1.3},${cy}Z" fill="${col}"/>`);
      else p.push(`<circle class="g-pt" cx="${cx}" cy="${cy}" r="${r0}" fill="${col}"/>`);
    }
    // labels: largest residuals on log axes, largest values on linear axes; skip collisions
    const boxes = [];
    const order = lin ? pts.slice().sort((a, b) => b.y - a.y).slice(0, 8) : pts.slice().sort((a, b) => Math.abs(Math.log10(b.y / (s.scale * b.x))) - Math.abs(Math.log10(a.y / (s.scale * a.x)))).filter(q => Math.abs(Math.log10(q.y / (s.scale * q.x))) > Math.log10(3));
    for (const q of order) {
      if (q.out) continue;
      const w = q.m.length * fs * 0.42, x = q.cx + r0 + 3, y = q.cy + fs * 0.25;
      const bx = x + w > W - R ? [q.cx - r0 - 3 - w, q.cx - r0 - 3] : [x, x + w];
      const box = [bx[0] - 2, y - fs * 0.78, bx[1] + 2, y + fs * 0.22];
      if (boxes.some(b => !(box[2] < b[0] || box[0] > b[2] || box[3] < b[1] || box[1] > b[3]))) continue;
      boxes.push(box);
      p.push(`<text class="g-lab" x="${bx[0]}" y="${y}" font-size="${fs * 0.72}">${esc(q.m)}</text>`);
    }
    p.push('<circle id="hotpt" class="g-pt hot" r="0" fill="none"/></svg>');
    body.innerHTML = p.join('');
    const svg = body.firstChild, hot = $('#hotpt', svg);
    const near = ev => { const b = svg.getBoundingClientRect(), x = ev.clientX - b.left, y = ev.clientY - b.top; let best = null, bd = fs * 1.3; for (const q of pts) { const d = Math.hypot(q.cx - x, q.cy - y); if (d < bd) { bd = d; best = q; } } return best; };
    svg.onpointermove = ev => {
      const q = near(ev);
      if (!q) { hot.setAttribute('r', 0); return hideTip(); }
      hot.setAttribute('cx', q.cx); hot.setAttribute('cy', q.cy); hot.setAttribute('r', r0 * 1.8);
      showTip(ev, tip => {
        tipHead(tip, q.m);
        tipRow(tip, null, `${fmt(q.y)} ns/px`, `CPU time, ${th} thread${th > 1 ? 's' : ''}`);
        tipRow(tip, null, fmt(q.x), `counted cost (${SHORT_T[S.tb] || S.tb})`);
        tipRow(tip, null, `${(q.y / (s.scale * q.x)).toFixed(2)}×`, 'measured ÷ fitted scale');
        if (q.out) tip.append(h('div', 'note', 'Outside the current view.'));
        tip.append(h('div', 'note', 'Click for details.'));
      });
    };
    svg.onpointerleave = () => { hot.setAttribute('r', 0); hideTip(); };
    svg.onclick = ev => { if (body._dragged) { body._dragged = false; return; } const q = near(ev); if (q) openDrawer(q.m); };
    if (nOut) head.querySelector('.legend').insertAdjacentHTML('beforeend', `<span class="mark-dev">${nOut} point${nOut > 1 ? 's' : ''} outside this view (edge arrows): scroll out or press “All points”</span>`);
    zoomable(body, zkey, { x: { log: !lin, dom: xd, p0: L, p1: W - R }, y: { log: !lin, dom: yd, p0: H - B, p1: T } });
  }

  // ---- reproduction
  function railRepro(rail) {
    const cnt = st => Object.values(M).filter(x => !x.alias && (st === 'all' || x.status === st)).length;
    const jn = j => Object.values(M).filter(x => !x.alias && (j === 'all' || (x.july_cols.length > 0) === (j === 'in'))).length;
    rail.append(seg('Published AIC2026 column', 'jf', [['all', `All (${jn('all')})`], ['in', `Has a column (${jn('in')})`], ['out', `No column (${jn('out')})`]]),
      seg('Evidence', 'rf', [['all', `All (${cnt('all')})`], ['deviation', `* Deviation (${cnt('deviation')})`], ['checked', `✓ Checked (${cnt('checked')})`], ['unchecked', `° Unchecked (${cnt('unchecked')})`]]),
      famSeg(), search());
    const links = h('div', 'ctl'); links.append(h('div', 'ctl-label', 'Full-set diff against the published scores'));
    const lseg = h('div', 'card');
    for (const st of D.repro.sets || []) {
      const ok = st.rows.filter(r => r.match === 'MATCH').length, miss = st.rows.filter(r => r.match === 'MISS').length;
      lseg.append(h('div', null, `${st.set}: ${ok} columns match, ${miss} miss the 1e-3 site-JND gate; source ${st.commit || '?'}`));
      const row = h('div', 'seg');
      for (const [href, text] of [[st.csv_url, 'Our scores (CSV.gz)'], [st.url, 'Per-column diff (TSV)']]) if (href) { const a = h('a', 'icon', text); a.href = href; a.target = '_blank'; a.rel = 'noopener'; row.append(a); }
      lseg.append(row);
    }
    for (const st of D.repro.canonical || []) { const a = h('a', 'icon', `Reference implementations, ${st.set} (TSV)`); a.href = st.url; a.target = '_blank'; a.rel = 'noopener'; lseg.append(a); }
    const ref = D.repro.reference || {};
    const pub = h('a', 'icon', 'Reference: July/drop combined CSV'); pub.href = ref.url || 'https://doi.org/10.18419/DARUS-6156'; pub.target = '_blank'; pub.rel = 'noopener';
    const dar = h('a', 'icon', 'July release (DaRUS)'); dar.href = 'https://doi.org/10.18419/DARUS-6156'; dar.target = '_blank'; dar.rel = 'noopener';
    lseg.append(pub, dar, h('div', 'hint', `Reference sha256 ${(ref.sha256 || '').slice(0, 12)}…, the July release plus the online-dashboard export. Diff locally: python3 scripts/aic2026/july_diff.py OURS.csv.gz combined.csv report.tsv worst.txt --view cropped`));
    links.append(lseg); rail.append(links);
    const box = h('div', 'ctl'); box.append(h('div', 'ctl-label', 'Published AIC2026 columns'));
    for (const [cat, cls, names] of D.aic) {
      const c = h('details', 'card'), s = h('summary');
      s.append(h('b', cls === 'dev' ? 'mark-dev' : cls === 'ok' ? 'mark-ok' : null, String(names.length)), document.createTextNode(' ' + cat));
      c.append(s, h('div', 'hint', names.join(', ') || '–')); box.append(c);
    }
    rail.append(box);
  }
  function sortable(body, cols, rows, onRow) {
    const [sk, sd] = (S.sort || '').split(':');
    const si = cols.findIndex(c => c.id === sk);
    if (si >= 0) rows.sort((a, b) => { const x = cols[si].sortv(a), y = cols[si].sortv(b); return (x == null) - (y == null) || (x < y ? -1 : x > y ? 1 : 0) * (sd === 'd' ? -1 : 1); });
    const tbl = h('table'), tr = h('tr');
    cols.forEach((c, i) => {
      const th = h('th', c.num ? 'num' : null, c.label); if (c.title) th.title = c.title;
      if (i === si) th.setAttribute('aria-sort', sd === 'd' ? 'descending' : 'ascending');
      th.onclick = () => set({ sort: `${c.id}:${i === si && sd !== 'd' ? 'd' : 'a'}` });
      tr.append(th);
    });
    const thead = h('thead'); thead.append(tr); tbl.append(thead);
    const tb = h('tbody');
    for (const r of rows) {
      const row = h('tr');
      for (const c of cols) { const td = h('td', c.num ? 'num' : null); c.cell(r, td); row.append(td); }
      if (onRow) row.onclick = () => onRow(r);
      tb.append(row);
    }
    tbl.append(tb); body.replaceChildren(tbl);
  }
  const evidenceCell = (m, td) => { const mm = meta(m); td.append(h('span', mm.status === 'deviation' ? 'mark-dev' : mm.status === 'checked' ? 'mark-ok' : 'mark-un', mm.status === 'deviation' ? '* deviation' : mm.status === 'checked' ? '✓ checked' : '° none yet')); if (mm.why.length) td.title = mm.why.join('; '); };
  function drawRepro(head, body) {
    const host = S.host === 'all' ? DEF.host : S.host;
    const rows = Object.keys(M).filter(m => !M[m].alias && famOn(m) && qOn(m) && (S.rf === 'all' || M[m].status === S.rf)
      && (S.jf === 'all' || (M[m].july_cols.length > 0) === (S.jf === 'in')));
    setHead(head, 'Ours against the reference implementations',
      `${rows.length} metrics. Full-set diff: our score on all 9,618 crops against the published column (July/drop union; median |Δ site-JND|, gate 1e-3) or, where the CSV has no column, against the reference implementation. Status doc: docs/AIC2026_REPRODUCTION_STATUS.md. Reference parity: the provenance ledger or model card. Oracle check: primary-oracle spot check.`,
      MARKS_LEGEND);
    const diffv = m => { const j = M[m].july.find(r => r.set === 'cropped') || M[m].july[0]; return j ? +j.median_dJND : null; };
    const diffCell = (m, td) => {
      const j = M[m].july.find(r => r.set === 'cropped') || M[m].july[0], c = M[m].canon[0];
      if (j) { td.append(h('span', j.match === 'MATCH' ? 'mark-ok' : 'mark-dev', (j.match === 'MATCH' ? '✓ ' : '✗ ') + j.median_dJND)); td.append(h('div', 'ci', `${j.column} · ρ ${j.spearman}`)); }
      else if (c) { td.append(h('span', c.match === 'MATCH' ? 'mark-ok' : c.match === 'MISS' ? 'mark-dev' : null, `${c.match === 'MATCH' ? '✓' : c.match === 'MISS' ? '✗' : ''} rel ${c.median_rel}`)); td.append(h('div', 'ci', `vs ${c.reference} · ρ ${c.spearman}`)); }
      else if (M[m].july_cols.length) { td.textContent = 'not scored'; td.title = `${M[m].july_cols.join(', ')}: the full-set run used a different route or has not reached this column`; }
      else td.textContent = '–';
    };
    const speed = m => { const c = cell(host, m, DEF.size, 1, 'warm'); return c ? c.median * perPx(DEF.size) : null; };
    const txt = (f) => (r, td) => { td.textContent = f(r) ?? '–'; };
    const tableHost = h('div');
    body.replaceChildren(missesPanel(), tableHost);
    sortable(tableHost, [
      { id: 'm', label: 'metric', cell: (m, td) => { td.append(h('code', null, m)); const f = flagText(m); if (f) td.append(h('span', meta(m).status === 'deviation' ? 'mark-dev' : 'mark-un', ' ' + f)); }, sortv: m => m },
      { id: 'fam', label: 'family', cell: txt(m => FAMS.find(f => f[0] === M[m].fam)?.[1]), sortv: m => M[m].fam },
      { id: 'ev', label: 'evidence', cell: evidenceCell, sortv: m => M[m].status },
      { id: 'ref', label: 'reference implementation', cell: txt(m => M[m].ref || 'not recorded'), sortv: m => M[m].ref },
      { id: 'par', label: 'reference parity', cell: txt(m => M[m].parity === 'Not documented' ? 'none recorded' : M[m].parity), sortv: m => M[m].parity },
      { id: 'col', label: 'published column', title: 'columns of the July/drop published tables; columns only named on the AIC site in grey', cell: (m, td) => { td.textContent = M[m].july_cols.join(', ') || (M[m].site_cols.length ? '' : '–'); if (M[m].site_cols.length) td.append(h('div', 'ci', 'site only: ' + M[m].site_cols.join(', '))); }, sortv: m => M[m].july_cols[0] },
      { id: 'diff', label: 'full-set diff', title: 'median |Δ site-JND| against the published column (✓ ≤ 1e-3), or median relative difference from the reference implementation', cell: (m, td) => diffCell(m, td), sortv: diffv },
      { id: 'aic', label: 'status doc', cell: txt(m => M[m].aic), sortv: m => M[m].aic },
      { id: 'spot', label: 'oracle check', cell: txt(m => M[m].spot === 'missing' ? 'not run' : M[m].spot), sortv: m => M[m].spot },
      { id: 'ns', label: 'ns/px', num: true, title: `${hostName(host)}, ${sizeName(DEF.size)}, 1 thread, warm median`, cell: (m, td) => { td.textContent = fmt(speed(m)); }, sortv: speed },
    ], rows, openDrawer);
  }

  // Published columns our full-set run misses, with the cause the status doc records for each.
  function missesPanel() {
    const misses = (D.repro.misses || []).slice().sort((a, b) => +b.median_dJND - +a.median_dJND);
    const box = h('section', 'misses');
    if (!misses.length) return box;
    const sets = [...new Set(misses.map(m => m.set))].join(', ');
    box.append(h('h3', null, `${misses.length} published columns miss the 1e-3 site-JND gate (${sets}, all 9,618 crops)`));
    box.append(h('p', 'hint', 'Cause and open detail from docs/AIC2026_REPRODUCTION_STATUS.md, the section named on each card. Numbers: median and max |Δ| in site-JND units, median raw |Δ|, Spearman against the published column.'));
    const grid = h('div', 'miss-grid');
    for (const m of misses) {
      const c = h('div', 'card miss');
      const t = h('div', 'miss-head'); t.append(h('b', null, m.column + ' '), h('code', null, m.cli));
      c.append(t, h('div', 'miss-num', `median ${m.median_dJND} JND (max ${m.max_dJND}) · raw median ${m.median_d} · Spearman ${m.spearman}`));
      const dl = h('dl');
      for (const [k, v] of m.cells) if (k !== 'Ours' && v) dl.append(h('dt', null, k), h('dd', null, v));
      c.append(dl);
      if (m.section) c.append(h('div', 'hint', m.section));
      if (!m.cells.length) c.append(h('div', 'mark-dev', 'No cause recorded in the status doc.'));
      grid.append(c);
    }
    box.append(grid);
    return box;
  }

  // ---- data
  function railData(rail) {
    rail.append(sizeSeg(), seg('Threads', 'th', [['1', '1 thread'], ['8', '8 threads']], { pressed: v => (S.th === '8' ? '8' : '1') === v }), phaseSeg(), seg('Unit', 'unit', [['px', 'ns per pixel'], ['img', 'ms per image']]), famSeg(), search());
    const box = h('div', 'ctl'); box.append(h('div', 'ctl-label', 'Download'));
    const b = h('button', null, 'This table (CSV)'); b.type = 'button'; b.onclick = downloadView;
    const seg2 = h('div', 'seg'); seg2.append(b);
    for (const [href, text] of [['timings.csv', 'All timings (CSV)'], ['complexity.csv', 'Counted costs (CSV)'], ['brief.pdf', 'Printable PDF']]) {
      const a = h('a', null, text); a.href = href; a.download = ''; a.className = 'icon'; a.style.textDecoration = 'none'; seg2.append(a);
    }
    box.append(seg2, h('div', 'hint', 'timings.csv has every cell: CPU, metric, size, threads, phase, median, mean, 95 % interval, spread, fastest, n, clean and flagged rounds, peak RSS, SIMD tier, placement.'));
    rail.append(box);
  }
  function dataRows() {
    const k = unitK(S.size), t = S.th === '8' ? 8 : 1;
    return NAMES.filter(m => famOn(m) && qOn(m)).map(m => ({ m, c: HOSTS.map(host => cell(host, m, S.size, t, S.ph)), k })).filter(r => r.c.some(Boolean));
  }
  function drawData(head, body) {
    const rows = dataRows(), u = unitShort();
    setHead(head, `Summary: ${unitName()} by metric and CPU`, `${sizeName(S.size)} · ${thName(S.th === '8' ? '8' : '1')} · ${S.ph} · median [95 % interval of the mean] (n) · ${rows.length} metrics · click a header to sort`, MARKS_LEGEND);
    const cols = [
      { id: 'm', label: 'metric', cell: (r, td) => { td.append(h('code', null, r.m)); const f = flagText(r.m); if (f) td.append(h('span', meta(r.m).status === 'deviation' ? 'mark-dev' : 'mark-un', ' ' + f)); }, sortv: r => r.m },
      { id: 'fam', label: 'family', cell: (r, td) => { td.textContent = FAMS.find(f => f[0] === meta(r.m).fam)?.[1] || ''; }, sortv: r => meta(r.m).fam },
      ...HOSTS.map((host, i) => ({ id: 'h' + i, label: `${hostName(host)} ${u}`, num: true, title: P.hosts[host],
        cell: (r, td) => { const c = r.c[i]; if (!c) { td.textContent = '–'; return; } td.append(h('b', null, fmt(c.median * r.k))); td.append(h('span', 'ci', (c.ci ? ` [${fmt(c.ci[0] * r.k)}, ${fmt(c.ci[1] * r.k)}]` : '') + ` (${c.n})`)); },
        sortv: r => r.c[i] ? r.c[i].median * r.k : null })),
      { id: 'ev', label: 'evidence', cell: (r, td) => evidenceCell(r.m, td), sortv: r => meta(r.m).status },
    ];
    if (!S.sort) S.sort = 'h' + HOSTS.indexOf(DEF.host) + ':a';
    sortable(body, cols, rows, r => openDrawer(r.m));
  }
  function downloadView() {
    const rows = dataRows(), q = v => /[",\n]/.test(v) ? `"${String(v).replace(/"/g, '""')}"` : v;
    const lines = [['metric', 'family', 'size', 'threads', 'phase', 'unit', ...HOSTS.flatMap(host => [`${host}_median`, `${host}_ci95_lo`, `${host}_ci95_hi`, `${host}_n`])].join(',')];
    for (const r of rows) lines.push([r.m, meta(r.m).fam, S.size, S.th, S.ph, unitShort(), ...r.c.flatMap(c => c ? [c.median * r.k, c.ci ? c.ci[0] * r.k : '', c.ci ? c.ci[1] * r.k : '', c.n].map(v => typeof v === 'number' ? +v.toPrecision(6) : v) : ['', '', '', ''])].map(q).join(','));
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], { type: 'text/csv' }));
    a.download = `metric-timings_${S.size}_${S.th}t_${S.ph}.csv`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // ---- method
  function mountMethod(app) {
    const src = D.sources, dv = D.dvifm.map(x => `<code>${esc(x.preset)}</code>`).join(', ');
    const o = h('div', 'prose');
    o.innerHTML = `
      <h3>What is timed</h3>
      <p>Each metric is the zenmetrics implementation (pure Rust), built once per host and identified by binary hash. Input: 8-bit sRGB RGB images already in memory, so file reading and decoding are outside the clock; colour conversion to whatever the metric uses (luma, YCbCr, XYB, linear RGB) is inside it.</p>
      <ul><li><b>Warm</b>: the scorer is constructed and warmed first, then at least ten rounds, each a batch of calls sized to about 100 ms. The median, spread (p10–p90) and 95 % interval are over the round batch means.</li>
      <li><b>Cold</b>: three fresh processes, each timing scorer construction plus the first full call: what a one-shot command-line call pays.</li>
      <li><b>Unit</b>: wall time per call divided by width × height. With 8 threads it is elapsed time, not CPU time summed over threads.</li>
      <li><b>Sizes</b>: ${SIZES.map(s => esc(sizeName(s.name))).join(', ')}. 840×944 is the crop size the AIC2026 subjective test showed at 1:1, so it is the default.</li></ul>
      <h3>How zenbench protects the timings</h3>
      <ul><li><b>Core placement on hybrid CPUs.</b> The Core Ultra 7 265K mixes performance and efficiency cores. On Linux each cell runs in a cgroup cpuset restricted to one core class (performance cores on that host), because an affinity mask alone cannot stop a library's thread pool from widening it. Placement is checked before and after every round. On macOS core affinity is only advisory, so M4 Pro runs are recorded as unrestricted.</li>
      <li><b>Busy detection.</b> A background sampler records whole-host CPU load (excluding the benchmark's own workers), available memory, temperature where a sensor exists, and foreign heavy processes. A round over the limits is flagged, not dropped; a warm cell qualifies only with ten gate-clean rounds. Flagged rounds show red in the Speed tab.</li>
      <li><b>Admission and serialization.</b> Jobs take a shared heavy-job lock and zenbench's first-in-first-out admission before warmup, so two benchmarks never time each other. A launch refused for high load waits and retries; the refusal is kept as a row, never as a timing.</li>
      <li><b>Jitter and order.</b> Iteration counts vary ±20 % and arm order is shuffled per round.</li>
      <li><b>Resource limits and identity.</b> Hard memory cap (12 GiB on Linux), outer timeout, OS-measured peak RSS. Every row carries binary, input and engine hashes; different binaries are never pooled, and a cell's score must be bit-identical across its warmup, calibration and timed calls.</li></ul>
      <p class="hint">Limits: a cpuset is a boundary, not a reservation; other work can still share those cores' SMT siblings, caches and memory bandwidth. Load is checked at round boundaries, not continuously. macOS placement cannot be certified.</p>
      <h3>Counted cost</h3>
      <p>Operations per pixel tallied from each implementation's source and weighted by ${D.tables.length} cost tables (flop count, the Pedzisz draft weights, scalar costs measured on the ${esc(D.fit_host)}, LLVM cost models, llvm-mca). Compared with CPU time from <code>perf stat</code> on the ${esc(D.fit_host)} at ${esc(D.fit_shape)}, one content pair, three repeats. Counts ignore vectorization, memory traffic, caches and threading. Several neural tallies await recounts; DVIFM counts are core-only aggregates.</p>
      <h3>Marks</h3>
      <p><span class="mark-dev">*</span> a recorded deviation: an AIC2026 column that is close but not exact or not reproduced, a primary-oracle finding, or partial reference parity. <span class="mark-un">°</span> no parity record yet (not a known difference). <span class="mark-un">†</span> DVIFM releases the author retuned on AIC2026 data after the published columns were computed.</p>
      <h3>DVIFM versions</h3>
      <p>dvifmish ports M. Pedzisz's four released C++ variants (${dv}) and adds presets reconstructed from his talk. dvifmish is not published; only aggregate results appear here.</p>
      <h3>Not shown</h3>
      <p>Agreement with subjective scores. The July numbers came from a preliminary JND reconstruction and are superseded; they return once the full-dataset reproduction of the published objective columns is checked. Reference-implementation speed is not shown either: the reference implementations have not yet been timed on the same hosts.</p>
      <h3>Sources</h3>
      <p>zenmetrics <code>${esc(src.public)}</code>, zenmetrics-private <code>${esc(src.private || '–')}</code>, dvifmish <code>${esc(src.dvifmish || '–')}</code>; timing protocol <code>${esc(P.protocol || '–')}</code>; timings synced ${esc((P.synced_utc || '').slice(0, 16).replace('T', ' '))} UTC. Generated by <code>scripts/complexity/brief_data.py</code> and <code>brief_page.py</code>. This page makes no network requests.${D.notes.length ? ' Notes: ' + esc(D.notes.join('; ')) : ''}</p>`;
    app.append(o);
  }

  // ------------------------------------------------------------------ shell
  function shell() {
    const tabs = $('#tabs');
    VIEWS.forEach((v, i) => {
      const b = h('button'); b.type = 'button'; b.dataset.v = v.id; b.append(document.createTextNode(v.name), h('kbd', null, String(i + 1)));
      b.onclick = () => set({ v: v.id }); tabs.append(b);
    });
    const C = coverageSummary();
    const pill = $('#datapill'); pill.textContent = `${C.final ? 'final' : 'unfinished'} · ${C.cells ? Math.round(100 * C.ok / C.cells) : 0} % of cells timed · synced ${(P.synced_utc || '').slice(11, 16)} UTC`;
    pill.title = C.hosts.join('\n');
    $('#fullscreen').onclick = toggleFs;
    const ck = D.checks || [], nerr = ck.filter(c => c.level === 'error').length, nwarn = ck.filter(c => c.level === 'warn').length;
    if (nerr || nwarn) {
      const b = $('#checkpill'); b.hidden = false; b.classList.toggle('err', nerr > 0);
      b.textContent = (nerr ? `${nerr} error${nerr > 1 ? 's' : ''} · ` : '') + `${nwarn} data warning${nwarn === 1 ? '' : 's'}`;
      b.onclick = openChecks;
    }
    document.addEventListener('keydown', ev => {
      if (ev.target.closest && ev.target.closest('input')) return;
      if (ev.key === 'Escape') { DRAWER.hidden = true; return; }
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      const i = VIEWS.findIndex(v => v.id === S.v);
      if (/^[1-7]$/.test(ev.key) && VIEWS[+ev.key - 1]) set({ v: VIEWS[+ev.key - 1].id });
      else if (ev.key === 'ArrowRight') set({ v: VIEWS[(i + 1) % VIEWS.length].id });
      else if (ev.key === 'ArrowLeft') set({ v: VIEWS[(i + VIEWS.length - 1) % VIEWS.length].id });
      else if (ev.key === 'f' || ev.key === 'F') toggleFs();
    });
    let raf = 0;
    new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); }).observe($('#app'));
    addEventListener('hashchange', () => { const v = S.v; readHash(); v === S.v ? (syncControls(), draw()) : mount(); });
  }
  function toggleFs() { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.(); }

  function printAll() {
    document.documentElement.classList.add('print');
    const app = $('#app');
    for (const v of VIEWS) {
      const page = h('section', 'page' + (['repro', 'data', 'method'].includes(v.id) ? ' flow' : ''));
      app.append(page);
      S.v = v.id; S.sort = '';
      const c = mountInto(page, v);
      if (v.draw) v.draw(c.head, c.body);
    }
    syncControls();
  }

  readHash();
  shell();
  if (PRINT) printAll(); else mount();
})();
