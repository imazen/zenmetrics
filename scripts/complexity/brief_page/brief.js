// View controls for the brief: weight-table buttons, one shared 1/8-thread state, and slide mode.
(function () {
  const root = document.documentElement;
  const tbl = document.getElementById('tbl');
  const slides = () => [...document.querySelectorAll('section.slide')];
  let cur = 0;

  function showTable(id) {
    document.querySelectorAll('.coefset').forEach(e => { e.hidden = e.dataset.table !== id; });
    tbl.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.table === id)));
    const b = tbl.querySelector(`button[data-table="${id}"]`);
    document.querySelectorAll('.tblname').forEach(e => { e.textContent = b ? b.title : id; });
  }
  tbl.addEventListener('click', e => { const b = e.target.closest('button'); if (b) showTable(b.dataset.table); });
  const tableIds = () => [...tbl.querySelectorAll('button')].map(b => b.dataset.table);
  const curTable = () => (tbl.querySelector('button[aria-pressed="true"]') || tbl.querySelector('button')).dataset.table;

  // Every .threads-seg holds the same two buttons; all charts follow one state.
  document.querySelectorAll('.threads-seg').forEach(seg => {
    seg.innerHTML = '<button type="button" data-t="1">1 thread</button><button type="button" data-t="8">8 threads</button>';
    seg.addEventListener('click', e => { const b = e.target.closest('button'); if (b) setThreads(b.dataset.t === '8'); });
  });
  function setThreads(eight) {
    root.classList.toggle('t8', eight);
    document.querySelectorAll('.threads-seg button').forEach(b => b.setAttribute('aria-pressed', String((b.dataset.t === '8') === eight)));
    document.dispatchEvent(new Event('threadschange'));
  }

  function go(i) {
    const s = slides();
    cur = Math.max(0, Math.min(s.length - 1, i));
    s[cur].scrollIntoView({ block: 'start' });
    s[cur].scrollTop = 0;
  }
  function present(on) {
    root.classList.toggle('slides', on);
    if (on) { go(0); if (location.hash !== '#slides') history.replaceState(null, '', '#slides'); }
    else if (location.hash === '#slides') history.replaceState(null, '', location.pathname + location.search);
  }
  document.getElementById('present').addEventListener('click', () => present(!root.classList.contains('slides')));

  document.addEventListener('keydown', e => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const k = e.key.toLowerCase();
    if (k === 't') { setThreads(!root.classList.contains('t8')); return; }
    if (k === 'w') { const ids = tableIds(); showTable(ids[(ids.indexOf(curTable()) + 1) % ids.length]); return; }
    if (!root.classList.contains('slides')) return;
    if (['arrowright', 'pagedown', ' ', 'arrowdown'].includes(k)) { go(cur + 1); e.preventDefault(); }
    else if (['arrowleft', 'pageup', 'arrowup'].includes(k)) { go(cur - 1); e.preventDefault(); }
    else if (k === 'home') go(0);
    else if (k === 'end') go(slides().length - 1);
    else if (k === 'f') { document.fullscreenElement ? document.exitFullscreen() : root.requestFullscreen && root.requestFullscreen(); }
    else if (k === 'escape') present(false);
  });
  const io = new IntersectionObserver(es => es.forEach(x => { if (x.isIntersecting && x.intersectionRatio > 0.6) cur = slides().indexOf(x.target); }), { threshold: [0.6] });
  slides().forEach(s => io.observe(s));

  window.addEventListener('beforeprint', () => document.querySelectorAll('details').forEach(d => { d.dataset.wasOpen = d.open; d.open = true; }));
  window.addEventListener('afterprint', () => document.querySelectorAll('details').forEach(d => { d.open = d.dataset.wasOpen === 'true'; }));
  if (location.hash === '#slides') present(true);
  setThreads(false);
  showTable(curTable());
})();
