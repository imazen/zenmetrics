// View controls for the brief: weight-table selector, 1/8-thread toggle, and slide mode.
(function () {
  const root = document.documentElement;
  const sel = document.getElementById('tbl');
  const threads = document.getElementById('threads');
  const slides = () => [...document.querySelectorAll('section.slide')];
  let cur = 0;

  function showTable(id) {
    document.querySelectorAll('.coefset').forEach(e => { e.hidden = e.dataset.table !== id; });
    const name = sel.options[sel.selectedIndex].text;
    document.querySelectorAll('.tblname').forEach(e => { e.textContent = name; });
  }
  sel.addEventListener('change', () => showTable(sel.value));

  function setThreads(eight) {
    root.classList.toggle('t8', eight);
    threads.setAttribute('aria-pressed', String(eight));
  }
  threads.addEventListener('click', () => setThreads(!root.classList.contains('t8')));

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
    if (!root.classList.contains('slides') || e.target.tagName === 'SELECT' || e.altKey || e.ctrlKey || e.metaKey) return;
    const k = e.key.toLowerCase();
    if (['arrowright', 'pagedown', ' ', 'arrowdown'].includes(k)) { go(cur + 1); e.preventDefault(); }
    else if (['arrowleft', 'pageup', 'arrowup'].includes(k)) { go(cur - 1); e.preventDefault(); }
    else if (k === 'home') go(0);
    else if (k === 'end') go(slides().length - 1);
    else if (k === 't') setThreads(!root.classList.contains('t8'));
    else if (k === 'w') { sel.selectedIndex = (sel.selectedIndex + 1) % sel.options.length; showTable(sel.value); }
    else if (k === 'f') { document.fullscreenElement ? document.exitFullscreen() : root.requestFullscreen && root.requestFullscreen(); }
    else if (k === 'escape') present(false);
  });
  // Track the visible slide when the presenter scrolls instead of using keys.
  const io = new IntersectionObserver(es => es.forEach(x => { if (x.isIntersecting && x.intersectionRatio > 0.6) cur = slides().indexOf(x.target); }), { threshold: [0.6] });
  slides().forEach(s => io.observe(s));

  window.addEventListener('beforeprint', () => document.querySelectorAll('details').forEach(d => { d.dataset.wasOpen = d.open; d.open = true; }));
  window.addEventListener('afterprint', () => document.querySelectorAll('details').forEach(d => { d.open = d.dataset.wasOpen === 'true'; }));
  if (location.hash === '#slides') present(true);
  showTable(sel.value);
})();
