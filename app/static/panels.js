/* Layout: the side panels resize and fold, the panels on the viewer move and fold, the minimap also resizes, and the
   sections of the review panel fold. Kept per browser. Whatever the window size, the side panels stay on screen: if both
   do not fit, the one opened less recently folds to its rail until there is room again. Loaded after app.js. */
'use strict';

const Panels = (() => {
  const KEY = 'hss_layout_v1', RAIL = 26, DIVIDER = 6, MIN_CENTER = 300, MIN_VIEW = 120, NARROW = 720, STEP = 20;
  const COMPACT_VIEW = 520, FOLD_WHEN_COMPACT = ['tools', 'legend', 'map'];   // in a small viewer these start folded, so the slide can be seen
  const SIDES = {
    left: { name: 'Slides', key: ',', def: 250, min: 160, max: 520 },
    right: { name: 'Review', key: '.', def: 340, min: 260, max: 820 },
  };
  const MAP_DEF = { w: 190, h: 120 }, MAP_MIN = 100;
  const blank = () => ({ left: { w: 250, folded: false }, right: { w: 340, folded: false }, last: 'right', float: {}, sec: {} });
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  let st = blank();
  try {
    const saved = JSON.parse(localStorage.getItem(KEY));
    if (saved) st = { ...st, ...saved, left: { ...st.left, ...saved.left }, right: { ...st.right, ...saved.right } };
  } catch (e) { /* storage blocked or unreadable: defaults */ }
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) { /* ignore */ } };

  const main = $('#roiMain'), wrap = $('#viewerWrap');
  const floats = [...document.querySelectorAll('.fp')];
  const open = { left: true, right: true };   // what is shown now, which can differ from st while the window is too small

  /* ---------------------------------------------------------- side panels */
  const other = (s) => (s === 'left' ? 'right' : 'left');
  const wanted = (s) => clamp(st[s].w, SIDES[s].min, SIDES[s].max);
  const minCenter = () => (main.clientWidth >= NARROW ? MIN_CENTER : 0);
  const room = (s) => main.clientWidth - minCenter() - DIVIDER - (open[other(s)] ? wanted(other(s)) + DIVIDER : RAIL);

  function applySides() {
    open.left = !st.left.folded; open.right = !st.right.folded;
    const both = wanted('left') + wanted('right') + 2 * DIVIDER + minCenter();
    if (main.clientWidth < NARROW || both > main.clientWidth) open[other(st.last)] = false;   // no room for two: the latest opened stays
    for (const s of ['left', 'right']) {
      const pre = s === 'left' ? 'l' : 'r';
      let w = open[s] ? Math.max(0, Math.min(wanted(s), room(s))) : RAIL;
      if (open[s] && main.clientWidth < NARROW && room(s) - w < MIN_VIEW) w = room(s);   // too little left for a usable slide: take it all
      main.style.setProperty(`--${pre}w`, `${w}px`);
      main.style.setProperty(`--${pre}d`, open[s] ? `${DIVIDER}px` : '0px');
      $(s === 'left' ? '#left' : '#right').classList.toggle('folded', !open[s]);
      const div = $(s === 'left' ? '#leftResizer' : '#rightResizer');
      div.setAttribute('aria-valuenow', Math.round(w));
      div.setAttribute('aria-valuemin', SIDES[s].min);
      div.setAttribute('aria-valuemax', SIDES[s].max);
    }
    renderMenu();
  }
  function toggleSide(s) {
    if (open[s]) st[s].folded = true;
    else { st[s].folded = false; st.last = s; }
    save(); applySides();
  }
  function setWidth(s, w) {
    st[s].w = clamp(w, SIDES[s].min, Math.max(SIDES[s].min, Math.min(SIDES[s].max, room(s))));
    save(); applySides();
  }
  function bindDivider(el, s) {
    const sign = s === 'left' ? 1 : -1;
    el.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      const x0 = e.clientX, w0 = wanted(s);
      document.body.classList.add('resizing'); el.classList.add('active');
      const move = (m) => setWidth(s, w0 + sign * (m.clientX - x0));
      const up = () => {
        el.removeEventListener('pointermove', move); el.removeEventListener('pointerup', up); el.removeEventListener('pointercancel', up);
        document.body.classList.remove('resizing'); el.classList.remove('active');
      };
      el.addEventListener('pointermove', move); el.addEventListener('pointerup', up); el.addEventListener('pointercancel', up);
    });
    el.addEventListener('dblclick', () => setWidth(s, SIDES[s].def));
    el.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        setWidth(s, wanted(s) + sign * (e.key === 'ArrowRight' ? STEP : -STEP));
      }
    });
  }

  /* ---------------------------------------------------------- panels on the viewer */
  const fpState = (id) => st.float[id] || (st.float[id] = {});
  const isFolded = (id) => {   // an explicit choice wins, otherwise it depends on the room
    const f = fpState(id).folded;
    return f === undefined ? FOLD_WHEN_COMPACT.includes(id) && wrap.clientWidth < COMPACT_VIEW : f;
  };
  const placed = (el) => st.float[el.dataset.fp] && st.float[el.dataset.fp].x != null;
  const headH = (el) => el.querySelector('.fp-head').offsetHeight;

  function place(el) {
    const s = fpState(el.dataset.fp), folded = isFolded(el.dataset.fp);
    el.classList.toggle('folded', folded);
    const b = el.querySelector('.fp-fold');
    b.textContent = folded ? '▸' : '▾';
    b.setAttribute('aria-expanded', String(!folded));
    b.title = `${folded ? 'Unfold' : 'Fold'} ${el.dataset.title}`;
    if (s.x != null) {   // always within reach, however small the viewer has become
      el.style.left = `${clamp(s.x, 0, Math.max(0, wrap.clientWidth - el.offsetWidth))}px`;
      el.style.top = `${clamp(s.y, 0, Math.max(0, wrap.clientHeight - headH(el)))}px`;
      el.style.right = 'auto'; el.style.bottom = 'auto';
    } else {
      el.style.left = ''; el.style.top = ''; el.style.right = ''; el.style.bottom = '';
    }
    if (el.dataset.fp === 'map') {
      const body = $('#navBody');
      const w = s.w || MAP_DEF.w, h = s.h || MAP_DEF.h;
      body.style.width = `${clamp(w, MAP_MIN, Math.max(MAP_MIN, wrap.clientWidth - 20))}px`;
      body.style.height = `${clamp(h, MAP_MIN / 2, Math.max(MAP_MIN / 2, wrap.clientHeight - 40))}px`;
    }
  }
  let stackTop = 74;
  function stackDefaults() {   // panels left in their default place stack under each other as sizes change
    const tools = $('#tools'), box = $('#toolbox'), opts = $('#toolOpts');
    if (!placed(tools)) stackTop = tools.offsetTop + tools.offsetHeight + 6;
    const top = stackTop;
    if (!placed(box)) box.style.top = `${top}px`;
    if (!placed($('#roiTools'))) $('#roiTools').style.top = `${top}px`;
    if (!placed($('#patchTools'))) $('#patchTools').style.top = `${top}px`;
    if (!placed(opts)) {
      opts.style.top = `${top}px`;
      opts.style.left = `${placed(box) ? 10 : box.offsetLeft + box.offsetWidth + 8}px`;
    }
  }
  function placeAll() { floats.forEach(place); stackDefaults(); }

  let topZ = 5;
  function initFloat(el) {
    const head = document.createElement('div');
    head.className = 'fp-head';
    head.title = 'Drag to move, double-click to put it back';
    head.innerHTML = `<span class="fp-title">${el.dataset.title}</span><button type="button" class="fp-fold"></button>`;
    el.prepend(head);
    el.addEventListener('pointerdown', () => { el.style.zIndex = ++topZ; });   // the panel in use comes to the front
    head.querySelector('.fp-fold').addEventListener('click', (e) => { e.currentTarget.blur(); toggleFloat(el.dataset.fp); });
    head.addEventListener('dblclick', (e) => {
      if (e.target.closest('button')) return;
      const s = fpState(el.dataset.fp);
      delete s.x; delete s.y;
      save(); placeAll();
    });
    head.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || e.target.closest('button')) return;
      e.preventDefault();
      head.setPointerCapture(e.pointerId);
      const s = fpState(el.dataset.fp), x0 = e.clientX - el.offsetLeft, y0 = e.clientY - el.offsetTop;
      const move = (m) => { s.x = m.clientX - x0; s.y = m.clientY - y0; placeAll(); };
      const up = () => {
        head.removeEventListener('pointermove', move); head.removeEventListener('pointerup', up); head.removeEventListener('pointercancel', up);
        if (s.x != null) { s.x = el.offsetLeft; s.y = el.offsetTop; }
        save();
      };
      head.addEventListener('pointermove', move); head.addEventListener('pointerup', up); head.addEventListener('pointercancel', up);
    });
  }
  function toggleFloat(id) {
    fpState(id).folded = !isFolded(id);
    save(); placeAll(); renderMenu();
  }
  function bindMapResize() {
    const grip = $('#mapPanel .fp-resize');
    grip.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      e.preventDefault(); e.stopPropagation();
      grip.setPointerCapture(e.pointerId);
      const el = $('#mapPanel'), body = $('#navBody'), s = fpState('map');
      const x0 = e.clientX, y0 = e.clientY, w0 = body.offsetWidth, h0 = body.offsetHeight;
      if (s.x == null) { s.x = el.offsetLeft; s.y = el.offsetTop; }   // from here on it grows away from its top left corner
      const move = (m) => { s.w = Math.max(MAP_MIN, w0 + m.clientX - x0); s.h = Math.max(MAP_MIN / 2, h0 + m.clientY - y0); place(el); };
      const up = () => {
        grip.removeEventListener('pointermove', move); grip.removeEventListener('pointerup', up); grip.removeEventListener('pointercancel', up);
        s.w = body.offsetWidth; s.h = body.offsetHeight;
        save();
      };
      grip.addEventListener('pointermove', move); grip.addEventListener('pointerup', up); grip.addEventListener('pointercancel', up);
    });
    new ResizeObserver(() => { if (viewer.navigator) viewer.navigator.updateSize(); }).observe($('#navBody'));
  }

  /* ---------------------------------------------------------- sections of the review panel */
  const secOpen = (id) => !st.sec[id];
  function toggleSec(id) { st.sec[id] = !st.sec[id]; save(); }
  function showSec(id) {   // returns true if it had to be unfolded
    if (!st.sec[id]) return false;
    st.sec[id] = false; save();
    return true;
  }
  function showReview() { if (!open.right) toggleSide('right'); }

  /* ---------------------------------------------------------- layout menu */
  const MENU_FLOATS = [['tools', 'Overlay'], ['toolbox', 'Tools', 'segments'], ['roitools', 'Tools', 'rois'], ['patchtools', 'Tools', 'patches'], ['legend', 'Legend'], ['map', 'Minimap']];
  function renderMenu() {
    const m = $('#layoutMenu');
    const item = (act, on, label, key) => `<button type="button" class="menu-item" data-act="${act}" role="menuitemcheckbox" aria-checked="${on}"><span class="check">${on ? '✓' : ''}</span><span class="spacer">${label}</span>${key ? `<kbd>${key}</kbd>` : ''}</button>`;
    m.innerHTML = Object.entries(SIDES).map(([s, d]) => item(`side:${s}`, open[s], d.name, d.key)).join('') +
      MENU_FLOATS.filter(([, , mode]) => !mode || mode === document.body.dataset.mode).map(([id, label]) => item(`float:${id}`, !isFolded(id), label)).join('') +
      '<div class="tool-sep"></div><button type="button" class="menu-item" data-act="reset"><span class="check"></span><span class="spacer">Reset layout</span></button>';
  }
  function reset() {
    st = blank();
    save();
    $('#navBody').style.width = ''; $('#navBody').style.height = '';
    applySides(); placeAll();
    if (typeof renderSR === 'function' && SR.data) renderSR();
    if (document.body.dataset.mode === 'rois') roiRender();
    if (document.body.dataset.mode === 'patches') paRender();
    toast('Layout reset');
  }
  $('#layoutBtn').addEventListener('click', (e) => {
    e.stopPropagation();
    $('#exportMenu').classList.add('hidden');
    $('#layoutMenu').classList.toggle('hidden');
  });
  $('#layoutMenu').addEventListener('click', (e) => {
    e.stopPropagation();
    const b = e.target.closest('[data-act]');
    if (!b) return;
    const [kind, id] = b.dataset.act.split(':');
    if (kind === 'side') toggleSide(id);
    else if (kind === 'float') toggleFloat(id);
    else { reset(); $('#layoutMenu').classList.add('hidden'); }
  });
  document.addEventListener('click', () => $('#layoutMenu').classList.add('hidden'));

  /* ---------------------------------------------------------- start */
  document.querySelectorAll('[data-fold-side]').forEach((b) => b.addEventListener('click', () => { b.blur(); toggleSide(b.dataset.foldSide); }));
  bindDivider($('#leftResizer'), 'left');
  bindDivider($('#rightResizer'), 'right');
  floats.forEach(initFloat);
  bindMapResize();
  applySides(); placeAll();
  let queued = false;
  const onResize = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; applySides(); placeAll(); });
  };
  window.addEventListener('resize', onResize);
  new ResizeObserver(onResize).observe(wrap);
  new ResizeObserver(stackDefaults).observe($('#tools'));

  const refresh = () => { placeAll(); renderMenu(); };   // the mode changed, and with it which panels are shown
  return { secOpen, toggleSec, showSec, showReview, toggleSide, refresh };
})();
