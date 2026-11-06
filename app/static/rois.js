/* Picking ROIs: draw a polygon, square or oval on a slide, say what it shows in the review panel, and see the ROIs picked so
   far in a gallery. Each reviewer works in a workspace of their own: only the ROIs under their initials are listed. Works
   without a segmentation, and does not depend on the segmentation run. Loaded after editor.js. */
'use strict';

const ROI_KEYS = ['1', '2', '3', '4', '5', '6', 'e', '7', '8', '9', '0'];   // one for each label, in the order of the list; E for Epithelioid, so the keys already learned stay
const MIN_SIDE = 32, MIN_DRAG_PX = 8;   // level-0 pixels (as on the server) and screen pixels
const OVAL_POINTS = 72;
const ROI_TOOLS = {
  move: { label: 'Move', key: 'V', icon: '<path d="M12 3l3 3h-2v5h5V9l3 3-3 3v-2h-5v5h2l-3 3-3-3h2v-5H6v2l-3-3 3-3v2h5V6H9z"/>' },
  polygon: { label: 'Polygon', key: 'D', hint: 'Click around the area. Double-click, Enter or a click on the first point closes it. Backspace removes a point.', icon: '<path d="M6 5l13 3-3 11-11-3z" fill="none" stroke="currentColor" stroke-width="2"/>' },
  square: { label: 'Square', key: 'S', hint: 'Drag from corner to corner. Hold Shift for a rectangle.', icon: '<rect x="5" y="5" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"/>' },
  oval: { label: 'Oval', key: 'O', hint: 'Drag across the area. Hold Shift for a circle.', icon: '<ellipse cx="12" cy="12" rx="9" ry="6.5" fill="none" stroke="currentColor" stroke-width="2"/>' },
};

const Roi = {
  owner: reviewerInput.value.trim(),   // whose workspace is on screen
  labels: [], rois: [], loaded: false,
  tool: 'move', space: false, busy: false,
  drag: null,       // while drawing, in slide pixels: { shape, pts, shift }; a polygon gains pts click by click
  cursor: null,     // the pointer while a polygon is being drawn
  draft: null,      // drawn but not saved: { shape, ring }
  sel: null,        // id of the saved ROI open in the panel
  form: null,       // what the panel holds for the draft or the selected ROI: { labels: Set, comment }
  hover: null, press: null,
  deleted: null,    // the ROI just deleted, offered for Undo
  filter: { label: null },   // gallery
  segView: { heat: 'all@clusters', lines: true }, roiView: { heat: 'none', lines: false },   // each mode keeps its own overlay
};
const modeIs = (m) => document.body.dataset.mode === m;
const roisHere = () => Roi.rois.filter((r) => r.slide_id === S.sid).sort((a, b) => a.n - b.n);
const roiById = (id) => Roi.rois.find((r) => r.id === id);
const roiCount = (sid) => Roi.rois.reduce((n, r) => n + (r.slide_id === sid), 0);
const labelOf = (k) => Roi.labels.find((l) => l.key === k);
const labelText = (k) => { const l = labelOf(k); return l ? l.short || l.name : k; };
const inOrder = (set) => Roi.labels.map((l) => l.key).filter((k) => set.has(k));
const roiThumb = (id) => `api/expert_rois/${encodeURIComponent(id)}/thumb.jpg`;
const roiDay = (ts) => esc((ts || '').slice(0, 10));
function sizeText(r) {
  if (r.width_um == null) return '';
  const u = (v) => (v >= 1000 ? `${(v / 1000).toFixed(2)} mm` : `${Math.round(v)} µm`);
  return `${u(r.width_um)} × ${u(r.height_um)}`;
}
const labelChips = (keys) => keys.map((k) => `<span class="rchip"><i style="background:${esc((labelOf(k) || {}).color || '#fff')}"></i>${esc(labelText(k))}</span>`).join('');

/* ------------------------------------------------------------ data */
const ownerKey = (s) => (s || '').trim().toLowerCase();
const mine = (kind) => `api/export/expert_rois.${kind}?reviewer=${encodeURIComponent(Roi.owner)}`;
async function roiLoad() {
  const owner = Roi.owner;
  try {
    const d = await api(`api/expert_rois?reviewer=${encodeURIComponent(owner)}`);
    if (owner !== Roi.owner) return;   // the initials changed meanwhile: that load is the one to show
    Roi.labels = d.labels; Roi.rois = d.rois; Roi.loaded = true;
  } catch (e) { toast('Could not load the ROIs: ' + errDetail(e), true); }
  roiRefresh();
}
/* New initials open that reviewer's workspace. Whatever is open belongs to the workspace it was drawn in. */
function roiSetOwner() {
  const next = reviewerInput.value.trim();
  if (next === Roi.owner) return;
  if (ownerKey(next) !== ownerKey(Roi.owner)) {
    dropPartial(); roiFlush();
    Roi.rois = []; Roi.deleted = null; Roi.hover = null; Roi.filter.label = null; Roi.loaded = false;
  }
  Roi.owner = next;
  roiRefresh();
  roiLoad();
}
function roiWho() {   // the initials to work under, asking for them if there are none
  if (!reviewer()) return '';
  roiSetOwner();
  return Roi.owner;
}
reviewerInput.addEventListener('change', roiSetOwner);
function roiRefresh() {   // everything that shows the ROIs
  $('#roiCount').textContent = Roi.rois.length ? `(${Roi.rois.length})` : '';
  if (S.slides.length && roiMode()) { renderSlideList(); renderProgress(); }
  if (modeIs('rois')) roiRender();
  if (modeIs('all')) roiGallery();
  edDraw();
}
const formDirty = () => {
  if (!Roi.form) return false;
  if (Roi.draft) return Roi.form.labels.size > 0 || !!Roi.form.comment.trim();
  const r = roiById(Roi.sel);
  return !!r && (inOrder(Roi.form.labels).join() !== r.labels.join() || Roi.form.comment.trim() !== r.comment);
};
function roiClose() { Roi.draft = null; Roi.sel = null; Roi.form = null; }

async function sendRoi(sid, draft, selId, form, who) {
  const labels = inOrder(form.labels);
  if (draft) {
    return post('api/expert_rois', { reviewer: who, slide_id: sid, shape: draft.shape, ring: draft.ring, labels, comment: form.comment });
  }
  await post(`api/expert_rois/${encodeURIComponent(selId)}`, { reviewer: who, action: 'annotate', labels, comment: form.comment });
  return roiById(selId);
}
async function roiSave() {
  if (!Roi.form || Roi.busy) return;
  const who = roiWho();
  if (!who || !Roi.form) return;   // changing the initials just now saved what was open, in the workspace it was drawn in
  if (!Roi.form.labels.size) { toast('Choose at least one label first', true); return; }
  const was = roiById(Roi.sel);
  if (was && !formDirty()) { roiClose(); roiRender(); edDraw(); return; }
  Roi.busy = true; roiRender();
  let saved = null;
  try {
    saved = await sendRoi(S.sid, Roi.draft, Roi.sel, Roi.form, who);
  } catch (e) { toast('Not saved: ' + errDetail(e), true); }
  Roi.busy = false;
  if (saved) {
    const text = inOrder(Roi.form.labels).map(labelText).join(', ');
    roiClose(); Roi.deleted = null;
    await roiLoad();
    toast(`ROI ${saved.n} · ${text} · saved as ${who}`);
  } else roiRender();
}
/* Moving on with something open keeps what was chosen, as long as it has a label: nobody has to remember to press Save
   before going to the next slide. Without a label there is nothing to keep yet. */
function roiFlush() {
  if (!Roi.form) return;
  const dirty = formDirty(), form = Roi.form, draft = Roi.draft, sel = Roi.sel, sid = S.sid;
  roiClose();
  if (!dirty && !draft) return;
  const who = Roi.owner;
  if (!form.labels.size) {   // said out loud: a shape someone drew should not vanish silently
    toast('ROI not saved: it had no label', true);
    return;
  }
  sendRoi(sid, draft, sel, form, who)
    .then((saved) => { toast(`ROI ${saved ? saved.n : ''} saved as ${who}`); return roiLoad(); })
    .catch((e) => toast('ROI not saved: ' + errDetail(e), true));
}
async function roiDelete() {
  const r = roiById(Roi.sel), who = Roi.owner;
  if (!r || !who || Roi.busy) return;
  try {
    await post(`api/expert_rois/${encodeURIComponent(r.id)}`, { reviewer: who, action: 'delete' });
  } catch (e) { toast('Not deleted: ' + errDetail(e), true); return; }
  roiClose();
  Roi.deleted = { id: r.id, n: r.n, sid: r.slide_id };
  await roiLoad();
}
async function roiRestore() {
  const d = Roi.deleted, who = Roi.owner;
  if (!d || !who) return;
  try {
    await post(`api/expert_rois/${encodeURIComponent(d.id)}`, { reviewer: who, action: 'restore' });
  } catch (e) { toast('Not restored: ' + errDetail(e), true); return; }
  Roi.deleted = null;
  await roiLoad();
  toast(`ROI ${d.n} restored`);
}
window.addEventListener('beforeunload', (e) => { if (formDirty()) e.preventDefault(); });

function dropPartial() {   // an unfinished polygon cannot be kept
  if (Roi.drag && Roi.drag.shape === 'polygon' && Roi.drag.pts.length >= 3) toast('Unfinished polygon discarded. Close it first to keep it.', true);
  Roi.drag = null; Roi.cursor = null;
}
function roiOnSlide() {   // a different slide is opening; whatever was open has been flushed by now
  dropPartial(); Roi.hover = null; Roi.press = null;
  delete $('#viewerWrap').dataset.hot;
  if (modeIs('rois')) { roiRender(); $('#rpPanel').scrollTop = 0; }
}

/* ------------------------------------------------------------ selecting */
function roiSelect(id, zoom) {
  const r = roiById(id);
  if (!r) return;
  if (Roi.sel !== id) {
    roiFlush();
    Roi.sel = id;
    Roi.form = { labels: new Set(r.labels), comment: r.comment };
  }
  Panels.showReview();
  roiRender();
  $('#rpPanel').scrollTop = 0;
  if (zoom) roiZoom(r);
  edDraw();
}
function roiZoom(r) {
  const b = r.bbox, m = 0.3;
  viewer.viewport.fitBounds(imgRect({ x: b.x - b.w * m, y: b.y - b.h * m, w: b.w * (1 + 2 * m), h: b.h * (1 + 2 * m) }));
}
function roiAt(p) {   // smallest ROI under the point, so one drawn inside another can be picked
  let best = null, area = Infinity;
  for (const r of roisHere()) {
    const b = r.bbox;
    if (p[0] < b.x || p[0] > b.x + b.w || p[1] < b.y || p[1] > b.y + b.h || !insideRing(p, r.ring)) continue;
    if (b.w * b.h < area) { best = r; area = b.w * b.h; }
  }
  return best;
}
function roiToggle(key) {
  if (!Roi.form) return;
  if (Roi.form.labels.has(key)) Roi.form.labels.delete(key); else Roi.form.labels.add(key);
  const b = document.querySelector(`#rpPanel [data-label="${key}"]`);   // in place: re-rendering would reload the thumbnails below
  if (b) { b.classList.toggle('on', Roi.form.labels.has(key)); b.setAttribute('aria-pressed', Roi.form.labels.has(key)); }
  edDraw();
}

/* ------------------------------------------------------------ review panel */
function formHtml() {
  const f = Roi.form, r = roiById(Roi.sel), keyOf = (k) => ROI_KEYS[Roi.labels.findIndex((l) => l.key === k)];
  const btn = (l) => `<button type="button" class="roi-opt ${f.labels.has(l.key) ? 'on' : ''}" data-label="${esc(l.key)}" aria-pressed="${f.labels.has(l.key)}"><i style="background:${esc(l.color)}"></i><span>${esc(l.name)}</span><kbd>${keyOf(l.key)}</kbd></button>`;
  const groups = [...new Set(Roi.labels.map((l) => l.group).filter(Boolean))];
  return `<div class="rform ${Roi.draft ? 'fresh' : ''}">
      <div class="rform-head"><b>${r ? `ROI ${r.n}` : 'New ROI'}</b>
        ${r ? `<span class="muted small">${esc(sizeText(r))}</span><span class="spacer"></span><button class="ghost zoom" id="rpZoom" title="Zoom to this ROI">⌖</button>` : ''}</div>
      <div class="rform-q">What does it show? <span class="muted">Choose one or more.</span></div>
      <div class="roi-opts">${Roi.labels.filter((l) => !l.group).map(btn).join('')}</div>
      ${groups.map((g) => `<div class="rform-group">${esc(g)}</div><div class="roi-opts two">${Roi.labels.filter((l) => l.group === g).map(btn).join('')}</div>`).join('')}
      <textarea id="rpComment" placeholder="Comment (optional)" aria-label="Comment on this ROI" maxlength="4000">${esc(f.comment)}</textarea>
      <div class="save-row"><button class="primary" id="rpSave" ${Roi.busy ? 'disabled' : ''}>${Roi.busy ? 'Saving…' : r ? 'Save changes' : 'Save ROI'} <kbd>Enter</kbd></button>
        <button id="rpCancel" title="${r ? 'Close without saving' : 'Discard this ROI'} (Esc)">${r ? 'Close' : 'Discard'}</button>
        ${r ? '<span class="spacer"></span><button class="ghost danger" id="rpDelete">Delete</button>' : ''}</div>
      ${r ? `<div class="small muted rform-who">Drawn ${roiDay(r.created_at)}${r.ts !== r.created_at ? `, last changed ${roiDay(r.ts)}` : ''}.</div>` : ''}
    </div>`;
}
function introHtml() {
  const t = ROI_TOOLS[Roi.tool];
  if (t.hint) return `<div class="rintro now"><b>${esc(t.hint)}</b><div class="small muted">Scroll to zoom, hold Space and drag to pan, Esc cancels.</div></div>`;
  return `<ol class="rintro">
      <li>Pan and zoom to the area.</li>
      <li>Press <b>Polygon</b> <kbd>D</kbd>, <b>Square</b> <kbd>S</kbd> or <b>Oval</b> <kbd>O</kbd> on the slide, then draw around it.</li>
      <li>Choose what it shows here, and save.</li></ol>
      <div class="small muted rworkspace">Your workspace: <b class="txt">${esc(Roi.owner)}</b>. Other reviewers don't see these ROIs here, and you don't see theirs.</div>`;
}
function roiRender() {
  const p = $('#rpPanel');
  if (!S.sid) { p.className = 'spanel empty'; p.textContent = 'Select a slide.'; return; }
  if (!Roi.owner) { p.className = 'spanel empty'; p.textContent = 'Enter your initials at the top to open your own workspace. Only the ROIs under your initials are shown here.'; return; }
  p.className = 'spanel';
  const list = roisHere(), top = p.scrollTop;
  const undo = Roi.deleted && Roi.deleted.sid === S.sid ? `<div class="rundo">ROI ${Roi.deleted.n} deleted <button class="mini" id="rpUndo">Undo</button></div>` : '';
  const drawing = Roi.tool !== 'move';
  p.innerHTML = (drawing ? introHtml() : '') + (Roi.form ? formHtml() : drawing ? '' : introHtml()) + undo +
    section('roilist', 'Your ROIs on this slide', `<span class="small muted">${list.length}</span>`, list.length ? list.map((r) => `
      <button type="button" class="rrow ${r.id === Roi.sel ? 'on' : ''}" data-roi="${esc(r.id)}">
        <img src="${roiThumb(r.id)}" alt="" loading="lazy">
        <span class="rrow-body"><b>ROI ${r.n}</b> <span class="muted small">${esc(sizeText(r))}</span>
          <span class="rchips">${labelChips(r.labels)}</span>
          ${r.comment ? `<span class="small rrow-note">${esc(r.comment)}</span>` : ''}
          <span class="small muted">${roiDay(r.created_at)}</span></span>
      </button>`).join('') : '<div class="small muted">None yet.</div>');
  p.scrollTop = top;

  p.querySelectorAll('[data-fold]').forEach((b) => b.addEventListener('click', () => { Panels.toggleSec(b.dataset.fold); roiRender(); }));
  p.querySelectorAll('[data-label]').forEach((b) => b.addEventListener('click', () => { roiToggle(b.dataset.label); b.blur(); }));
  p.querySelectorAll('[data-roi]').forEach((b) => b.addEventListener('click', () => roiSelect(b.dataset.roi, true)));
  const bind = (sel, ev, fn) => { const el = p.querySelector(sel); if (el) el.addEventListener(ev, fn); };
  bind('#rpComment', 'input', (e) => { Roi.form.comment = e.target.value; });
  bind('#rpComment', 'keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); e.stopPropagation(); roiSave(); } });
  bind('#rpSave', 'click', roiSave);
  bind('#rpCancel', 'click', () => { roiClose(); roiRender(); edDraw(); });
  bind('#rpDelete', 'click', roiDelete);
  bind('#rpZoom', 'click', () => roiZoom(roiById(Roi.sel)));
  bind('#rpUndo', 'click', roiRestore);
}

/* ------------------------------------------------------------ on the slide, called from paint() in editor.js */
function roiTag(text, x, y, color) {
  edCtx.font = '600 11px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif';
  const w = edCtx.measureText(text).width + 8;
  const ty = Math.max(y - 16, 0);
  edCtx.fillStyle = 'rgba(11,12,14,.82)'; edCtx.fillRect(x, ty, w, 16);
  edCtx.fillStyle = color; edCtx.textBaseline = 'middle'; edCtx.fillText(text, x + 4, ty + 8.5);
}
function roiPaint(m, W, H) {
  if (!modeIs('rois') || !slideShown()) return;   // the outlines of the new slide must not appear on the old image
  const colorOf = (keys) => (keys.length && labelOf(keys[0]) ? labelOf(keys[0]).color : '#ffffff');
  for (const r of roisHere()) {
    const b = r.bbox, x0 = m.x([b.x, 0]), y0 = m.y([0, b.y]), x1 = m.x([b.x + b.w, 0]), y1 = m.y([0, b.y + b.h]);
    if (x1 < 0 || x0 > W || y1 < 0 || y0 > H) continue;
    const on = r.id === Roi.sel, hot = r.id === Roi.hover;
    const keys = on && Roi.form ? inOrder(Roi.form.labels) : r.labels, color = colorOf(keys);
    trace(r.ring, m, true);
    if (on) { edCtx.globalAlpha = .14; edCtx.fillStyle = color; edCtx.fill(); edCtx.globalAlpha = 1; }
    casedStroke(on ? '#ffffff' : color, on ? 3 : hot ? 2.5 : 1.75);
    if (on || hot || x1 - x0 > 28) {
      const extra = keys.length > 1 ? ` +${keys.length - 1}` : '';
      roiTag(on || hot || x1 - x0 > 110 ? `${r.n} ${keys.length ? labelText(keys[0]) : ''}${extra}` : String(r.n), x0, y0, color);
    }
  }
  const open = Roi.drag && Roi.drag.shape === 'polygon';
  const pending = open ? (Roi.cursor ? [...Roi.drag.pts, Roi.cursor] : Roi.drag.pts) : Roi.drag ? shapeRing(Roi.drag) : Roi.draft && Roi.draft.ring;
  if (pending && pending.length > 1) {
    const keys = Roi.form ? inOrder(Roi.form.labels) : [], color = colorOf(keys);
    trace(pending, m, !open);
    if (!open || pending.length > 2) { edCtx.globalAlpha = .18; edCtx.fillStyle = color; edCtx.fill(); edCtx.globalAlpha = 1; }
    casedStroke(color, 2.5);
    if (Roi.draft) roiTag('New ROI', Math.min(...pending.map((p) => m.x(p))), Math.min(...pending.map((p) => m.y(p))), color);
  }
  if (open) {   // the points placed so far; the first is larger, as a click on it closes the shape
    Roi.drag.pts.forEach((p, i) => {
      edCtx.beginPath(); edCtx.arc(m.x(p), m.y(p), i ? 3.5 : 5.5, 0, 2 * Math.PI);
      edCtx.fillStyle = '#ffffff'; edCtx.fill(); edCtx.lineWidth = 1.5; edCtx.strokeStyle = '#000'; edCtx.stroke();
    });
  }
}

/* ------------------------------------------------------------ drawing. Captured on the way down, as in editor.js. */
const roiDrawing = () => modeIs('rois') && Roi.tool !== 'move' && !Roi.space;
const toSlide = (p) => [Math.max(0, Math.min(S.info.width, p[0])), Math.max(0, Math.min(S.info.height, p[1]))];
function shapeRing(d) {   // the outline of what has been drawn so far
  if (d.shape === 'polygon') return d.pts;
  const [a, b] = d.pts;
  let dx = b[0] - a[0], dy = b[1] - a[1];
  if ((d.shape === 'square') !== !!d.shift) {   // a square unless Shift is held, an oval is free unless Shift is held
    const sx = dx < 0 ? -1 : 1, sy = dy < 0 ? -1 : 1;
    const side = Math.min(Math.max(Math.abs(dx), Math.abs(dy)), sx > 0 ? S.info.width - a[0] : a[0], sy > 0 ? S.info.height - a[1] : a[1]);
    dx = sx * side; dy = sy * side;
  }
  const x0 = Math.min(a[0], a[0] + dx), y0 = Math.min(a[1], a[1] + dy), w = Math.abs(dx), h = Math.abs(dy);
  if (d.shape === 'square') return [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]];
  return Array.from({ length: OVAL_POINTS }, (_, i) => {
    const t = (2 * Math.PI * i) / OVAL_POINTS;
    return [x0 + w / 2 + (w / 2) * Math.cos(t), y0 + h / 2 + (h / 2) * Math.sin(t)];
  });
}
function polygonClick(p) {
  const now = performance.now(), d = Roi.drag;
  if (!d) { Roi.drag = { shape: 'polygon', pts: [p], at: now }; Roi.cursor = p; return; }
  const m = screenMap(), near = (q) => Math.hypot(m.x(p) - m.x(q), m.y(p) - m.y(q)) < HANDLE_PX;
  const again = near(d.pts[d.pts.length - 1]) && now - d.at < 400;   // the second click of a double-click
  d.at = now;
  if (d.pts.length >= 3 && (again || near(d.pts[0]))) { finishDrag(); return; }
  if (!again) d.pts.push(p);
}
viewerEl.addEventListener('pointerdown', (e) => {
  if (!modeIs('rois') || e.button !== 0 || !slideItem() || !S.info) return;
  if (!roiDrawing()) { Roi.press = Roi.tool === 'move' ? { x: e.clientX, y: e.clientY } : null; return; }
  e.stopPropagation(); e.preventDefault();
  if (!slideShown()) { toast('The slide is still opening', true); return; }
  if (!roiWho()) return;
  const p = toSlide(imgPoint(e));
  if (Roi.tool === 'polygon') polygonClick(p);
  else {
    viewerEl.setPointerCapture(e.pointerId);
    Roi.drag = { shape: Roi.tool, pts: [p, p], shift: e.shiftKey };
  }
  edDraw();
}, true);
viewerEl.addEventListener('pointermove', (e) => {
  if (!modeIs('rois') || !slideItem() || !S.info) return;
  const p = imgPoint(e);
  if (Roi.drag) {
    if (!Roi.space) e.stopPropagation();
    if (Roi.drag.shape === 'polygon') Roi.cursor = toSlide(p);
    else { Roi.drag.pts[1] = toSlide(p); Roi.drag.shift = e.shiftKey; }
    edDraw();
    return;
  }
  if (roiDrawing()) { e.stopPropagation(); return; }
  if (Roi.tool !== 'move' || e.buttons || !slideShown()) return;
  const r = roiAt(p), id = r ? r.id : null;
  if (id !== Roi.hover) {
    Roi.hover = id;
    if (id) $('#viewerWrap').dataset.hot = '1'; else delete $('#viewerWrap').dataset.hot;
    edDraw();
  }
}, true);
viewerEl.addEventListener('pointerup', (e) => {
  if (!modeIs('rois') || !slideItem()) return;
  if (Roi.drag) {
    if (!Roi.space) e.stopPropagation();
    if (Roi.drag.shape !== 'polygon') { Roi.drag.shift = e.shiftKey; finishDrag(); }   // a polygon goes on until it is closed
    return;
  }
  if (Roi.press && slideShown() && Math.hypot(e.clientX - Roi.press.x, e.clientY - Roi.press.y) < 4) {
    const r = roiAt(imgPoint(e));
    if (r) roiSelect(r.id, false);
    else if (Roi.sel && !formDirty()) { roiClose(); roiRender(); edDraw(); }
  }
  Roi.press = null;
}, true);
viewerEl.addEventListener('pointercancel', () => { if (Roi.drag && Roi.drag.shape !== 'polygon') Roi.drag = null; edDraw(); }, true);
viewerEl.addEventListener('pointerleave', () => { if (Roi.hover) { Roi.hover = null; delete $('#viewerWrap').dataset.hot; edDraw(); } });

function finishDrag() {
  const d = Roi.drag, ring = shapeRing(d).map(roundPoint);
  const xs = ring.map((p) => p[0]), ys = ring.map((p) => p[1]);
  const w = Math.max(...xs) - Math.min(...xs), h = Math.max(...ys) - Math.min(...ys), s = screenMap().s;
  if (w < MIN_SIDE || h < MIN_SIDE || w * s < MIN_DRAG_PX || h * s < MIN_DRAG_PX) {
    if (d.shape !== 'polygon') Roi.drag = null;   // a polygon stays, so its points can be added to or removed
    edDraw();
    toast('Too small for a ROI. Draw across the area, or zoom out first.', true);
    return;
  }
  Roi.drag = null; Roi.cursor = null;
  roiFlush();
  Roi.draft = { shape: d.shape, ring };
  Roi.form = { labels: new Set(), comment: '' };
  Roi.deleted = null;
  setRoiTool('move');
  Panels.showReview();
  roiRender();
  $('#rpPanel').scrollTop = 0;
  edDraw();
}

/* ------------------------------------------------------------ tools on the viewer */
function setRoiTool(t) {
  if (t !== 'move' && !roiWho()) return;
  if (t !== 'move' && Roi.draft && !Roi.form.labels.size) roiClose();   // starting over: a shape without a label is dropped
  if (t !== Roi.tool) dropPartial();
  Roi.tool = t;
  $('#viewerWrap').dataset.tool = t === 'move' || !modeIs('rois') ? 'nav' : 'roi';
  $('#roiToolsBody').querySelectorAll('[data-roitool]').forEach((b) => {
    b.classList.toggle('on', b.dataset.roitool === t);
    b.setAttribute('aria-pressed', b.dataset.roitool === t);
  });
  if (modeIs('rois')) roiRender();
  edDraw();
}
$('#roiToolsBody').innerHTML = Object.entries(ROI_TOOLS).map(([id, t]) => `
    <button data-roitool="${id}" title="${t.label} (${t.key})"><svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">${t.icon}</svg><span>${t.label}</span></button>`).join('');
$('#roiToolsBody').addEventListener('click', (e) => {
  const b = e.target.closest('[data-roitool]');
  if (b) { setRoiTool(b.dataset.roitool); b.blur(); }
});
document.addEventListener('keydown', (e) => {
  if (e.key === ' ' && modeIs('rois') && Roi.tool !== 'move' && !/INPUT|TEXTAREA|SELECT|BUTTON/.test(document.activeElement.tagName)) {
    Roi.space = true; e.preventDefault(); $('#viewerWrap').dataset.pan = '1';
  }
});
document.addEventListener('keyup', (e) => { if (e.key === ' ' && Roi.space) { Roi.space = false; delete $('#viewerWrap').dataset.pan; } });

/* ------------------------------------------------------------ keys, called from the handler in app.js. True = key used. */
const ROI_SHARED_KEYS = new Set(['ArrowDown', 'ArrowUp', 'f', 'h']);   // these work here as they do in Segments; the rest are swallowed
function roiKey(e, k) {
  if (k === ',' || k === '.' || k === '?' || k === ' ') return false;   // Space pans: its own listener above
  if (modeIs('all')) return true;
  const polygon = Roi.drag && Roi.drag.shape === 'polygon';
  if (k === 'Enter') {
    if (polygon) { if (Roi.drag.pts.length >= 3) finishDrag(); else toast('A polygon needs at least 3 points', true); }
    else if (Roi.form) roiSave();
    return true;
  }
  if (k === 'Backspace' || k === 'Delete') {
    if (polygon) { Roi.drag.pts.pop(); if (!Roi.drag.pts.length) { Roi.drag = null; Roi.cursor = null; } edDraw(); }
    return true;
  }
  if (Roi.form) {
    const i = ROI_KEYS.indexOf(k);
    if (i >= 0 && Roi.labels[i]) { roiToggle(Roi.labels[i].key); return true; }
  }
  const t = Object.keys(ROI_TOOLS).find((id) => ROI_TOOLS[id].key.toLowerCase() === k);
  if (t) { setRoiTool(t); return true; }
  return !ROI_SHARED_KEYS.has(k);
}
function roiEscape() {   // one step back: the shape being drawn, then the tool, then whatever is open in the panel
  if (Roi.drag) { Roi.drag = null; Roi.cursor = null; edDraw(); return; }
  if (Roi.tool !== 'move') { setRoiTool('move'); return; }
  if (Roi.form) { roiClose(); roiRender(); edDraw(); }
}

/* ------------------------------------------------------------ all ROIs */
function roiGallery() {
  const g = $('#gallery'), f = Roi.filter, all = Roi.rois;
  if (!Roi.owner) { g.innerHTML = '<div class="gal-empty"><p>Enter your initials at the top to open your own workspace.</p></div>'; return; }
  if (!Roi.loaded) { g.innerHTML = '<div class="gal-empty muted">Loading…</div>'; return; }
  const shown = all.filter((r) => !f.label || r.labels.includes(f.label));
  const count = (k) => all.filter((r) => r.labels.includes(k)).length;
  const slides = new Set(all.map((r) => r.slide_id)).size;
  const order = new Map(S.slides.map((s, i) => [s.slide_id, i]));
  const bySlide = new Map();
  [...shown].sort((a, b) => (order.get(a.slide_id) ?? 1e9) - (order.get(b.slide_id) ?? 1e9) || a.n - b.n)
    .forEach((r) => { if (!bySlide.has(r.slide_id)) bySlide.set(r.slide_id, []); bySlide.get(r.slide_id).push(r); });
  const top = g.scrollTop;
  g.innerHTML = `<div class="gal-head">
      <h2>My ROIs</h2><span class="muted"><b class="txt">${esc(Roi.owner)}</b> · ${all.length} ${all.length === 1 ? 'ROI' : 'ROIs'} on ${slides} of ${S.slides.length} slides</span>
      <span class="spacer"></span>
      <a class="ghost-link" href="${esc(mine('csv'))}" download>Export CSV</a>
      <a class="ghost-link" href="${esc(mine('geojson'))}" download>Export GeoJSON</a>
    </div>
    <div class="gal-filters" role="group" aria-label="Show only one label">
      <button class="gal-f ${f.label ? '' : 'on'}" data-gf="">All <span class="muted">${all.length}</span></button>
      ${Roi.labels.map((l) => `<button class="gal-f ${f.label === l.key ? 'on' : ''}" data-gf="${esc(l.key)}" ${count(l.key) ? '' : 'disabled'}><i style="background:${esc(l.color)}"></i>${esc(l.short || l.name)} <span class="muted">${count(l.key)}</span></button>`).join('')}
    </div>
    ${all.length ? [...bySlide.entries()].map(([sid, rs]) => `
      <section class="gal-slide"><h3><button class="linkish" data-gs="${esc(sid)}" title="Open this slide">${esc(rs[0].patient)}</button> <span class="muted">${esc(rs[0].cohort)} · ${esc(sid.replace(/^(MSK|LUND|Utah)_/, ''))}</span></h3>
        <div class="gal-grid">${rs.map((r) => `
          <button type="button" class="gcard" data-roi="${esc(r.id)}" title="Open on the slide">
            <span class="gthumb"><img src="${roiThumb(r.id)}" alt="ROI ${r.n}" loading="lazy"></span>
            <span class="gcard-body"><span class="rchips">${labelChips(r.labels)}</span>
              ${r.comment ? `<span class="small rrow-note">${esc(r.comment)}</span>` : ''}
              <span class="small muted">ROI ${r.n}${sizeText(r) ? ` · ${esc(sizeText(r))}` : ''}</span>
              <span class="small muted">${roiDay(r.created_at)}</span></span>
          </button>`).join('')}</div></section>`).join('') : `<div class="gal-empty"><p>You have no ROIs yet.</p><button class="primary" id="galStart">Pick the first one</button></div>`}`;
  g.scrollTop = top;
  g.querySelectorAll('[data-gf]').forEach((b) => b.addEventListener('click', () => { f.label = b.dataset.gf || null; roiGallery(); }));
  g.querySelectorAll('[data-gs]').forEach((b) => b.addEventListener('click', () => { setMode('rois'); openSlide(b.dataset.gs); }));
  g.querySelectorAll('[data-roi]').forEach((b) => b.addEventListener('click', async () => {
    const r = roiById(b.dataset.roi);
    setMode('rois');
    await openSlide(r.slide_id);
    roiSelect(r.id, true);
  }));
  const st = g.querySelector('#galStart');
  if (st) st.addEventListener('click', () => setMode('rois'));
}

/* ------------------------------------------------------------ modes */
const HASH = { rois: '#rois', all: '#my-rois', patches: '#patches' };
function swapView(toRoi) {   // ROIs are picked on the plain stain: each mode keeps its own overlay and lines
  const from = toRoi ? Roi.segView : Roi.roiView, to = toRoi ? Roi.roiView : Roi.segView;
  from.heat = S.heat; from.lines = Ed.showLines;
  Ed.showLines = to.lines; $('#linesChk').checked = to.lines;
  if (slideItem()) setOverlay(to.heat);
  else { S.heat = to.heat; $('#heatSel').value = to.heat; }
}
function paintTabs() {
  document.querySelectorAll('#modeTabs [data-mode]').forEach((b) => {
    const on = modeIs(b.dataset.mode);
    b.classList.toggle('on', on);
    b.setAttribute('aria-selected', on);
  });
  const filter = $('#slideFilter');
  filter.querySelector('[value="todo"]').textContent = patchMode() ? 'No patches yet' : roiMode() ? 'No ROIs yet' : 'Not reviewed';
  filter.querySelector('[value="done"]').textContent = patchMode() ? 'Has patches' : roiMode() ? 'Has ROIs' : 'Reviewed';
}
function setMode(m) {
  const was = document.body.dataset.mode;
  if (m === was) return;
  if (was === 'rois') { roiFlush(); setRoiTool('move'); }
  if (was === 'segments') setTool('nav');
  if (was === 'patches') { patchFlush('left the tab'); setPaTool('move'); paLog('mode', { on: false }); }
  document.body.dataset.mode = m;
  try { localStorage.setItem('hss_mode', m); } catch (e) { /* ignore */ }
  history.replaceState(null, '', HASH[m] || location.pathname + location.search);
  if ((was === 'segments') !== (m === 'segments')) swapView(m !== 'segments');
  $('#viewerWrap').dataset.tool = 'nav';
  paintTabs();
  renderSlideList(); renderProgress();
  Panels.refresh();
  if (m === 'all') Panels.showReview();   // in a narrow window the gallery takes the review panel's place
  if (m === 'rois') roiRender();    // at once, from what is known; the load below brings it up to date
  if (m === 'all') roiGallery();
  if (m === 'patches') { paLog('mode', { on: true }); paRender(); patchLoad(); }
  else if (m !== 'segments') roiLoad();
  edDraw();
}
document.querySelectorAll('#modeTabs [data-mode]').forEach((b) => b.addEventListener('click', () => { b.blur(); setMode(b.dataset.mode); }));

/* The first mode: a link can name one (…/#rois), otherwise the one this browser used last. */
(function startMode() {
  let m = location.hash === HASH.rois ? 'rois' : location.hash === HASH.all ? 'all' : location.hash === HASH.patches ? 'patches' : null;
  try { m = m || localStorage.getItem('hss_mode'); } catch (e) { /* ignore */ }
  if (m !== 'rois' && m !== 'all' && m !== 'patches') m = 'segments';
  document.body.dataset.mode = m;
  if (m !== 'segments') {
    S.heat = Roi.roiView.heat; Ed.showLines = Roi.roiView.lines;
    $('#linesChk').checked = false; $('#heatSel').value = S.heat;
    history.replaceState(null, '', HASH[m]);
  }
  paintTabs();
  setRoiTool('move');
  Panels.refresh();
})();
