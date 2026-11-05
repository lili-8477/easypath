/* Segmentation toolbox: a canvas over the viewer that draws the segment outlines and the draft of the current tool.
   Every finished edit is sent straight to the server's edit log, which rebuilds the segmentation. Loaded after app.js. */
'use strict';

const TOOLS = {
  nav: { label: 'Move', key: 'V', hint: 'Drag to pan, scroll to zoom. Click a region to select its segment.',
    icon: '<path d="M12 3l3 3h-2v5h5V9l3 3-3 3v-2h-5v5h2l-3 3-3-3h2v-5H6v2l-3-3 3-3v2h5V6H9z"/>' },
  lines: { label: 'Lines', key: 'E', hint: 'Click a region to edit its outline. Drag a point or a line, Alt-click a point to delete it. Enter applies, Esc cancels.',
    icon: '<path d="M4 18L9 7l6 8 5-9" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="4" cy="18" r="2.2"/><circle cx="9" cy="7" r="2.2"/><circle cx="15" cy="15" r="2.2"/><circle cx="20" cy="6" r="2.2"/>' },
  polygon: { label: 'Polygon', key: 'D', hint: 'Click to place points. Double-click or Enter closes the shape, Backspace removes a point, Esc cancels.',
    icon: '<path d="M6 5l13 3-3 11-11-3z" fill="none" stroke="currentColor" stroke-width="2"/>' },
  lasso: { label: 'Lasso', key: 'L', hint: 'Drag around a region. Releasing closes the shape.',
    icon: '<ellipse cx="12" cy="9.5" rx="8" ry="5.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="M8 14c-2 2-1 5 2 5" fill="none" stroke="currentColor" stroke-width="2"/>' },
  brush: { label: 'Brush', key: 'R', hint: 'Drag to paint. [ and ] change the size.',
    icon: '<path d="M19 3l2 2-9 10-3-3zM8 13l3 3c-1 3-4 5-7 4 2-1 2-2 2-3 0-2 1-3 2-4z"/>' },
  eraser: { label: 'Erase', key: 'X', hint: 'Drag to remove tissue from the segmentation. [ and ] change the size.',
    icon: '<path d="M14 4l6 6-9 10H7l-4-4zM9 9l6 6" fill="none" stroke="currentColor" stroke-width="2"/>' },
};
const NEW_SEGMENT = 'new';
const HANDLE_PX = 8;   // how close to a point or line a press has to be, in screen pixels

const Ed = {
  tool: 'nav', target: NEW_SEGMENT, brushUm: 100, showLines: true, rings: [], space: false, busy: false,
  draft: null,      // polygon or lasso points, or { points } of a brush stroke, in slide pixels
  cursor: null,     // last pointer position in slide pixels
  hover: null,      // ring under the pointer
  sel: null,        // lines tool: { segment, old, ring, dirty, grab }
  press: null,      // nav tool: where a possible click began
};

const edCanvas = $('#edCanvas');
const edCtx = edCanvas.getContext('2d');

/* ------------------------------------------------------------ coordinates */
function imgPoint(e) {
  const r = $('#viewer').getBoundingClientRect();
  const p = slideItem().viewerElementToImageCoordinates(new OpenSeadragon.Point(e.clientX - r.left, e.clientY - r.top));
  return [p.x, p.y];
}
function screenMap() {   // slide pixels to canvas pixels; the viewer never rotates, so offset and scale are enough
  const o = slideItem().imageToViewerElementCoordinates(new OpenSeadragon.Point(0, 0));
  const u = slideItem().imageToViewerElementCoordinates(new OpenSeadragon.Point(1000, 0));
  const s = (u.x - o.x) / 1000;
  return { s, x: (p) => o.x + p[0] * s, y: (p) => o.y + p[1] * s };
}
const canEdit = () => !!(SR.data && slideItem());
const segRow = (k) => SR.data && SR.data.segments.find((s) => s.segment === k);
const segColor = (k) => (segRow(k) ? segRow(k).color : '#ffffff');

function insideRing(p, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function ringAt(p) {   // smallest piece under the point, so a piece enclosed by another can be picked
  let best = null;
  for (const r of Ed.rings) {
    const b = r.box;
    if (p[0] < b[0] || p[0] > b[2] || p[1] < b[1] || p[1] > b[3] || !insideRing(p, r.ring)) continue;
    if (!best || r.size < best.size) best = r;
  }
  return best;
}
function segDistance(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}

/* ------------------------------------------------------------ drawing */
let drawQueued = false;
function edDraw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; paint(); });
}
function trace(pts, m, close) {
  edCtx.beginPath();
  pts.forEach((p, i) => (i ? edCtx.lineTo(m.x(p), m.y(p)) : edCtx.moveTo(m.x(p), m.y(p))));
  if (close) edCtx.closePath();
}
function casedStroke(color, width) {   // a dark casing keeps a line visible on any stain or overlay colour
  edCtx.lineJoin = 'round'; edCtx.lineCap = 'round';
  edCtx.strokeStyle = 'rgba(0,0,0,.75)'; edCtx.lineWidth = width + 2; edCtx.stroke();
  edCtx.strokeStyle = color; edCtx.lineWidth = width; edCtx.stroke();
}
function paint() {
  const dpr = window.devicePixelRatio || 1, W = edCanvas.clientWidth, H = edCanvas.clientHeight;
  if (edCanvas.width !== Math.round(W * dpr) || edCanvas.height !== Math.round(H * dpr)) {
    edCanvas.width = Math.round(W * dpr); edCanvas.height = Math.round(H * dpr);
  }
  edCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  edCtx.clearRect(0, 0, W, H);
  if (!slideItem() || viewer.world.getItemCount() === 0) return;
  const m = screenMap();
  if (Ed.showLines || Ed.tool === 'lines') {
    for (const r of Ed.rings) {
      const b = r.box;
      if (m.x([b[2], 0]) < 0 || m.x([b[0], 0]) > W || m.y([0, b[3]]) < 0 || m.y([0, b[1]]) > H) continue;
      if (Ed.sel && Ed.sel.source === r) continue;
      const on = r.segment === SR.focus, hot = r === Ed.hover;
      trace(r.ring, m, true);
      casedStroke(on ? '#ffffff' : segColor(r.segment), on || hot ? 2.5 : 1.25);
    }
  }
  if (Ed.sel) {
    trace(Ed.sel.ring, m, true);
    casedStroke('#ffffff', 2.5);
    Ed.sel.ring.forEach((p, i) => {
      edCtx.beginPath(); edCtx.arc(m.x(p), m.y(p), i === Ed.sel.grab ? 6 : 4, 0, 2 * Math.PI);
      edCtx.fillStyle = i === Ed.sel.grab ? '#7ab8ff' : '#ffffff'; edCtx.fill();
      edCtx.lineWidth = 1.5; edCtx.strokeStyle = '#000'; edCtx.stroke();
    });
  }
  const color = Ed.tool === 'eraser' ? '#ffffff' : Ed.target === NEW_SEGMENT ? '#ffffff' : segColor(Ed.target);
  if (Ed.draft && (Ed.tool === 'polygon' || Ed.tool === 'lasso')) {
    const pts = Ed.tool === 'polygon' && Ed.cursor ? [...Ed.draft, Ed.cursor] : Ed.draft;
    trace(pts, m, Ed.tool === 'lasso');
    edCtx.globalAlpha = .25; edCtx.fillStyle = color; edCtx.fill(); edCtx.globalAlpha = 1;
    casedStroke(color, 2);
    if (Ed.tool === 'polygon') {
      Ed.draft.forEach((p) => { edCtx.beginPath(); edCtx.arc(m.x(p), m.y(p), 3.5, 0, 2 * Math.PI); edCtx.fillStyle = color; edCtx.fill(); });
    }
  }
  if (Ed.tool === 'brush' || Ed.tool === 'eraser') {
    const w = brushPx() * m.s;
    if (Ed.draft && Ed.draft.length) {
      trace(Ed.draft, m, false);
      edCtx.lineJoin = 'round'; edCtx.lineCap = 'round'; edCtx.lineWidth = Math.max(w, 1);
      edCtx.globalAlpha = .55; edCtx.strokeStyle = Ed.tool === 'eraser' ? '#0b0c0e' : color; edCtx.stroke(); edCtx.globalAlpha = 1;
    }
    if (Ed.cursor) {
      edCtx.beginPath(); edCtx.arc(m.x(Ed.cursor), m.y(Ed.cursor), Math.max(w / 2, 2), 0, 2 * Math.PI);
      edCtx.lineWidth = 1; edCtx.strokeStyle = '#000'; edCtx.stroke();
      edCtx.beginPath(); edCtx.arc(m.x(Ed.cursor), m.y(Ed.cursor), Math.max(w / 2, 2) + 1, 0, 2 * Math.PI);
      edCtx.strokeStyle = '#fff'; edCtx.stroke();
    }
  }
}
viewer.addHandler('update-viewport', edDraw);
viewer.addHandler('animation', edDraw);
viewer.addHandler('resize', edDraw);

/* ------------------------------------------------------------ data */
async function edLoadOutlines() {
  const sid = S.sid;
  let rows = [];
  try { rows = await api(`api/slides/${encodeURIComponent(sid)}/outlines`); } catch (e) { /* no segmentation: nothing to draw */ }
  if (S.sid !== sid) return;
  Ed.rings = rows.map((r) => {
    const xs = r.ring.map((p) => p[0]), ys = r.ring.map((p) => p[1]);
    const box = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    return { ...r, box, size: (box[2] - box[0]) * (box[3] - box[1]) };
  });
  Ed.hover = null;
  edDraw();
}
function edOnSlide() {   // a different slide is opening: drop anything half drawn
  Ed.rings = []; Ed.draft = null; Ed.sel = null; Ed.hover = null;
  if (Ed.target !== NEW_SEGMENT) Ed.target = NEW_SEGMENT;
  edDraw(); edRenderTools();
}
function edSync() {   // the segment list changed
  if (Ed.target !== NEW_SEGMENT && !segRow(Ed.target)) Ed.target = NEW_SEGMENT;
  edRenderTools();
}

/* ------------------------------------------------------------ sending edits */
async function sendEdit(action, segment, geometry) {
  const who = reviewer();
  if (!who || Ed.busy) return null;
  Ed.busy = true; edRenderTools();
  let res = null;
  try {
    res = await post(`api/slides/${encodeURIComponent(S.sid)}/edits`, { reviewer: who, action, segment, geometry });
    await loadSegments(S.sid, true);
    await edLoadOutlines();
    setHeat();
  } catch (e) {
    toast('Edit not saved: ' + errDetail(e), true);
  }
  Ed.busy = false; edRenderTools();
  return res;
}
const targetSegment = () => (Ed.target === NEW_SEGMENT ? null : Ed.target);
async function finishPaint(action, geometry) {
  const res = await sendEdit(action, targetSegment(), geometry);
  if (res) {
    Ed.draft = null;
    if (Ed.target === NEW_SEGMENT) {
      Ed.target = res.segment;
      SR.focus = res.segment;
      SR.form.checked.delete(res.segment);
      renderSR();
      toast(`New segment ${cname(res.segment)} added. Choose its label in the panel.`);
    }
  }
  edRenderTools(); edDraw();
  return !!res;
}
function closePolygon() {
  if (!Ed.draft || Ed.draft.length < 3) { toast('A shape needs at least 3 points', true); return; }
  finishPaint('paint', { polygons: [Ed.draft.map(roundPoint)] });
}
const roundPoint = (p) => [Math.round(p[0]), Math.round(p[1])];
const brushPx = () => Ed.brushUm / ((S.info && S.info.mpp) || 0.5);

async function applyLines() {
  const s = Ed.sel;
  if (!s) return;
  if (!s.dirty) { Ed.sel = null; edDraw(); edRenderTools(); return; }
  const row = segRow(s.segment);
  const res = await sendEdit('reshape', s.segment, { old: s.old, new: s.ring.map(roundPoint), classes: row ? row.classes : [s.segment] });
  if (res) { Ed.sel = null; toast(`${cname(s.segment)} outline updated`); }
  edRenderTools(); edDraw();
}
async function edUndoRedo(action) {
  if (!SR.data || !SR.data.edits[action === 'undo' ? 'can_undo' : 'can_redo']) return;
  if (await sendEdit(action)) toast(action === 'undo' ? 'Edit undone' : 'Edit restored');
}

/* ------------------------------------------------------------ pointer input.
   Captured on the way down, so while a drawing tool is active OpenSeadragon does not pan. Scroll zoom still reaches it. */
const viewerEl = $('#viewer');
const onNavigator = (e) => e.target.closest && e.target.closest('.navigator');
const drawing = () => Ed.tool !== 'nav' && !Ed.space;

viewerEl.addEventListener('pointerdown', (e) => {
  if (onNavigator(e) || !slideItem() || e.button !== 0) return;
  if (!drawing()) { Ed.press = Ed.tool === 'nav' ? { x: e.clientX, y: e.clientY } : null; return; }
  e.stopPropagation(); e.preventDefault();
  if (!canEdit()) { toast('This slide has no segmentation to edit', true); return; }
  if (Ed.busy || !reviewer()) return;
  viewerEl.setPointerCapture(e.pointerId);
  const p = imgPoint(e);
  if (Ed.tool === 'polygon') {
    const m = screenMap(), last = Ed.draft && Ed.draft[Ed.draft.length - 1], first = Ed.draft && Ed.draft[0];
    const near = (q) => q && Math.hypot(m.x(p) - m.x(q), m.y(p) - m.y(q)) < HANDLE_PX;
    const again = near(last) && performance.now() - (Ed.lastClick || 0) < 400;
    Ed.lastClick = performance.now();
    if (Ed.draft && Ed.draft.length >= 3 && (again || near(first))) { closePolygon(); return; }
    if (!again) Ed.draft = [...(Ed.draft || []), p];
  } else if (Ed.tool === 'lasso' || Ed.tool === 'brush' || Ed.tool === 'eraser') {
    Ed.draft = [p];
  } else if (Ed.tool === 'lines') {
    linesPress(p, e);
  }
  edRenderTools(); edDraw();
}, true);

viewerEl.addEventListener('pointermove', (e) => {
  if (!slideItem()) return;
  const p = imgPoint(e);
  Ed.cursor = p;
  if (!drawing()) { if (Ed.tool === 'nav') { Ed.hover = ringAt(p); edDraw(); } return; }
  e.stopPropagation();
  if (Ed.tool === 'lines') {
    if (Ed.sel && Ed.sel.dragging) { Ed.sel.ring[Ed.sel.grab] = p; Ed.sel.dirty = true; }
    else if (Ed.sel) Ed.sel.grab = nearestPoint(p);
    else Ed.hover = ringAt(p);
  } else if ((Ed.tool === 'lasso' || Ed.tool === 'brush' || Ed.tool === 'eraser') && Ed.draft && e.buttons & 1) {
    const m = screenMap(), last = Ed.draft[Ed.draft.length - 1];
    if (Math.hypot(m.x(p) - m.x(last), m.y(p) - m.y(last)) >= 3) Ed.draft.push(p);
  }
  edDraw();
}, true);

viewerEl.addEventListener('pointerup', (e) => {
  if (!slideItem()) return;
  if (!drawing()) {
    if (Ed.tool === 'nav' && Ed.press && Math.hypot(e.clientX - Ed.press.x, e.clientY - Ed.press.y) < 4) {
      const r = ringAt(imgPoint(e));
      if (r) focusSeg(r.segment);
    }
    Ed.press = null;
    return;
  }
  e.stopPropagation();
  if (Ed.tool === 'lasso' && Ed.draft) {
    if (Ed.draft.length >= 3) closeLasso();
    else { Ed.draft = null; edDraw(); }
  } else if ((Ed.tool === 'brush' || Ed.tool === 'eraser') && Ed.draft) {
    finishStroke();
  } else if (Ed.tool === 'lines' && Ed.sel) {
    Ed.sel.dragging = false; edRenderTools(); edDraw();
  }
}, true);
viewerEl.addEventListener('pointerleave', () => { Ed.cursor = null; if (Ed.hover) Ed.hover = null; edDraw(); });
function closeLasso() { finishPaint('paint', { polygons: [Ed.draft.map(roundPoint)] }); }
function finishStroke() {
  const geometry = { points: Ed.draft.map(roundPoint), width: Math.round(brushPx()) };
  if (Ed.tool === 'eraser') {
    sendEdit('stroke', 255, geometry).then((res) => { if (res) Ed.draft = null; edDraw(); });
  } else finishPaint('stroke', geometry);
}

/* ------------------------------------------------------------ lines tool */
function nearestPoint(p) {
  const m = screenMap();
  let best = null, bd = HANDLE_PX;
  Ed.sel.ring.forEach((q, i) => {
    const d = Math.hypot(m.x(p) - m.x(q), m.y(p) - m.y(q));
    if (d < bd) { bd = d; best = i; }
  });
  return best;
}
function linesPress(p, e) {
  if (!Ed.sel) {
    const r = ringAt(p);
    if (r) {
      Ed.sel = { source: r, segment: r.segment, old: r.ring, ring: r.ring.map((q) => [...q]), dirty: false, grab: null, dragging: false };
      SR.focus = r.segment; renderSR();
    }
    return;
  }
  const i = nearestPoint(p);
  if (i != null) {
    if (e.altKey) { deletePoint(i); return; }
    Ed.sel.grab = i; Ed.sel.dragging = true;
    return;
  }
  const m = screenMap(), R = Ed.sel.ring, tol = HANDLE_PX / m.s;
  let bi = -1, bd = tol;
  R.forEach((a, j) => {
    const d = segDistance(p, a, R[(j + 1) % R.length]);
    if (d < bd) { bd = d; bi = j; }
  });
  if (bi >= 0) {   // press on a line: a new point, ready to drag
    R.splice(bi + 1, 0, p);
    Ed.sel.grab = bi + 1; Ed.sel.dragging = true; Ed.sel.dirty = true;
  }
}
function deletePoint(i) {
  if (Ed.sel.ring.length <= 3) { toast('An outline needs at least 3 points', true); return; }
  Ed.sel.ring.splice(i, 1);
  Ed.sel.grab = null; Ed.sel.dirty = true;
  edDraw();
}

/* ------------------------------------------------------------ toolbox */
function setTool(t) {
  if (Ed.tool === t) return;
  Ed.tool = t; Ed.draft = null; Ed.sel = null; Ed.hover = null;
  $('#viewerWrap').dataset.tool = t;
  edRenderTools(); edDraw();
}
function edRenderTools() {
  const box = $('#toolbox'), d = SR.data && SR.data.edits;
  box.querySelectorAll('[data-tool]').forEach((b) => {
    b.classList.toggle('on', b.dataset.tool === Ed.tool);
    b.setAttribute('aria-pressed', b.dataset.tool === Ed.tool);
    b.disabled = b.dataset.tool !== 'nav' && !SR.data;
  });
  $('#edUndo').disabled = !d || !d.can_undo || Ed.busy;
  $('#edRedo').disabled = !d || !d.can_redo || Ed.busy;

  const paints = ['polygon', 'lasso', 'brush'].includes(Ed.tool), sized = ['brush', 'eraser'].includes(Ed.tool);
  const opts = $('#toolOpts');
  opts.classList.toggle('hidden', Ed.tool === 'nav');
  $('#edTargetWrap').classList.toggle('hidden', !paints);
  $('#edSizeWrap').classList.toggle('hidden', !sized);
  $('#edApply').classList.toggle('hidden', !(Ed.tool === 'lines' && Ed.sel) && !(Ed.tool === 'polygon' && Ed.draft && Ed.draft.length));
  $('#edApply').textContent = Ed.tool === 'polygon' ? 'Close shape' : 'Apply';
  $('#edApply').disabled = Ed.busy || (Ed.tool === 'lines' && !(Ed.sel && Ed.sel.dirty));
  $('#edHint').textContent = Ed.busy ? 'Saving…' : TOOLS[Ed.tool].hint;
  if (paints && SR.data) {
    const sel = $('#edTarget');
    sel.innerHTML = `<option value="${NEW_SEGMENT}">New segment</option>` + SR.data.segments.map((s) =>
      `<option value="${s.segment}">${cname(s.segment)} · ${esc(slab(SR.form ? SR.form.labels[s.segment] : s.pred_label))}</option>`).join('');
    sel.value = String(Ed.target);
  }
  $('#edSize').value = Ed.brushUm;
  $('#edSizeVal').textContent = Ed.brushUm >= 1000 ? `${(Ed.brushUm / 1000).toFixed(1)} mm` : `${Ed.brushUm} µm`;
}
$('#toolbox').innerHTML = Object.entries(TOOLS).map(([id, t]) => `
    <button data-tool="${id}" title="${t.label} (${t.key})"><svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">${t.icon}</svg><span>${t.label}</span></button>`).join('') + `
    <div class="tool-sep"></div>
    <button id="edUndo" title="Undo the last edit on this slide (Z)"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 5L4 9l4 4M4 9h10a5 5 0 010 10h-3"/></svg><span>Undo</span></button>
    <button id="edRedo" title="Redo (Shift+Z)"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 5l4 4-4 4M20 9H10a5 5 0 000 10h3"/></svg><span>Redo</span></button>`;
$('#toolbox').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  if (b.dataset.tool) setTool(b.dataset.tool);
  else edUndoRedo(b.id === 'edUndo' ? 'undo' : 'redo');
  b.blur();
});
$('#edTarget').addEventListener('change', (e) => { Ed.target = e.target.value === NEW_SEGMENT ? NEW_SEGMENT : +e.target.value; e.target.blur(); edDraw(); });
$('#edSize').addEventListener('input', (e) => { Ed.brushUm = +e.target.value; edRenderTools(); edDraw(); });
$('#edApply').addEventListener('click', () => { if (Ed.tool === 'polygon') closePolygon(); else applyLines(); });
$('#linesChk').addEventListener('change', (e) => { Ed.showLines = e.target.checked; edDraw(); });
function brushStep(dir) {
  Ed.brushUm = Math.max(+$('#edSize').min, Math.min(+$('#edSize').max, Math.round(Ed.brushUm * (dir > 0 ? 1.25 : 0.8) / 10) * 10));
  edRenderTools(); edDraw();
}

/* ------------------------------------------------------------ keys, called from the handler in app.js. True = key used. */
function edKey(e, k) {
  if (k === 'Enter') {
    if (Ed.tool === 'polygon' && Ed.draft) { closePolygon(); return true; }
    if (Ed.tool === 'lines' && Ed.sel) { applyLines(); return true; }
    return false;
  }
  if (k === 'Backspace' || k === 'Delete') {
    if (Ed.tool === 'polygon' && Ed.draft) {
      Ed.draft.pop();
      if (!Ed.draft.length) Ed.draft = null;
      edRenderTools(); edDraw();
      return true;
    }
    if (Ed.tool === 'lines' && Ed.sel && Ed.sel.grab != null) { deletePoint(Ed.sel.grab); return true; }
    return false;
  }
  if (k === 'z') { edUndoRedo(e.shiftKey ? 'redo' : 'undo'); return true; }
  if (k === '[' || k === ']') { brushStep(k === ']' ? 1 : -1); return true; }
  const t = Object.keys(TOOLS).find((id) => TOOLS[id].key.toLowerCase() === k);
  if (t) { setTool(t); return true; }
  return false;
}
function edEscape() {   // one step back: cancel the draft or selection, then return to Move
  if (Ed.draft || Ed.sel) { Ed.draft = null; Ed.sel = null; edRenderTools(); edDraw(); return true; }
  if (Ed.tool !== 'nav') { setTool('nav'); return true; }
  return false;
}
document.addEventListener('keydown', (e) => {
  if (e.key === ' ' && Ed.tool !== 'nav' && !/INPUT|TEXTAREA|SELECT|BUTTON/.test(document.activeElement.tagName)) { Ed.space = true; e.preventDefault(); $('#viewerWrap').dataset.pan = '1'; }
});
document.addEventListener('keyup', (e) => { if (e.key === ' ') { Ed.space = false; delete $('#viewerWrap').dataset.pan; } });
new ResizeObserver(edDraw).observe(edCanvas);
$('#viewerWrap').dataset.tool = Ed.tool;
edRenderTools();
