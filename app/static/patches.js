/* Annotating patches for model training: a grid of square patches over the slide. Pick a patch, draw a region with the brush,
   polygon or freehand tool, say what it shows, and go on until every pixel of the patch is in a region. Regions may overlap.
   Each reviewer has a workspace of their own. Everything is saved by itself, shortly after each change. Which patches get picked
   is up to the reviewer, so the order they are picked in, the time on each and every operation are logged. Loaded after rois.js. */
'use strict';

const PA_LABEL_KEYS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'b', 'u'];   // one for each label, in the order of the list
const GRID_MIN_PX = 10, ZOOM_TO_PX = 48;   // on screen: the smallest cell worth drawing, and how large a click on the bare slide makes it
const PICK_PAD_PX = 16;                    // around a picked patch, for drawing past its edge
const IDLE_MS = 120000, CLOCK_MS = 1000, MAX_STEP_MS = 5000;   // no input for this long is not active time; a longer gap between clock ticks is a sleeping computer
const TICK_MS = 15000, SEND_MS = 5000, SEND_MAX = 500, LOG_MAX = 20000, VIEW_MS = 2000;
const SAVE_DELAY_MS = 500, RETRY_MS = 8000, UNDO_MAX = 40, LIST_MAX = 200;
const PA_TOOLS = {
  move: { label: 'Move', key: 'V', icon: '<path d="M12 3l3 3h-2v5h5V9l3 3-3 3v-2h-5v5h2l-3 3-3-3h2v-5H6v2l-3-3 3-3v2h5V6H9z"/>' },
  brush: { label: 'Brush', key: 'R', hint: 'Paint over the region, in as many strokes as it takes. [ and ] change the size.', icon: '<path d="M19 3l2 2-9 10-3-3zM8 13l3 3c-1 3-4 5-7 4 2-1 2-2 2-3 0-2 1-3 2-4z"/>' },
  polygon: { label: 'Polygon', key: 'D', hint: 'Click around the region. Double-click, Enter or a click on the first point closes it. Backspace removes a point.', icon: '<path d="M6 5l13 3-3 11-11-3z" fill="none" stroke="currentColor" stroke-width="2"/>' },
  lasso: { label: 'Freehand', key: 'L', hint: 'Drag around the region. Releasing closes the shape.', icon: '<ellipse cx="12" cy="9.5" rx="8" ry="5.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="M8 14c-2 2-1 5 2 5" fill="none" stroke="currentColor" stroke-width="2"/>' },
  eraser: { label: 'Erase', key: 'X', hint: 'Drag to remove. This erases every region under the brush. To erase from one only, click it in the list below first.', icon: '<path d="M14 4l6 6-9 10H7l-4-4zM9 9l6 6" fill="none" stroke="currentColor" stroke-width="2"/>' },
};

const Pa = {
  owner: reviewerInput.value.trim(),   // whose workspace is on screen
  size: 1024,     // the side of a patch on the slide, in level-0 pixels; grid and coordinates
  mask: 512, ds: 2, mag: '20×',   // the side of its masks, how many slide pixels one mask pixel covers, and what that makes it
  labels: [], loaded: false,
  bySlide: new Map(),   // slide id -> Map('x,y' -> summary)
  tool: 'move', lastDraw: 'brush', brush: 64, space: false,
  cur: null,      // the open patch: { sid, x, y, loading, regions, comment, dirty, timer, covered, undo, redo }
  draft: null,    // the region being drawn or changed: { mask, labels: Set, opened }. opened: it came from the list, and strokes add to it
  stroke: null,   // a brush or eraser stroke under way: { last, targets, value }
  drag: null,     // polygon or freehand points in slide pixels, while being drawn
  dragAt: 0, cursor: null, hoverCell: null, press: null,
  alpha: 0.4, hide: false, gen: 0,
  visit: null,    // the stay on the open patch: { id, sid, x, y, start, last, active, ticked }
};
const paN = () => Pa.mask * Pa.mask;
const paCells = (sid) => Pa.bySlide.get(sid) || new Map();
function paCount(sid) {
  let done = 0, part = 0;
  paCells(sid).forEach((c) => { if (c.status === 'complete') done++; else part++; });
  return { done, part, all: done + part };
}
const paLabel = (k) => Pa.labels.find((l) => l.key === k);
const paLabelText = (k) => { const l = paLabel(k); return l ? l.short || l.name : k; };
const paInOrder = (set) => Pa.labels.map((l) => l.key).filter((k) => set.has(k));
const paKeyOf = (k) => PA_LABEL_KEYS[Pa.labels.findIndex((l) => l.key === k)];
const paChips = (keys) => keys.map((k) => `<span class="rchip"><i style="background:${esc((paLabel(k) || {}).color || '#fff')}"></i>${esc(paLabelText(k))}</span>`).join('');
const pctText = (f) => { const p = f * 100; return p >= 100 ? '100%' : p > 99 ? `${p.toFixed(1)}%` : p > 0 && p < 1 ? '<1%' : `${Math.floor(p)}%`; };
const toGoText = (f) => { const p = (1 - f) * 100; return p < 1 ? '<1%' : `${Math.ceil(p)}%`; };   // rounded up: "covered" rounds down, and they should add up
const cellName = (x, y) => `Column ${x / Pa.size + 1}, row ${y / Pa.size + 1}`;

/* ------------------------------------------------------------ masks: one byte per pixel, level-0 pixels of the patch */
function rleEncode(mask) {   // pixels of 0, then of 1, and so on: the server's form
  const runs = [];
  let v = 0, run = 0;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] !== v) { runs.push(run); run = 0; v = mask[i]; }
    run++;
  }
  runs.push(run);
  return runs;
}
function rleDecode(runs) {
  const m = new Uint8Array(paN());
  let i = 0, v = 0;
  for (const r of runs) { if (v) m.fill(1, i, i + r); i += r; v ^= 1; }
  return m;
}
const maskAny = (m) => m.indexOf(1) >= 0;
function fillPolygon(mask, pts) {   // pts in patch pixels; a pixel is in if its centre is, by the winding rule the screen draws with
  const n = Pa.mask, ys = pts.map((p) => p[1]);
  const y0 = Math.max(0, Math.ceil(Math.min(...ys) - 0.5)), y1 = Math.min(n - 1, Math.floor(Math.max(...ys) - 0.5));
  for (let y = y0; y <= y1; y++) {
    const yc = y + 0.5, hits = [];
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [ax, ay] = pts[j], [bx, by] = pts[i];
      if ((ay <= yc) !== (by <= yc)) hits.push([ax + ((yc - ay) / (by - ay)) * (bx - ax), by > ay ? 1 : -1]);
    }
    hits.sort((a, b) => a[0] - b[0]);
    let w = 0;
    for (let i = 0; i < hits.length - 1; i++) {
      w += hits[i][1];
      if (w !== 0) {
        const a = Math.max(0, Math.ceil(hits[i][0] - 0.5)), b = Math.min(n, Math.ceil(hits[i + 1][0] - 0.5));
        if (b > a) mask.fill(1, y * n + a, y * n + b);
      }
    }
  }
}
function stamp(mask, cx, cy, r, value) {
  const n = Pa.mask;
  const y0 = Math.max(0, Math.ceil(cy - r - 0.5)), y1 = Math.min(n - 1, Math.floor(cy + r - 0.5));
  for (let y = y0; y <= y1; y++) {
    const dy = y + 0.5 - cy, half = Math.sqrt(Math.max(0, r * r - dy * dy));
    const a = Math.max(0, Math.ceil(cx - half - 0.5)), b = Math.min(n, Math.floor(cx + half - 0.5) + 1);
    if (b > a) mask.fill(value, y * n + a, y * n + b);
  }
}
function strokeTo(mask, a, b, r, value) {   // stamps close enough to leave no gaps
  const d = Math.hypot(b[0] - a[0], b[1] - a[1]), steps = Math.max(1, Math.ceil(d / Math.max(1, r / 3)));
  for (let i = 0; i <= steps; i++) stamp(mask, a[0] + ((b[0] - a[0]) * i) / steps, a[1] + ((b[1] - a[1]) * i) / steps, r, value);
}
const patchPoint = (p) => [(p[0] - Pa.cur.x) / Pa.ds, (p[1] - Pa.cur.y) / Pa.ds];   // slide pixels to mask pixels

/* ------------------------------------------------------------ the regions of the open patch */
const newRegion = (mask, labels, opened) => ({ mask, labels: new Set(labels), opened: !!opened, cv: null, stale: true });
const labelledDraft = () => (Pa.draft && Pa.draft.labels.size && maskAny(Pa.draft.mask) ? Pa.draft : null);
const allRegions = () => { const d = labelledDraft(); return d ? [...Pa.cur.regions, d] : Pa.cur.regions; };   // what is annotated: an unlabelled draft is not, yet
function unionOf(regions) {
  const u = new Uint8Array(paN());
  for (const r of regions) { const m = r.mask; for (let i = 0; i < m.length; i++) if (m[i]) u[i] = 1; }
  return u;
}
function countOn(m) { let c = 0; for (let i = 0; i < m.length; i++) c += m[i]; return c; }
function paMeasure() { Pa.cur.covered = countOn(unionOf(allRegions())); }
const sameLabels = (a, b) => a.size === b.size && [...a].every((k) => b.has(k));
function commitDraft() {   // the draft joins the regions; those with the same labels are one region
  const d = labelledDraft();
  if (!d) return;
  const same = Pa.cur.regions.find((r) => sameLabels(r.labels, d.labels));
  if (same) {
    for (let i = 0; i < d.mask.length; i++) if (d.mask[i]) same.mask[i] = 1;
    same.stale = true;
  } else { d.opened = false; Pa.cur.regions.push(d); }
  Pa.draft = null;
}
function drawTarget() {   // where the next stroke goes: a draft still waiting for its label, the opened region, or a new draft
  if (Pa.draft && Pa.draft.labels.size && !Pa.draft.opened) commitDraft();
  if (Pa.draft && !Pa.draft.labels.size && !maskAny(Pa.draft.mask)) Pa.draft = null;
  if (!Pa.draft) Pa.draft = newRegion(new Uint8Array(paN()), [], false);
  return Pa.draft;
}
function serial(regions) { return regions.map((r) => ({ labels: paInOrder(r.labels), counts: rleEncode(r.mask) })); }

/* Undo holds the state before each change, run-length encoded: a patch is small that way, and it is a single case to restore. */
function snapshot() {
  const c = Pa.cur, d = Pa.draft;
  return { regions: serial(c.regions), draft: d ? { labels: [...d.labels], counts: rleEncode(d.mask), opened: d.opened } : null };
}
function pushUndo() {
  const c = Pa.cur;
  c.undo.push(snapshot());
  if (c.undo.length > UNDO_MAX) c.undo.shift();
  c.redo = [];
  paToolState();
}
function restore(s) {
  Pa.cur.regions = s.regions.map((r) => newRegion(rleDecode(r.counts), r.labels));
  Pa.draft = s.draft ? newRegion(rleDecode(s.draft.counts), s.draft.labels, s.draft.opened) : null;
}
function paUndoRedo(action) {
  const c = Pa.cur;
  if (!c || c.loading || Pa.stroke) return;
  const from = action === 'undo' ? c.undo : c.redo, to = action === 'undo' ? c.redo : c.undo;
  if (!from.length) return;
  to.push(snapshot());
  restore(from.pop());
  paTouch();
  paOp(action);
  paRender(); edDraw();
}

/* ------------------------------------------------------------ the log of how the reviewer worked */
const A = { session: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`, seq: 0, visits: 0, queue: [], busy: false, lastInput: Date.now(), lastView: 0 };
function paLog(type, detail = {}) {
  if (!Pa.owner) return;   // no initials, no workspace to log it under
  const v = Pa.visit;
  A.queue.push({ seq: ++A.seq, t: Date.now(), reviewer: Pa.owner, type, visit: v ? v.id : null, sid: v ? v.sid : S.sid, x: v ? v.x : null, y: v ? v.y : null, detail });
  if (A.queue.length > LOG_MAX) A.queue.shift();
}
const coveredNow = () => (Pa.cur && !Pa.cur.loading ? Math.round((Pa.cur.covered / paN()) * 1e4) / 1e4 : 0);
function paOp(type, detail = {}) { paLog(type, { ...detail, covered: coveredNow() }); }   // something done to the open patch
function visitClock() {
  const v = Pa.visit;
  if (!v) return;
  const now = Date.now(), step = now - v.last;
  v.last = now;
  if (document.visibilityState === 'visible' && now - A.lastInput < IDLE_MS) v.active += Math.min(step, MAX_STEP_MS);
}
const visitTimes = () => ({ active_ms: Pa.visit.active, elapsed_ms: Date.now() - Pa.visit.start });
function visitStart(via) {
  const c = Pa.cur, now = Date.now();
  A.lastInput = now;   // arriving at a patch counts as being there
  Pa.visit = { id: `${A.session}-${++A.visits}`, sid: c.sid, x: c.x, y: c.y, start: now, last: now, active: 0, ticked: now };
  paOp('open', { via });
}
function visitEnd(reason) {
  if (!Pa.visit) return;
  visitClock();
  const c = Pa.cur;
  paOp('close', { reason, status: !c || !c.regions.length && !labelledDraft() ? 'empty' : c.covered === paN() ? 'complete' : 'partial', ...visitTimes() });
  Pa.visit = null;
}
async function actSend(leaving) {
  if (!A.queue.length || (A.busy && !leaving)) return;
  const batch = A.queue.slice(0, SEND_MAX), body = JSON.stringify({ session: A.session, events: batch });
  if (leaving) {   // the page is closing: this is the only way that outlives it
    if (navigator.sendBeacon('api/patch_activity', new Blob([body], { type: 'application/json' }))) A.queue.splice(0, batch.length);
    return;
  }
  A.busy = true;
  try { await post('api/patch_activity', { session: A.session, events: batch }); A.queue.splice(0, batch.length); } catch (e) { /* kept, and sent again: the annotator need not hear about it */ }
  A.busy = false;
}
setInterval(() => {
  visitClock();
  const v = Pa.visit;
  if (v && Date.now() - v.ticked >= TICK_MS) {   // so a crash loses at most this much of the time on a patch
    v.ticked = Date.now();
    paLog('tick', visitTimes());
  }
}, CLOCK_MS);
setInterval(() => actSend(false), SEND_MS);
['pointermove', 'pointerdown', 'keydown', 'wheel'].forEach((t) => document.addEventListener(t, () => { A.lastInput = Date.now(); }, { capture: true, passive: true }));
viewer.addHandler('animation-finish', () => {   // where they looked: part of how a patch comes to be picked
  if (!modeIs('patches') || !slideShown() || Date.now() - A.lastView < VIEW_MS) return;
  A.lastView = Date.now();
  const b = slideItem().viewportToImageRectangle(viewer.viewport.getBounds(true));
  paLog('view', { vx: Math.round(b.x), vy: Math.round(b.y), vw: Math.round(b.width), vh: Math.round(b.height) });
});
window.addEventListener('pagehide', () => { paSaveNow(); visitEnd('page closed'); actSend(true); });
window.addEventListener('pageshow', (e) => { if (e.persisted && Pa.cur && !Pa.cur.loading && !Pa.visit) visitStart('page reopened'); });

/* ------------------------------------------------------------ data */
const paOwnerKey = (s) => (s || '').trim().toLowerCase();
async function patchLoad() {
  const owner = Pa.owner;
  try {
    const d = await api(`api/patches?reviewer=${encodeURIComponent(owner)}`);
    if (owner !== Pa.owner) return;
    Pa.size = d.size; Pa.mask = d.mask_size; Pa.ds = d.size / d.mask_size; Pa.mag = d.magnification; Pa.labels = d.labels; Pa.loaded = true;
    Pa.bySlide = new Map();
    d.patches.forEach(paStore);
    // saves still on their way are newer than what the server just listed
    Q.waiting.forEach((job) => { if (job.owner === Pa.owner) paStore(jobSummary(job)); });
  } catch (e) { toast('Could not load the patches: ' + errDetail(e), true); }
  patchRefresh();
}
function paStore(s) {   // the state of one patch, as saved or as just sent
  if (!Pa.bySlide.has(s.slide_id)) Pa.bySlide.set(s.slide_id, new Map());
  const cells = Pa.bySlide.get(s.slide_id), k = `${s.x},${s.y}`;
  if (s.status === 'empty') cells.delete(k); else cells.set(k, s);
}
function patchRefresh() {   // everything that shows the patches
  const n = [...Pa.bySlide.keys()].reduce((a, sid) => a + paCount(sid).done, 0);
  $('#patchCount').textContent = n ? `(${n})` : '';
  if (S.slides.length && modeIs('patches')) { renderSlideList(); renderProgress(); }
  if (modeIs('patches')) { paList(); paMeter(); }
  edDraw();
}
function patchSetOwner() {
  const next = reviewerInput.value.trim();
  if (next === Pa.owner) return;
  if (paOwnerKey(next) !== paOwnerKey(Pa.owner)) {
    paClose('initials changed');   // saved in the workspace it was drawn in
    Pa.bySlide = new Map(); Pa.loaded = false;
  }
  Pa.owner = next;
  paRender();
  patchRefresh();
  patchLoad();
}
function patchWho() {   // the initials to work under, asking for them if there are none
  if (!reviewer()) return '';
  patchSetOwner();
  return Pa.owner;
}
reviewerInput.addEventListener('change', patchSetOwner);

/* Saving: one request at a time, and the latest state of each patch. What fails stays in the queue and is tried again, so a
   lost connection does not lose work. */
const Q = { waiting: new Map(), busy: null, err: null, timer: null };
const jobKey = (owner, sid, x, y) => `${paOwnerKey(owner)}|${sid}|${x},${y}`;
const pendingJob = (k) => Q.waiting.get(k) || (Q.busy && Q.busy.k === k ? Q.busy.job : null);
function jobSummary(job) {
  return { slide_id: job.sid, x: job.x, y: job.y, coverage: job.covered / paN(), comment: job.comment, ts: '',
    status: !job.regions.length ? 'empty' : job.covered === paN() ? 'complete' : 'partial', fractions: job.fractions };
}
async function qPump() {
  clearTimeout(Q.timer);
  if (Q.busy) return;
  const first = Q.waiting.entries().next();
  if (first.done) { paSaveState(); return; }
  const [k, job] = first.value;
  Q.waiting.delete(k);
  Q.busy = { k, job };
  paSaveState();
  try {
    const saved = await post(`api/patches/${encodeURIComponent(job.sid)}/${job.x}/${job.y}`, { reviewer: job.owner, regions: job.regions, comment: job.comment });
    Q.busy = null; Q.err = null;
    if (paOwnerKey(job.owner) === paOwnerKey(Pa.owner) && !Q.waiting.has(k)) { paStore(saved); patchRefresh(); }
    qPump();
  } catch (e) {
    Q.busy = null;
    if (!Q.waiting.has(k)) Q.waiting.set(k, job);
    const why = /^\d+ /.test(e.message || '') ? errDetail(e) : 'No connection to the server.';
    if (!Q.err) toast('Patch not saved: ' + why, true);
    Q.err = why;
    paSaveState();
    Q.timer = setTimeout(qPump, RETRY_MS);
  }
}
function paSaveNow() {
  const c = Pa.cur;
  if (!c || !c.dirty || c.loading) return;
  clearTimeout(c.timer);
  c.dirty = false;
  const regions = allRegions(), fractions = {}, sent = serial(regions), state = JSON.stringify([sent, c.comment.trim()]);
  if (state === c.last) { paSaveState(); return; }   // drawing a region that has no label yet changes nothing that is kept
  c.last = state;
  regions.forEach((r) => r.labels.forEach((k) => { fractions[k] = 1; }));   // which labels; the server sends the shares back
  const job = { owner: Pa.owner, sid: c.sid, x: c.x, y: c.y, regions: sent, comment: c.comment, covered: c.covered, fractions };
  Q.waiting.set(jobKey(job.owner, job.sid, job.x, job.y), job);
  paStore(jobSummary(job));
  patchRefresh();
  qPump();
}
function paTouch() {   // the annotation changed
  const c = Pa.cur;
  paMeasure();
  c.dirty = true;
  clearTimeout(c.timer);
  c.timer = setTimeout(paSaveNow, SAVE_DELAY_MS);
  paMeter(); paToolState(); paSaveState();
}
window.addEventListener('beforeunload', (e) => {
  paSaveNow();
  const unlabelled = Pa.draft && !Pa.draft.labels.size && maskAny(Pa.draft.mask);
  if (Q.waiting.size || Q.busy || unlabelled) e.preventDefault();
});

/* ------------------------------------------------------------ opening and closing a patch */
function dropDrag() {
  if (Pa.drag && Pa.tool === 'polygon' && Pa.drag.length >= 3) toast('Unfinished polygon discarded. Close it first to keep it.', true);
  Pa.drag = null;
}
function paClose(reason = 'closed') {   // whatever is labelled is kept; a region without a label is not annotated yet, so it goes, and says so
  const c = Pa.cur;
  if (!c) return;
  dropDrag(); Pa.stroke = null;
  if (c.loading) { Pa.gen++; Pa.cur = null; Pa.draft = null; return; }
  if (Pa.draft && !Pa.draft.labels.size && maskAny(Pa.draft.mask)) toast('A region without a label was not kept', true);
  paSaveNow();
  visitEnd(reason);
  commitDraft();
  clearTimeout(c.timer);
  Pa.cur = null; Pa.draft = null;
}
/* Every pixel of a patch has to be reachable, so it goes where the panels on the viewer leave room: the largest square clear of
   them, wherever they have been moved to. */
function clearOfPanels() {
  const v = $('#viewer').getBoundingClientRect();
  const free = { x0: PICK_PAD_PX, y0: PICK_PAD_PX, x1: v.width - PICK_PAD_PX, y1: v.height - PICK_PAD_PX };
  const side = (f) => Math.min(f.x1 - f.x0, f.y1 - f.y0);
  document.querySelectorAll('#viewerWrap .fp').forEach((el) => {
    if (getComputedStyle(el).display === 'none' || el.classList.contains('hidden')) return;
    const r = el.getBoundingClientRect(), a = { x0: r.left - v.left - PICK_PAD_PX, y0: r.top - v.top - PICK_PAD_PX, x1: r.right - v.left + PICK_PAD_PX, y1: r.bottom - v.top + PICK_PAD_PX };
    if (a.x1 <= free.x0 || a.x0 >= free.x1 || a.y1 <= free.y0 || a.y0 >= free.y1) return;
    const options = [{ ...free, x0: a.x1 }, { ...free, x1: a.x0 }, { ...free, y0: a.y1 }, { ...free, y1: a.y0 }];
    Object.assign(free, options.reduce((best, o) => (side(o) > side(best) ? o : best)));
  });
  return { ...free, W: v.width, H: v.height, side: Math.max(side(free), 0) };
}
function paZoom() {
  const c = Pa.cur;
  if (!c) return;
  const f = clearOfPanels();
  let d = Math.min(f.side, Pa.size * 2);   // 2 screen pixels per slide pixel is as far as the viewer zooms
  if (d < 120) d = Math.min(f.W, f.H) - 2 * PICK_PAD_PX;   // panels take up nearly everything: fold them to see more, it still has to open
  if (d <= 0) return;
  const k = Pa.size / d, sx = (f.x0 + f.x1 - d) / 2, sy = (f.y0 + f.y1 - d) / 2;   // slide pixels per screen pixel, and where the patch goes
  viewer.viewport.fitBounds(imgRect({ x: c.x - sx * k, y: c.y - sy * k, w: f.W * k, h: f.H * k }));
}
async function paOpen(x, y, via = 'other') {   // via: how the reviewer got here
  paClose('moved to another patch');
  const sid = S.sid, owner = Pa.owner, gen = ++Pa.gen, k = jobKey(owner, sid, x, y);
  Pa.cur = { sid, x, y, loading: true, regions: [], comment: '', dirty: false, timer: null, covered: 0, undo: [], redo: [] };
  Pa.hoverCell = null;
  Panels.showReview();
  paZoom();
  if (Pa.tool === 'move') setPaTool(Pa.lastDraw);
  paRender(); edDraw();
  $('#paPanel').scrollTop = 0;
  const job = pendingJob(k);   // what was just drawn here may not have reached the server yet
  let d = job;
  if (!d) {
    if (Pa.loaded && !paCells(sid).has(`${x},${y}`)) d = { regions: [], comment: '' };   // nothing there: no need to ask
    else {
      try { d = await api(`api/patches/${encodeURIComponent(sid)}/${x}/${y}?reviewer=${encodeURIComponent(owner)}`); } catch (e) {
        if (gen === Pa.gen) { toast('Could not open the patch: ' + errDetail(e), true); Pa.cur = null; paRender(); edDraw(); }
        return;
      }
    }
  }
  if (gen !== Pa.gen) return;
  const c = Pa.cur;
  c.regions = d.regions.map((r) => newRegion(rleDecode(r.counts), r.labels));
  c.comment = d.comment || '';
  c.loading = false;
  c.last = JSON.stringify([serial(c.regions), c.comment.trim()]);
  paMeasure();
  visitStart(via);
  paRender(); edDraw();
}
function cellAt(p) {   // the patch under a point; only whole patches are in the grid
  if (!S.info) return null;
  const x = Math.floor(p[0] / Pa.size) * Pa.size, y = Math.floor(p[1] / Pa.size) * Pa.size;
  return x >= 0 && y >= 0 && x + Pa.size <= S.info.width && y + Pa.size <= S.info.height ? { x, y } : null;
}
function paStep(dx, dy, via = 'arrow key') {
  const c = Pa.cur;
  if (!c || !S.info) return;
  const x = c.x + dx * Pa.size, y = c.y + dy * Pa.size;
  if (x < 0 || y < 0 || x + Pa.size > S.info.width || y + Pa.size > S.info.height) { toast('That is the edge of the grid', true); return; }
  paOpen(x, y, via);
}
function patchFlush(reason) { paClose(reason); paRender(); edDraw(); }
function patchOnSlide() {
  Pa.hoverCell = null; Pa.press = null;
  if (modeIs('patches')) { paLog('slide'); paRender(); $('#paPanel').scrollTop = 0; }
}

/* ------------------------------------------------------------ actions on the regions */
function paToggleLabel(key) {
  if (!Pa.cur || Pa.cur.loading) return;
  if (!Pa.draft || !maskAny(Pa.draft.mask)) { toast('Draw a region first, then say what it shows', true); return; }
  pushUndo();
  const d = Pa.draft;
  if (d.labels.has(key)) d.labels.delete(key); else d.labels.add(key);
  d.stale = true;
  paTouch();
  paOp('label', { key, on: d.labels.has(key) });
  paRender(); edDraw();
}
/* What a click on a label means. A region without a label gets it, and so does one opened from the list, where it is switched on
   or off. Otherwise the region just drawn is finished, and the label goes to everything not yet annotated. Shift adds it to the
   region just drawn as a second label. So a label can never land on the last region by accident. */
function paLabelClick(key, shift) {
  const c = Pa.cur;
  if (!c || c.loading) return;
  const d = Pa.draft && maskAny(Pa.draft.mask) ? Pa.draft : null;
  if (d && (!d.labels.size || d.opened || shift)) { paToggleLabel(key); return; }
  if (c.covered === paN()) {
    toast(d ? 'The whole patch is covered. Shift+click adds a second label to this region.' : 'The whole patch is covered. To change a region, click it in the list.', true);
    return;
  }
  if (c.covered === 0) { toast("Draw a region first. If the patch is all one type, press Select whole patch.", true); return; }
  pushUndo();
  commitDraft();
  const u = unionOf(c.regions), m = new Uint8Array(paN());
  for (let i = 0; i < m.length; i++) if (!u[i]) m[i] = 1;
  const share = toGoText(c.covered / paN());
  Pa.draft = newRegion(m, [key], false);
  paTouch();
  paOp('label_rest', { key, area: Math.round((countOn(m) / paN()) * 1e4) / 1e4 });
  toast(`The remaining ${share} is now ${paLabelText(key)}. Z undoes this.`);
  paRender(); edDraw();
}
function paSelectRest() {
  if (!Pa.cur || Pa.cur.loading) return;
  if (Pa.cur.covered === paN() && !(Pa.draft && !Pa.draft.labels.size)) { toast('The whole patch is covered'); return; }
  pushUndo();
  commitDraft();
  const u = unionOf(Pa.cur.regions), m = Pa.draft ? Pa.draft.mask : new Uint8Array(paN());   // a region still waiting for its label stays in
  for (let i = 0; i < m.length; i++) if (!u[i]) m[i] = 1;
  Pa.draft = newRegion(m, Pa.draft ? [...Pa.draft.labels] : [], Pa.draft ? Pa.draft.opened : false);
  paTouch();
  paOp('select_rest');
  paRender(); edDraw();
}
function paOpenRegion(i) {   // a region from the list is on the draft's place: its labels can be changed, and strokes add to it or erase from it
  if (!Pa.cur || Pa.cur.loading) return;
  pushUndo();
  commitDraft();
  const r = Pa.cur.regions.splice(i, 1)[0];
  if (!r) return;
  r.opened = true;
  Pa.draft = r;
  paOp('region_open', { labels: paInOrder(r.labels) });
  paRender(); edDraw();
}
function paCloseRegion() { commitDraft(); paMeasure(); paOp('region_close'); paRender(); edDraw(); }
function paDeleteRegion(i) {   // i: in the list, or null for the draft
  if (!Pa.cur || Pa.cur.loading) return;
  pushUndo();
  const gone = i == null ? Pa.draft : Pa.cur.regions[i];
  if (i == null) Pa.draft = null; else Pa.cur.regions.splice(i, 1);
  paTouch();
  paOp('region_delete', { labels: gone ? paInOrder(gone.labels) : [] });
  paRender(); edDraw();
}

/* ------------------------------------------------------------ review panel */
function paMeter() {
  const el = $('#paMeter');
  if (!el || !Pa.cur || Pa.cur.loading) return;
  const f = Pa.cur.covered / paN(), done = Pa.cur.covered === paN();
  el.classList.toggle('done', done);
  el.querySelector('i').style.width = `${f * 100}%`;
  $('#paMeterText').innerHTML = done ? '<b>Complete</b>, every pixel is annotated' : `<b>${pctText(f)}</b> covered, ${toGoText(f)} to go`;
  const rest = $('#paRest');
  if (rest) rest.classList.toggle('hidden', done && !(Pa.draft && !Pa.draft.labels.size));
}
function paSaveState() {
  const el = $('#paSaved');
  if (!el) return;
  const waiting = Q.err ? Q.waiting.size + (Q.busy ? 1 : 0) : 0;
  el.className = 'small ' + (waiting ? 'pa-err' : 'muted');
  if (waiting) { el.innerHTML = `Not saved yet, trying again. ${esc(Q.err)} <button class="mini" id="paRetry">Retry now</button>`; el.querySelector('#paRetry').addEventListener('click', qPump); return; }
  const c = Pa.cur, busy = Q.busy || Q.waiting.size || (c && c.dirty);
  el.textContent = busy ? 'Saving…' : c && (c.regions.length || labelledDraft()) ? `Saved as ${Pa.owner}` : '';
}
function paList() {
  const el = $('#paList'), side = $('#paListCount'), n = paCount(S.sid);
  if (side) side.textContent = `${n.done} complete${n.part ? `, ${n.part} partial` : ''}`;
  if (!el) return;
  const cells = [...paCells(S.sid).values()].sort((a, b) => a.y - b.y || a.x - b.x);
  el.innerHTML = !cells.length ? '<div class="small muted">None yet.</div>' : cells.slice(0, LIST_MAX).map((s) => `
      <button type="button" class="parow ${Pa.cur && Pa.cur.x === s.x && Pa.cur.y === s.y ? 'on' : ''}" data-cell="${s.x},${s.y}" title="Level-0 corner ${s.x}, ${s.y}">
        <span class="pa-dot ${s.status}"></span><span class="parow-body"><span>${esc(cellName(s.x, s.y))} <span class="muted small">${s.status === 'complete' ? '' : pctText(s.coverage)}</span></span>
        <span class="rchips">${paChips(Pa.labels.map((l) => l.key).filter((k) => k in s.fractions))}</span>${s.comment ? `<span class="small rrow-note">${esc(s.comment)}</span>` : ''}</span></button>`).join('') +
      (cells.length > LIST_MAX ? `<div class="small muted">And ${cells.length - LIST_MAX} more. They are all on the grid and in the export.</div>` : '');
  el.querySelectorAll('[data-cell]').forEach((b) => b.addEventListener('click', () => { const [x, y] = b.dataset.cell.split(',').map(Number); paOpen(x, y, 'list in the panel'); }));
}
function regionRow(r, i) {
  const f = countOn(r.mask) / paN(), isDraft = i == null, keys = paInOrder(r.labels);
  return `<div class="parg ${isDraft ? 'on' : ''}">
      <button type="button" class="parg-main" ${isDraft ? 'data-rg-close' : `data-rg="${i}"`} title="${isDraft ? 'Done with this region' : 'Change this region'}">
        <span class="rchips">${keys.length ? paChips(keys) : '<span class="pa-need">No label yet</span>'}</span><span class="muted small">${pctText(f)}</span></button>
      <button type="button" class="mini ghost danger" ${isDraft ? 'data-rg-del-draft' : `data-rg-del="${i}"`} title="Delete this region" aria-label="Delete this region">✕</button></div>`;
}
function labelButtons() {
  const on = Pa.draft ? Pa.draft.labels : new Set(), c = Pa.cur, idle = !Pa.draft && (c.covered === 0 || c.covered === paN());
  const btn = (l) => `<button type="button" class="roi-opt ${on.has(l.key) ? 'on' : ''}" data-palabel="${esc(l.key)}" aria-pressed="${on.has(l.key)}"><i style="background:${esc(l.color)}"></i><span>${esc(l.name)}</span><kbd>${paKeyOf(l.key).toUpperCase()}</kbd></button>`;
  const groups = [...new Set(Pa.labels.map((l) => l.group).filter(Boolean))];
  return `<div class="roi-opts ${idle ? 'idle' : ''}">${Pa.labels.filter((l) => !l.group).map(btn).join('')}</div>` +
    groups.map((g) => `<div class="rform-group">${esc(g)}</div><div class="roi-opts two ${idle ? 'idle' : ''}">${Pa.labels.filter((l) => l.group === g).map(btn).join('')}</div>`).join('');
}
function paRender() {
  const p = $('#paPanel');
  if (!modeIs('patches')) return;
  if (!S.sid) { p.className = 'spanel empty'; p.textContent = 'Select a slide.'; return; }
  if (!Pa.owner) { p.className = 'spanel empty'; p.textContent = 'Enter your initials at the top to open your own workspace. Only the patches under your initials are shown here.'; return; }
  p.className = 'spanel';
  const top = p.scrollTop, c = Pa.cur, t = PA_TOOLS[Pa.tool], um = S.info && S.info.mpp ? Pa.size * S.info.mpp : null;
  const sized = Pa.tool === 'brush' || Pa.tool === 'eraser', d = Pa.draft && maskAny(Pa.draft.mask) ? Pa.draft : null;
  let body;
  if (!c) {
    body = `<ol class="rintro">
        <li>Zoom in until the grid shows, then click a patch. ${Pa.mask} × ${Pa.mask} pixels at ${esc(Pa.mag)}${um ? `, ${Math.round(um)} µm across` : ''}.</li>
        <li>Draw a region with <b>Brush</b> <kbd>R</kbd>, <b>Polygon</b> <kbd>D</kbd> or <b>Freehand</b> <kbd>L</kbd>.</li>
        <li>Choose what it shows here.</li>
        <li>Go on until the whole patch is covered. Regions may overlap.</li></ol>
      <div class="small muted rworkspace">Your workspace: <b class="txt">${esc(Pa.owner)}</b>. Other reviewers don't see these patches here, and you don't see theirs. The order you pick patches in, your time on each and your steps are recorded.</div>`;
  } else if (c.loading) {
    body = '<div class="small muted">Opening the patch…</div>';
  } else {
    const left = toGoText(c.covered / paN()), done = c.covered === paN(), open = d && d.opened, fresh = d && d.labels.size && !d.opened;
    const ask = d && !d.labels.size ? 'What does this region show?'
      : open ? 'Changing this region. Click labels to add or remove them. Esc when done.'
      : done ? 'Press Enter for the next patch, or draw on to add to this one.'
      : c.covered === 0 ? 'Draw a region, then choose what it shows. Is it all one type? Press Select whole patch.'
      : `Draw the next region, or choose a label for the remaining ${left}.`;
    const askClass = (d && !d.labels.size) || (!d && c.covered > 0 && !done) || (fresh && !done) ? 'pa-ask' : '';
    body = `<div class="rform-head"><b>${esc(cellName(c.x, c.y))}</b><span class="spacer"></span>
        <button class="ghost zoom" id="paZoom" title="Zoom to this patch">⌖</button><button class="ghost" id="paDone" title="Close this patch (Esc)">Close</button></div>
      <div id="paMeter" class="pa-meter"><i></i></div>
      <div class="pa-meter-row"><span id="paMeterText"></span><button class="mini" id="paRest" title="Select everything not yet in a region, so it can be labelled (G)">${c.covered === 0 ? 'Select whole patch' : "Select what's left"} <kbd>G</kbd></button></div>
      <div id="paSaved" class="small muted"></div>
      ${Pa.tool === 'move' ? '' : `<div class="pa-hint">${esc(t.hint)}</div>`}
      ${sized ? `<label class="pa-size">Size <input id="paSize" type="range" min="4" max="256" step="2" value="${Pa.brush}" aria-label="Brush size"> <span id="paSizeVal" class="tabnum"></span></label>` : ''}
      <div class="rform-q ${askClass}">${esc(ask)}</div>
      ${labelButtons()}
      ${fresh ? '<div class="small muted pa-second">Shows two things? Shift+click adds a second label to the last region.</div>' : ''}
      <div class="rform-group">Regions in this patch</div>
      <div id="paRegions">${(d ? regionRow(d, null) : '') + c.regions.map(regionRow).join('') || '<div class="small muted">None yet.</div>'}</div>
      <label class="pa-show"><input type="checkbox" id="paHide" ${Pa.hide ? '' : 'checked'}> Show regions <kbd>H</kbd> <input id="paAlpha" type="range" min="0.1" max="0.9" step="0.05" value="${Pa.alpha}" aria-label="Opacity of the regions"></label>
      <textarea id="paComment" placeholder="Comment on this patch (optional)" aria-label="Comment on this patch" maxlength="4000">${esc(c.comment)}</textarea>
      <div class="small muted pa-move">Arrow keys go to the next patch. Your work is saved as you go.</div>`;
  }
  p.innerHTML = (c ? `<div class="rform">${body}</div>` : body) + section('palist', 'Your patches on this slide', '<span class="small muted" id="paListCount"></span>', '<div id="paList"></div>');
  p.scrollTop = top;
  p.querySelectorAll('[data-fold]').forEach((b) => b.addEventListener('click', () => { Panels.toggleSec(b.dataset.fold); paRender(); }));
  p.querySelectorAll('[data-palabel]').forEach((b) => b.addEventListener('click', (e) => { paLabelClick(b.dataset.palabel, e.shiftKey); }));
  p.querySelectorAll('[data-rg]').forEach((b) => b.addEventListener('click', () => paOpenRegion(+b.dataset.rg)));
  p.querySelectorAll('[data-rg-del]').forEach((b) => b.addEventListener('click', () => paDeleteRegion(+b.dataset.rgDel)));
  const bind = (sel, ev, fn) => { const el = p.querySelector(sel); if (el) el.addEventListener(ev, fn); };
  bind('[data-rg-close]', 'click', paCloseRegion);
  bind('[data-rg-del-draft]', 'click', () => paDeleteRegion(null));
  bind('#paZoom', 'click', () => { paZoom(); paOp('zoom_to_patch'); });
  bind('#paDone', 'click', () => patchFlush('closed'));
  bind('#paRest', 'click', paSelectRest);
  bind('#paSize', 'input', (e) => { Pa.brush = +e.target.value; paSizeText(); edDraw(); });
  bind('#paHide', 'change', (e) => { Pa.hide = !e.target.checked; paOp('hide', { hidden: Pa.hide }); edDraw(); });
  bind('#paAlpha', 'input', (e) => { Pa.alpha = +e.target.value; edDraw(); });
  ['#paSize', '#paAlpha', '#paHide'].forEach((s) => bind(s, 'change', (e) => e.target.blur()));   // or the shortcuts would go to the control
  bind('#paComment', 'input', (e) => { if (Pa.cur) { Pa.cur.comment = e.target.value; paTouch(); } });
  bind('#paComment', 'change', (e) => paOp('comment', { length: e.target.value.trim().length }));
  paSizeText(); paList(); paMeter(); paSaveState(); paToolState();
}
function paSizeText() {
  const el = $('#paSizeVal');
  if (el) el.textContent = `${Pa.brush} px${S.info && S.info.mpp ? `, ${Math.round(Pa.brush * Pa.ds * S.info.mpp)} µm` : ''}`;
}

/* ------------------------------------------------------------ on the slide, called from paint() in editor.js */
function layerCanvas(r) {
  if (!r.cv) { r.cv = document.createElement('canvas'); r.cv.width = r.cv.height = Pa.mask; r.stale = true; }
  if (r.stale) {
    r.stale = false;
    const n = Pa.mask, ctx = r.cv.getContext('2d'), img = ctx.createImageData(n, n), px = new Uint32Array(img.data.buffer), m = r.mask;
    const rgba = (hex) => { const v = parseInt(hex.slice(1), 16); return (255 << 24 | (v & 255) << 16 | (v >> 8 & 255) << 8 | v >> 16) >>> 0; };
    const cols = r.labels.size ? paInOrder(r.labels).map((k) => rgba(paLabel(k).color)) : [0xffffffff];
    if (cols.length === 1) { for (let i = 0; i < m.length; i++) if (m[i]) px[i] = cols[0]; } else {   // more than one label: stripes
      for (let y = 0, i = 0; y < n; y++) for (let x = 0; x < n; x++, i++) if (m[i]) px[i] = cols[((x + y) >> 4) % cols.length];
    }
    ctx.putImageData(img, 0, 0);
  }
  return r.cv;
}
const STATUS_FILL = { complete: 'rgba(87,200,120,.32)', partial: 'rgba(255,190,70,.32)' };
const STATUS_LINE = { complete: '#57c878', partial: '#ffbe46' };
function patchPaint(m, W, H) {
  if (!modeIs('patches') || !slideShown()) return;
  const n = Pa.size, c = n * m.s, grid = c >= GRID_MIN_PX, cols = Math.floor(S.info.width / n), rows = Math.floor(S.info.height / n);
  const col0 = Math.max(0, Math.floor(-m.x([0, 0]) / c)), col1 = Math.min(cols, Math.ceil((W - m.x([0, 0])) / c));
  const row0 = Math.max(0, Math.floor(-m.y([0, 0]) / c)), row1 = Math.min(rows, Math.ceil((H - m.y([0, 0])) / c));
  // patches annotated so far; a little larger than they are when zoomed far out, or they could not be seen
  paCells(S.sid).forEach((s) => {
    const col = s.x / n, row = s.y / n;
    if (col < col0 - 1 || col > col1 || row < row0 - 1 || row > row1) return;
    const side = Math.max(c, 4), x = m.x([s.x, 0]), y = m.y([0, s.y]);
    edCtx.fillStyle = STATUS_FILL[s.status]; edCtx.fillRect(x, y, side, side);
    if (grid) { edCtx.lineWidth = 1.5; edCtx.strokeStyle = STATUS_LINE[s.status]; edCtx.strokeRect(x + 0.75, y + 0.75, side - 1.5, side - 1.5); }
  });
  if (grid) {
    edCtx.beginPath();
    const yA = Math.max(0, m.y([0, row0 * n])), yB = Math.min(H, m.y([0, row1 * n])), xA = Math.max(0, m.x([col0 * n, 0])), xB = Math.min(W, m.x([col1 * n, 0]));
    for (let i = col0; i <= col1; i++) { const x = Math.round(m.x([i * n, 0])) + 0.5; edCtx.moveTo(x, yA); edCtx.lineTo(x, yB); }
    for (let j = row0; j <= row1; j++) { const y = Math.round(m.y([0, j * n])) + 0.5; edCtx.moveTo(xA, y); edCtx.lineTo(xB, y); }
    edCtx.lineWidth = 1; edCtx.strokeStyle = 'rgba(0,0,0,.28)'; edCtx.stroke();
    edCtx.strokeStyle = c >= 60 ? 'rgba(255,255,255,.55)' : 'rgba(255,255,255,.34)'; edCtx.stroke();
    if (Pa.hoverCell && !Pa.cur) {
      edCtx.lineWidth = 2; edCtx.strokeStyle = '#ffffff';
      edCtx.strokeRect(m.x([Pa.hoverCell.x, 0]), m.y([0, Pa.hoverCell.y]), c, c);
    }
  } else if (!Pa.cur) roiTag('Zoom in to see the grid, or click the slide to go there', 12, H - 12, '#ffffff');
  const cur = Pa.cur;
  if (!cur) return;
  const x0 = m.x([cur.x, 0]), y0 = m.y([0, cur.y]);
  edCtx.beginPath(); edCtx.rect(0, 0, W, H); edCtx.rect(x0, y0, c, c);
  edCtx.fillStyle = 'rgba(8,10,14,.55)'; edCtx.fill('evenodd');
  if (!cur.loading && !Pa.hide) {
    edCtx.imageSmoothingEnabled = false;
    edCtx.globalAlpha = Pa.alpha;
    cur.regions.forEach((r) => edCtx.drawImage(layerCanvas(r), x0, y0, c, c));
    if (Pa.draft) { edCtx.globalAlpha = Pa.draft.labels.size ? Math.min(0.9, Pa.alpha + 0.15) : 0.6; edCtx.drawImage(layerCanvas(Pa.draft), x0, y0, c, c); }
    edCtx.globalAlpha = 1;
  }
  edCtx.lineJoin = 'miter'; edCtx.lineWidth = 3; edCtx.strokeStyle = 'rgba(0,0,0,.8)'; edCtx.strokeRect(x0 - 1.5, y0 - 1.5, c + 3, c + 3);
  edCtx.lineWidth = 1.5; edCtx.strokeStyle = cur.covered === paN() ? STATUS_LINE.complete : '#ffffff'; edCtx.strokeRect(x0 - 1, y0 - 1, c + 2, c + 2);
  if (Pa.drag && Pa.drag.length) {
    const open = Pa.tool === 'polygon', pts = open && Pa.cursor ? [...Pa.drag, Pa.cursor] : Pa.drag;
    trace(pts, m, !open);
    if (!open || pts.length > 2) { edCtx.globalAlpha = 0.25; edCtx.fillStyle = '#ffffff'; edCtx.fill(); edCtx.globalAlpha = 1; }
    casedStroke('#ffffff', 2);
    if (open) {
      Pa.drag.forEach((p, i) => {
        edCtx.beginPath(); edCtx.arc(m.x(p), m.y(p), i ? 3.5 : 5.5, 0, 2 * Math.PI);
        edCtx.fillStyle = '#ffffff'; edCtx.fill(); edCtx.lineWidth = 1.5; edCtx.strokeStyle = '#000'; edCtx.stroke();
      });
    }
  }
  if ((Pa.tool === 'brush' || Pa.tool === 'eraser') && Pa.cursor && !Pa.space) {
    const r = Math.max((Pa.brush / 2) * Pa.ds * m.s, 2), x = m.x(Pa.cursor), y = m.y(Pa.cursor);
    edCtx.beginPath(); edCtx.arc(x, y, r, 0, 2 * Math.PI); edCtx.lineWidth = 1; edCtx.strokeStyle = '#000'; edCtx.stroke();
    edCtx.beginPath(); edCtx.arc(x, y, r + 1, 0, 2 * Math.PI); edCtx.strokeStyle = '#fff'; edCtx.stroke();
  }
}

/* ------------------------------------------------------------ pointer input. Captured on the way down, as in editor.js. */
const paDrawing = () => modeIs('patches') && Pa.tool !== 'move' && !Pa.space;
function pickAt(e) {
  const m = screenMap(), p = imgPoint(e);
  if (Pa.size * m.s < GRID_MIN_PX) {   // too small to aim at: go there, so the grid shows
    const v = $('#viewer'), s = ZOOM_TO_PX / Pa.size, w = v.clientWidth / s, h = v.clientHeight / s;
    viewer.viewport.fitBounds(imgRect({ x: p[0] - w / 2, y: p[1] - h / 2, w, h }));
    return;
  }
  const cell = cellAt(p);
  if (cell && patchWho()) paOpen(cell.x, cell.y, 'click on the grid');
}
function polygonPoint(p) {
  const now = performance.now(), m = screenMap(), near = (q) => Math.hypot(m.x(p) - m.x(q), m.y(p) - m.y(q)) < HANDLE_PX;
  if (!Pa.drag) { Pa.drag = [p]; Pa.cursor = p; Pa.dragAt = now; return; }
  const again = near(Pa.drag[Pa.drag.length - 1]) && now - Pa.dragAt < 400;   // the second click of a double-click
  Pa.dragAt = now;
  if (Pa.drag.length >= 3 && (again || near(Pa.drag[0]))) { closeShape(); return; }
  if (!again) Pa.drag.push(p);
}
function closeShape() {   // a shape may go past the edge of the patch, which is the easy way to reach the edge; only the patch is kept
  const pts = Pa.drag.map(patchPoint), tool = Pa.tool === 'polygon' ? 'polygon' : 'freehand';
  Pa.drag = null;
  const probe = new Uint8Array(paN());
  fillPolygon(probe, pts);
  if (!maskAny(probe)) { toast('That shape is outside this patch', true); edDraw(); return; }
  pushUndo();
  const d = drawTarget();
  for (let i = 0; i < probe.length; i++) if (probe[i]) d.mask[i] = 1;
  d.stale = true;
  paTouch();
  paOp(tool, { points: pts.length, area: Math.round((countOn(probe) / paN()) * 1e4) / 1e4 });
  paRender(); edDraw();
}
viewerEl.addEventListener('pointerdown', (e) => {
  if (!modeIs('patches') || e.button !== 0 || !slideItem() || !S.info) return;
  if (!paDrawing()) { Pa.press = Pa.tool === 'move' && !Pa.space ? { x: e.clientX, y: e.clientY } : null; return; }
  e.stopPropagation(); e.preventDefault();
  if (!slideShown()) { toast('The slide is still opening', true); return; }
  if (!patchWho()) return;
  if (!Pa.cur) { pickAt(e); return; }   // nothing open to draw on yet
  if (Pa.cur.loading) return;
  const p = imgPoint(e);
  Pa.cursor = p;
  if (Pa.tool === 'polygon') polygonPoint(p);
  else {
    viewerEl.setPointerCapture(e.pointerId);
    if (Pa.tool === 'lasso') Pa.drag = [p];
    else {
      pushUndo();
      const erase = Pa.tool === 'eraser';
      let targets;
      if (!erase) targets = [drawTarget()];
      else {
        if (Pa.draft && Pa.draft.labels.size && !Pa.draft.opened) commitDraft();
        targets = Pa.draft ? [Pa.draft] : Pa.cur.regions;
      }
      const q = patchPoint(p);
      Pa.stroke = { last: q, targets, value: erase ? 0 : 1, at: Date.now(), path: 0 };
      targets.forEach((t) => { strokeTo(t.mask, q, q, Pa.brush / 2, Pa.stroke.value); t.stale = true; });
    }
  }
  edDraw();
}, true);
viewerEl.addEventListener('pointermove', (e) => {
  if (!modeIs('patches') || !slideItem() || !S.info) return;
  const p = imgPoint(e);
  Pa.cursor = p;
  if (Pa.stroke) {
    if (!Pa.space) e.stopPropagation();
    const q = patchPoint(p), st = Pa.stroke;
    st.targets.forEach((t) => { strokeTo(t.mask, st.last, q, Pa.brush / 2, st.value); t.stale = true; });
    st.path += Math.hypot(q[0] - st.last[0], q[1] - st.last[1]);
    st.last = q;
    edDraw();
    return;
  }
  if (Pa.drag) {
    if (!Pa.space) e.stopPropagation();
    if (Pa.tool === 'lasso' && e.buttons & 1) {
      const m = screenMap(), last = Pa.drag[Pa.drag.length - 1];
      if (Math.hypot(m.x(p) - m.x(last), m.y(p) - m.y(last)) >= 2) Pa.drag.push(p);
    }
    edDraw();
    return;
  }
  if (paDrawing()) { e.stopPropagation(); edDraw(); return; }
  if (Pa.tool !== 'move' || e.buttons || !slideShown() || Pa.cur) return;
  const cell = Pa.size * screenMap().s >= GRID_MIN_PX ? cellAt(p) : null, was = Pa.hoverCell;
  if ((cell && cell.x) !== (was && was.x) || (cell && cell.y) !== (was && was.y)) {
    Pa.hoverCell = cell;
    if (cell) $('#viewerWrap').dataset.hot = '1'; else delete $('#viewerWrap').dataset.hot;
    edDraw();
  }
}, true);
viewerEl.addEventListener('pointerup', (e) => {
  if (!modeIs('patches') || !slideItem()) return;
  if (Pa.stroke) {
    e.stopPropagation();
    endStroke();
    return;
  }
  if (Pa.drag) {
    e.stopPropagation();
    if (Pa.tool === 'lasso') {
      if (Pa.drag.length >= 3) closeShape(); else { Pa.drag = null; edDraw(); }
    }
    return;
  }
  if (Pa.press && slideShown() && Math.hypot(e.clientX - Pa.press.x, e.clientY - Pa.press.y) < 4) {
    const cell = cellAt(imgPoint(e));
    if (Pa.cur && cell && cell.x === Pa.cur.x && cell.y === Pa.cur.y) { /* already open */ } else pickAt(e);
  }
  Pa.press = null;
}, true);
function endStroke() {
  const st = Pa.stroke;
  Pa.stroke = null;
  const c = Pa.cur;
  if (st.value === 0) {   // an erased region can end up empty: it is gone then
    c.regions = c.regions.filter((r) => maskAny(r.mask));
    if (Pa.draft && !maskAny(Pa.draft.mask)) Pa.draft = null;
  }
  paTouch();
  paOp(st.value === 0 ? 'erase' : 'brush', { size: Pa.brush, path: Math.round(st.path), ms: Date.now() - st.at });
  paRender(); edDraw();
}
viewerEl.addEventListener('pointercancel', () => {
  if (!modeIs('patches')) return;
  if (Pa.stroke) endStroke();
  if (Pa.drag && Pa.tool === 'lasso') Pa.drag = null;
  edDraw();
}, true);
viewerEl.addEventListener('pointerleave', () => {
  if (!modeIs('patches')) return;
  Pa.cursor = null;
  if (Pa.hoverCell) { Pa.hoverCell = null; delete $('#viewerWrap').dataset.hot; }
  edDraw();
});

/* ------------------------------------------------------------ tools on the viewer */
function paToolState() {
  const c = Pa.cur, can = !!c && !c.loading;
  $('#paUndoBtn').disabled = !can || !c.undo.length;
  $('#paRedoBtn').disabled = !can || !c.redo.length;
}
function setPaTool(t) {
  if (t !== 'move' && !patchWho()) return;
  if (t !== Pa.tool) { dropDrag(); if (Pa.loaded) paLog('tool', { tool: t }); }
  Pa.tool = t;
  if (t !== 'move') Pa.lastDraw = t === 'eraser' ? Pa.lastDraw : t;
  if (modeIs('patches')) $('#viewerWrap').dataset.tool = t === 'move' ? 'nav' : 'roi';
  delete $('#viewerWrap').dataset.hot;
  Pa.hoverCell = null;
  $('#patchToolsBody').querySelectorAll('[data-patool]').forEach((b) => {
    b.classList.toggle('on', b.dataset.patool === t);
    b.setAttribute('aria-pressed', b.dataset.patool === t);
  });
  paRender();
  edDraw();
}
$('#patchToolsBody').innerHTML = Object.entries(PA_TOOLS).map(([id, t]) => `
    <button data-patool="${id}" title="${t.label} (${t.key})"><svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">${t.icon}</svg><span>${t.label}</span></button>`).join('') + `
    <div class="tool-sep"></div>
    <button id="paUndoBtn" title="Undo in this patch (Z)"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 5L4 9l4 4M4 9h10a5 5 0 010 10h-3"/></svg><span>Undo</span></button>
    <button id="paRedoBtn" title="Redo in this patch (Shift+Z)"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 5l4 4-4 4M20 9H10a5 5 0 000 10h3"/></svg><span>Redo</span></button>`;
$('#patchToolsBody').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  b.blur();
  if (b.dataset.patool) setPaTool(b.dataset.patool);
  else paUndoRedo(b.id === 'paUndoBtn' ? 'undo' : 'redo');
});
document.addEventListener('keydown', (e) => {
  if (e.key === ' ' && modeIs('patches') && Pa.tool !== 'move' && !/INPUT|TEXTAREA|SELECT|BUTTON/.test(document.activeElement.tagName)) {
    Pa.space = true; e.preventDefault(); $('#viewerWrap').dataset.pan = '1'; edDraw();
  }
});
document.addEventListener('keyup', (e) => { if (e.key === ' ' && Pa.space) { Pa.space = false; delete $('#viewerWrap').dataset.pan; edDraw(); } });

/* ------------------------------------------------------------ keys, called from the handler in app.js. True = key used. */
const PA_ARROWS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
function patchKey(e, k) {
  if (k === ',' || k === '.' || k === '?' || k === ' ') return false;   // Space pans: its own listener above
  if (k === 'f') return false;
  const c = Pa.cur, polygon = Pa.drag && Pa.tool === 'polygon';
  if (PA_ARROWS[k]) {
    if (c) { paStep(...PA_ARROWS[k]); return true; }
    return k === 'ArrowLeft' || k === 'ArrowRight';   // with no patch open, up and down change the slide, as elsewhere
  }
  if (k === 'Enter') {
    if (polygon) { if (Pa.drag.length >= 3) closeShape(); else toast('A polygon needs at least 3 points', true); return true; }
    if (!c || c.loading) return true;
    if (Pa.draft && maskAny(Pa.draft.mask) && !Pa.draft.labels.size) { toast('Choose what this region shows first', true); return true; }
    if (Pa.draft) { paCloseRegion(); if (c.covered !== paN()) return true; }
    if (c.covered === paN()) paStep(1, 0, 'next after complete');
    else toast(`${toGoText(c.covered / paN())} of this patch is not annotated yet`, true);
    return true;
  }
  if (k === 'Backspace' || k === 'Delete') {
    if (polygon) { Pa.drag.pop(); if (!Pa.drag.length) Pa.drag = null; edDraw(); }
    return true;
  }
  if (k === 'z') { paUndoRedo(e.shiftKey ? 'redo' : 'undo'); return true; }
  if (k === '[' || k === ']') {
    Pa.brush = Math.max(4, Math.min(256, Math.round((Pa.brush * (k === ']' ? 1.25 : 0.8)) / 2) * 2));
    const s = $('#paSize');
    if (s) s.value = Pa.brush;
    paSizeText(); edDraw();
    return true;
  }
  if (k === 'g') { paSelectRest(); return true; }
  if (k === 'h') {
    Pa.hide = !Pa.hide;
    const h = $('#paHide');
    if (h) h.checked = !Pa.hide;
    paOp('hide', { hidden: Pa.hide });
    edDraw();
    return true;
  }
  const digit = /^Digit(\d)$/.exec(e.code || ''), i = PA_LABEL_KEYS.indexOf(digit ? digit[1] : k);   // Shift turns 1 into ! on the key itself
  if (i >= 0 && Pa.labels[i]) { paLabelClick(Pa.labels[i].key, e.shiftKey); return true; }
  const t = Object.keys(PA_TOOLS).find((id) => PA_TOOLS[id].key.toLowerCase() === k);
  if (t) { setPaTool(t); return true; }
  return true;
}
function patchEscape() {   // one step back: what is being drawn, the region, the tool, then the patch
  if (Pa.stroke) { endStroke(); return; }
  if (Pa.drag) { Pa.drag = null; edDraw(); return; }
  if (Pa.draft) {
    if (!Pa.draft.labels.size) { pushUndo(); Pa.draft = null; paMeasure(); paOp('discard'); paMeter(); paRender(); edDraw(); return; }
    paCloseRegion();
    return;
  }
  if (Pa.tool !== 'move') { setPaTool('move'); return; }
  if (Pa.cur) patchFlush('closed');
}

/* ------------------------------------------------------------ export: whose work first, then what */
function exportMenu(level) {
  const m = $('#exportMenu');
  if (!level) {
    const who = reviewerInput.value.trim();
    m.innerHTML = `<button type="button" class="menu-item" role="menuitem" data-exp="mine"><span class="spacer">Individual review${who ? ` <span class="muted">(${esc(who)})</span>` : ''}</span><span class="muted">›</span></button>
      <button type="button" class="menu-item" role="menuitem" data-exp="all"><span class="spacer">All reviews</span><span class="muted">›</span></button>`;
    return;
  }
  const q = level === 'mine' ? `?reviewer=${encodeURIComponent(Pa.owner)}` : '';
  m.innerHTML = `<button type="button" class="menu-item exp-back" data-exp="" title="Back"><span class="muted">‹</span><span class="spacer">${level === 'mine' ? `Individual review <span class="muted">(${esc(Pa.owner)})</span>` : 'All reviews'}</span></button>
    <div class="tool-sep"></div>
    <a href="api/export/patch_masks.zip${q}" download role="menuitem"><b>Complete patches</b> <span class="muted">ZIP</span><span class="exp-note">Training masks of every fully annotated patch</span></a>
    <a href="api/export/patch_activity.csv${q}" download role="menuitem"><b>Operation log</b> <span class="muted">CSV</span><span class="exp-note">Patch pick order, annotation order in each patch, stay time on each patch</span></a>`;
}
$('#exportMenu').addEventListener('click', (e) => {
  const b = e.target.closest('[data-exp]');
  if (!b) return;   // a download: the menu closes as usual
  e.stopPropagation();
  const level = b.dataset.exp;
  if (level === 'mine' && !patchWho()) { $('#exportMenu').classList.add('hidden'); return; }
  exportMenu(level);
});
$('#exportBtn').addEventListener('click', () => exportMenu(''));

/* ------------------------------------------------------------ start */
setPaTool('move');
if (modeIs('patches')) paLog('mode', { on: true });
