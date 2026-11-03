/* hSS review, slides view: whole-slide viewer with the self-supervised segmentation overlay, a model-predicted
   label per segment (SS pattern or stroma type) and the reviewer's segment labels and slide subtype. */
'use strict';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = (v) => (v == null ? '' : Math.round(v * 100) + '%');
const SLAB = {
  monophasic: 'Monophasic', biphasic: 'Biphasic', poorly_differentiated: 'Poorly differentiated', not_ss: 'Not SS / no tumour',
  cannot_assess: 'Cannot assess', glandular: 'Glandular', corded: 'Corded', stroma: 'Stroma, fibrous',
  stroma_hyalinized: 'Stroma, hyalinized', stroma_myxoid: 'Stroma, myxoid', calcification: 'Calcification',
  spindle: 'spindle', round_cell: 'round cell', fibrous: 'fibrous', hyalinized: 'hyalinized', myxoid: 'myxoid', calcified: 'calcified',
};
const slab = (k) => SLAB[k] ?? (typeof clab === 'function' ? clab(k) : k);
const SUBTYPE_KEY = { monophasic: 'M', biphasic: 'B', poorly_differentiated: 'P', not_ss: 'N', cannot_assess: 'Q' };
const PATTERN_KEY = { glandular: 'G', corded: 'C' };
const TAG = { model: 'model', class: 'from class', slide: 'confirmed', edited: 'edited' };
const OVERLAYS = ['all@clusters', 'slide@clusters', 'pred@clusters', 'none'];

let MODE = 'rois';   // 'clusters' | 'rois' (slides view), switched by clusters.js

const S = { slides: [], sid: null, info: null, heat: 'all@clusters', opacity: 0.45, rev: {} };
const SR = { data: null, form: null, solo: null, pick: new Set() };

/* ------------------------------------------------------------ api */
async function api(path, opt) {
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

function toast(msg, err) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = 'toast' + (err ? ' err' : '');
  clearTimeout(toast.h);
  toast.h = setTimeout(() => t.classList.add('hidden'), err ? 4000 : 1600);
}

/* ------------------------------------------------------------ reviewer */
const reviewerInput = $('#reviewer');
try { reviewerInput.value = localStorage.getItem('hss_reviewer') || ''; } catch (e) { /* storage blocked */ }
reviewerInput.addEventListener('input', () => {
  reviewerInput.classList.remove('need');
  try { localStorage.setItem('hss_reviewer', reviewerInput.value.trim()); } catch (e) { /* ignore */ }
});
function reviewer() {
  const v = reviewerInput.value.trim();
  if (!v) { reviewerInput.classList.add('need'); reviewerInput.focus(); toast('Enter your initials first', true); }
  return v;
}

/* ------------------------------------------------------------ viewer */
const viewer = OpenSeadragon({
  id: 'viewer', showNavigationControl: false, showNavigator: true, navigatorPosition: 'BOTTOM_RIGHT',
  navigatorBackground: '#0b0c0e', animationTime: 0.35, maxZoomPixelRatio: 2, visibilityRatio: 0.3,
  gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true }, preserveImageSizeOnResize: true,
});
const slideItem = () => viewer.world.getItemAt(0);
const imgRect = (r) => slideItem().imageToViewportRectangle(r.x, r.y, r.w, r.h);

function openSlide(sid) {
  return new Promise(async (resolve) => {
    if (S.sid === sid && slideItem()) return resolve();
    S.sid = sid; SR.data = null; SR.solo = null; SR.pick = new Set();
    S.openedHidden = !$('#viewer').clientWidth;
    if (S.heat !== 'none' && !OVERLAYS.includes(S.heat)) S.heat = 'all@clusters';
    $('#heatSel').value = S.heat;
    $('#srPanel').className = 'spanel empty'; $('#srPanel').textContent = 'Loading…';
    S.info = await api(`api/slides/${encodeURIComponent(sid)}/info`);
    renderSlideTitle(); renderSlideList();
    const li = document.querySelector('#slideList li.active');
    if (li) li.scrollIntoView({ block: 'nearest' });
    $('#exportSlide').href = `api/export/segments.geojson?slide=${encodeURIComponent(sid)}`;
    viewer.addOnceHandler('open', () => { setHeat(); resolve(); });
    viewer.open(`dzi/${encodeURIComponent(sid)}.dzi`);
    loadSegments(sid);
  });
}

let heatGen = 0;   // overlay requests finish out of order; only the latest may stay on the viewer
function setHeat() {
  const gen = ++heatGen;
  while (viewer.world.getItemCount() > 1) viewer.world.removeItem(viewer.world.getItemAt(1));
  const seg = S.info && S.info.seg;
  if (S.heat === 'none' || !seg || !slideItem()) { $('#legend').classList.add('hidden'); return; }
  const cls = S.heat.split('@')[0];
  const url = `api/slides/${encodeURIComponent(S.sid)}/clustermap/${cls}.png?t=${Date.now()}`;
  viewer.addTiledImage({
    tileSource: { type: 'image', url }, opacity: S.opacity,
    x: seg.x0 / S.info.width, y: seg.y0 / S.info.width, width: (seg.w * seg.px_l0) / S.info.width,
    success: (e) => { if (gen !== heatGen) { e.item.setOpacity(0); setTimeout(() => viewer.world.removeItem(e.item)); } },
  });
  refreshLegend();
}

function refreshLegend() {
  const lg = $('#legend');
  if (S.heat === 'none' || !S.info || !S.info.seg) { lg.classList.add('hidden'); return; }
  lg.innerHTML = clusterLegend(S.heat.split('@')[0]);
  lg.classList.toggle('hidden', !lg.innerHTML);
}

function clusterLegend(cls) {
  const sw = (c, t) => `<span><i style="background:${esc(c)}"></i>${esc(t)}</span>`;
  if (cls === 'slide' || cls === 'pred') {
    if (!SR.data) return '';
    const used = [...new Set(SR.data.segments.map((s) => (cls === 'pred' ? s.pred_label : s.label)).filter(Boolean))];
    return `<span>${cls === 'pred' ? 'Model-predicted' : 'Saved'} labels:</span>` +
      (used.length ? used.map((l) => sw(SR.data.label_colors[l] || '#000', slab(l))).join('') : '<span>none saved yet</span>');
  }
  if (typeof C === 'undefined' || !C.data) return '';
  const here = SR.data ? new Set(SR.data.segments.map((s) => s.segment)) : null;
  const cs = C.data.clusters.filter((c) => (cls === 'all' ? !here || here.has(c.cluster) : String(c.cluster) === cls));
  return '<span>Segments:</span>' + cs.map((c) => sw(c.color, cname(c.cluster))).join('');
}

function setOverlay(v) {
  S.heat = v;
  let opt = [...$('#heatSel').options].find((o) => o.value === v);
  if (!opt) {
    opt = document.querySelector('#heatSel option[data-sel="1"]') || document.createElement('option');
    opt.dataset.sel = '1';
    document.querySelector('#heatSel optgroup[data-g="clusters"]').appendChild(opt);
    opt.value = v; opt.textContent = `Only segment ${cname(+v.split('@')[0])}`;
  }
  $('#heatSel').value = v;
  setHeat();
}

/* ------------------------------------------------------------ slides */
async function loadSlides() {
  [S.slides, S.rev] = await Promise.all([api('api/slides'), api('api/slide_reviews')]);
  renderSlideList(); renderProgress();
}
function renderProgress() {
  const n = S.slides.filter((s) => S.rev[s.slide_id]).length;
  $('#progress').textContent = `${n} / ${S.slides.length} slides reviewed`;
}
function visibleSlides() {
  const f = $('#slideFilter').value, q = $('#slideSearch').value.trim().toLowerCase();
  return S.slides.filter((s) => (f === 'all' || (f === 'todo' && !S.rev[s.slide_id]) || (f === 'done' && S.rev[s.slide_id]) ||
    (f === 'qc' && s.qc_status !== 'ok')) && (!q || `${s.slide_id} ${s.patient}`.toLowerCase().includes(q)));
}
function renderSlideList() {
  $('#slideList').innerHTML = visibleSlides().map((s) => {
    const flags = Object.entries(s.qc_flags || {}).filter(([, v]) => v > 0.1).map(([k]) => k.replace('_', ' '));
    const r = S.rev[s.slide_id];
    return `<li data-sid="${esc(s.slide_id)}" class="${s.slide_id === S.sid ? 'active' : ''}">
      <div class="sl-row"><span class="qc-dot qc-${esc(s.qc_status)}" title="QC ${esc(s.qc_status)}${flags.length ? ': ' + esc(flags.join(', ')) : ''}"></span>
        <span class="sl-name">${esc(s.patient)}</span>
        ${r ? `<span class="lchip done" title="${esc(r.reviewer)} · ${esc((r.ts || '').slice(0, 10))}">${esc(slab(r.subtype))}${r.patterns ? ' · ' + esc(r.patterns.split(',').map(slab).join(', ')) : ''}</span>` : ''}</div>
      <div class="sl-meta">${esc(s.cohort)} · ${esc(s.slide_id.replace(/^(MSK|LUND|Utah)_/, ''))}${s.histology ? ' · ' + esc(s.histology) : ''}</div></li>`;
  }).join('');
}
$('#slideList').addEventListener('click', (e) => {
  const li = e.target.closest('li');
  if (li) openSlide(li.dataset.sid);
});
$('#slideFilter').addEventListener('change', renderSlideList);
$('#slideSearch').addEventListener('input', renderSlideList);

function stepSlide(dir) {
  const v = visibleSlides(), i = v.findIndex((s) => s.slide_id === S.sid);
  const s = v[Math.min(Math.max(i + dir, 0), v.length - 1)];
  if (s) openSlide(s.slide_id);
}

function renderSlideTitle() {
  const i = S.info;
  $('#slideTitle').innerHTML = `<b>${esc(i.patient)}</b> · ${esc(i.slide_id)}` +
    (i.histology ? ` · Histology ${esc(i.histology)}` : '') + (i.fusion ? ` · ${esc(i.fusion)}` : '') +
    ` · ${(i.mpp * 1).toFixed(3)} µm/px`;
}

/* ------------------------------------------------------------ slide review panel */
async function loadSegments(sid) {
  let d;
  try { d = await api(`api/slides/${encodeURIComponent(sid)}/segments`); } catch (e) {
    if (S.sid !== sid) return;
    $('#srPanel').className = 'spanel empty'; $('#srPanel').textContent = 'No segmentation for this slide.';
    return;
  }
  if (S.sid !== sid) return;
  SR.data = d;
  const rv = d.review;
  SR.form = {
    subtype: rv ? rv.subtype : null, comment: '',
    patterns: new Set(rv && rv.patterns ? rv.patterns.split(',').filter(Boolean) : []),
    labels: Object.fromEntries(d.segments.map((s) => [s.segment, s.label || s.pred_label])),
  };
  renderSR(); refreshLegend();
}

function tagOf(s) {
  const f = SR.form.labels[s.segment];
  if (f !== (s.label || s.pred_label)) return 'edited';
  return s.slide_label ? 'slide' : s.class_label ? 'class' : 'model';
}

function scoreLine(rel) {
  if (!rel) return '';
  return Object.entries(rel).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${esc(slab(k).toLowerCase())} ${pct(v)}`).join(' · ');
}

function renderSR() {
  const d = SR.data, f = SR.form, p = $('#srPanel'), pr = d.predicted, dv = d.derived;
  p.className = 'spanel';
  const opts = (sel) => d.labels.map((l) => `<option value="${l}" ${l === sel ? 'selected' : ''}>${esc(slab(l))}</option>`).join('');
  const segs = [...d.segments].sort((a, b) => b.area_mm2 - a.area_mm2);
  p.innerHTML = `
    <div class="sec first"><h3>Synovial sarcoma subtype</h3>
      ${S.info && S.info.histology ? `<div class="small muted">Clinical histology: <b class="txt">${esc(S.info.histology)}</b></div>` : ''}
      <div class="sugg">Model suggests <b>${esc(pr.call)}</b>${pr.tumor_mm2 > 0 ? ` <span class="muted">(glandular ${pct(pr.glandular)}, corded ${pct(pr.corded)}, round cell ${pct(pr.round_cell)} of SS area)</span>` : ''}</div>
      ${dv.tumor_mm2 > 0 && segs.some((s) => s.label) ? `<div class="small muted">From saved segment labels: <b class="txt">${esc(dv.call)}</b></div>` : ''}
      <div class="sgrid">${d.subtypes.map((t) => `<button data-st="${t}" class="${f.subtype === t ? 'on' : ''}">${esc(slab(t))} <kbd>${SUBTYPE_KEY[t]}</kbd></button>`).join('')}</div>
      <div class="seg"><span class="seg-lab">Patterns</span>
        ${d.patterns.map((t) => `<button data-pt="${t}" class="${f.patterns.has(t) ? 'on' : ''}">${esc(slab(t))} <kbd>${PATTERN_KEY[t]}</kbd></button>`).join('')}</div>
    </div>
    <div class="sec"><h3>Segments on this slide (${segs.length})</h3>
      <div class="small muted note">Model label: CONCH zero-shot scores averaged over the segment; SS pattern and stroma type
        are called relative to the cohort baseline. Model stroma: ${pct(pr.stroma_frac)} of tissue.${d.patterns_ready ? '' : ' Corded and stroma types are still being computed.'}</div>
      <div class="mergebar"><span class="small muted">${SR.pick.size ? `${SR.pick.size} selected` : 'Tick segments to merge'}</span>
        <button data-merge="slide" ${SR.pick.size < 2 ? 'disabled' : ''} title="Merge into one segment on this slide">Merge on this slide</button>
        <button data-merge="all" ${SR.pick.size < 2 ? 'disabled' : ''} title="Merge these classes on every slide of this run">Merge on all slides</button></div>
      ${segs.map((s, i) => `
        <div class="srow ${SR.solo === s.segment ? 'solo' : ''}" data-k="${s.segment}">
          <button class="sw" style="background:${esc(s.color)}" title="Show only this segment (${i + 1}); again for all"></button>
          <div class="sinfo"><label class="pick" title="Select to merge"><input type="checkbox" data-pick="${s.segment}" ${SR.pick.has(s.segment) ? 'checked' : ''}><b>${cname(s.segment)}</b></label>${s.merged.length ? ` <span class="mchip" title="Merged by a reviewer ${s.merge_scope === 'slide' ? 'on this slide' : 'on all slides'}">+ ${s.merged.map(cname).join(', ')}${s.merge_scope === 'slide' ? '' : ' · all slides'} <button class="linkbtn" data-unmerge="${s.segment}">unmerge</button></span>` : ''}${autoMerged(s).length ? ` <span class="muted small" title="Classes the model could not separate on this slide">(auto ${autoMerged(s).map(cname).join(', ')})</span>` : ''} <span class="muted">${pct(s.frac)} · ${s.area_mm2.toFixed(2)} mm²</span>
            <div class="small muted">Model: <span class="txt">${esc(slab(s.pred_label))}</span> ${pct(s.pred_conf)}${s.pattern || s.stroma ? ` <span title="Share of cohort-relative score">(${scoreLine(s.pattern || s.stroma)})</span>` : ''}</div></div>
          <button class="ghost zoom" title="Zoom to the largest piece" ${s.bbox ? '' : 'disabled'}>⌖</button>
          <select data-k="${s.segment}" aria-label="Label for ${cname(s.segment)}">${opts(f.labels[s.segment])}</select>
          <span class="tag t-${tagOf(s)}">${TAG[tagOf(s)]}</span>
        </div>`).join('')}
    </div>
    <textarea id="srComment" placeholder="Comment (optional)">${esc(f.comment)}</textarea>
    <div class="save-row"><button class="primary" id="srSave">Save slide <kbd>Enter</kbd></button><span class="muted small">Then the next unreviewed slide</span></div>
    <div class="sec"><h3>History</h3><div class="hist">${d.history.length ? d.history.map((h) => `<div>${esc((h.ts || '').replace('T', ' ').slice(0, 16))} · <b>${esc(h.reviewer)}</b> · ${esc(slab(h.subtype))}${h.patterns ? ' · ' + esc(h.patterns.split(',').map(slab).join(', ')) : ''} <span class="muted">(model: ${esc(h.predicted_subtype)})</span>${h.comment ? `<div class="muted">“${esc(h.comment)}”</div>` : ''}</div>`).join('') : 'Not reviewed yet.'}</div></div>`;

  p.querySelectorAll('[data-st]').forEach((b) => b.addEventListener('click', () => setSubtype(b.dataset.st)));
  p.querySelectorAll('[data-pt]').forEach((b) => b.addEventListener('click', () => togglePattern(b.dataset.pt)));
  p.querySelectorAll('select[data-k]').forEach((el) => el.addEventListener('change', () => { f.labels[+el.dataset.k] = el.value; renderSR(); }));
  p.querySelectorAll('.srow .sw').forEach((b) => b.addEventListener('click', () => solo(+b.closest('.srow').dataset.k)));
  p.querySelectorAll('.srow .zoom').forEach((b) => b.addEventListener('click', () => zoomSeg(+b.closest('.srow').dataset.k)));
  p.querySelectorAll('[data-pick]').forEach((el) => el.addEventListener('change', () => {
    const k = +el.dataset.pick;
    if (el.checked) SR.pick.add(k); else SR.pick.delete(k);
    renderSR();
  }));
  p.querySelectorAll('[data-merge]').forEach((b) => b.addEventListener('click', () => mergeSegments(b.dataset.merge)));
  p.querySelectorAll('[data-unmerge]').forEach((b) => b.addEventListener('click', () => unmergeSegment(+b.dataset.unmerge)));
  $('#srComment').addEventListener('input', (e) => { f.comment = e.target.value; });
  $('#srSave').addEventListener('click', saveSR);
}

/* ------------------------------------------------------------ merging */
function errDetail(e) {   // "409 {"detail":"..."}" -> the detail text
  const m = /^\d+ (.*)$/s.exec(e.message || '');
  try { return JSON.parse(m[1]).detail || m[1]; } catch (x) { return m ? m[1] : String(e); }
}
function autoMerged(s) { return (s.members || []).filter((m) => !s.classes.includes(m)); }

async function sendMerge(scope, classes, action) {
  const who = reviewer();
  if (!who) return false;
  try {
    await post('api/merges', { reviewer: who, scope, classes, action });
  } catch (e) { toast((action === 'merge' ? 'Could not merge: ' : 'Could not unmerge: ') + errDetail(e), true); return false; }
  SR.pick = new Set(); SR.solo = null;
  if (S.heat !== 'none' && !OVERLAYS.includes(S.heat)) S.heat = 'all@clusters';
  if (S.sid) { await loadSegments(S.sid); setOverlay(S.heat); }
  if (typeof loadClusters === 'function' && typeof C !== 'undefined' && C.data) loadClusters();
  return true;
}
async function mergeSegments(where) {
  if (!SR.data || SR.pick.size < 2) return;
  const segs = SR.data.segments.filter((s) => SR.pick.has(s.segment));
  const names = segs.map((s) => cname(s.segment)).join(' + ');
  if (await sendMerge(where === 'slide' ? S.sid : '*', segs.flatMap((s) => s.classes), 'merge')) {
    toast(`${names} merged ${where === 'slide' ? 'on this slide' : 'on all slides'}`);
  }
}
async function unmergeSegment(k) {
  const s = SR.data && SR.data.segments.find((x) => x.segment === k);
  if (!s) return;
  if (await sendMerge(s.merge_scope === 'slide' ? S.sid : '*', s.classes, 'unmerge')) {
    toast(`${cname(k)} unmerged ${s.merge_scope === 'slide' ? 'on this slide' : 'on all slides'}`);
  }
}

function setSubtype(t) { if (SR.form) { SR.form.subtype = t; renderSR(); } }
function togglePattern(t) {
  if (!SR.form) return;
  if (SR.form.patterns.has(t)) SR.form.patterns.delete(t); else SR.form.patterns.add(t);
  renderSR();
}
function solo(k) {
  SR.solo = SR.solo === k ? null : k;
  setOverlay(SR.solo == null ? 'all@clusters' : `${k}@clusters`);
  renderSR();
}
function zoomSeg(k) {
  const s = SR.data.segments.find((x) => x.segment === k);
  if (!s || !s.bbox) return;
  const b = s.bbox, m = 0.3;
  viewer.viewport.fitBounds(imgRect({ x: b.x - b.w * m, y: b.y - b.h * m, w: b.w * (1 + 2 * m), h: b.h * (1 + 2 * m) }));
}

async function saveSR() {
  const who = reviewer();
  if (!who || !SR.data) return;
  if (!SR.form.subtype) { toast('Choose the subtype first (M, B, P, N or Q)', true); return; }
  const sid = S.sid;
  try {
    await post(`api/slides/${encodeURIComponent(sid)}/slide_review`, {
      reviewer: who, subtype: SR.form.subtype, patterns: [...SR.form.patterns], segments: SR.form.labels, comment: SR.form.comment,
    });
  } catch (err) { toast('Not saved: ' + err.message, true); return; }
  toast(`${slab(SR.form.subtype)} · saved`);
  S.rev = await api('api/slide_reviews');
  renderProgress(); renderSlideList();
  const v = visibleSlides(), i = v.findIndex((s) => s.slide_id === sid);
  const next = v.slice(i + 1).concat(v.slice(0, Math.max(i, 0))).find((s) => !S.rev[s.slide_id]);
  if (next) openSlide(next.slide_id);
  else { await loadSegments(sid); setHeat(); toast('Every slide in this list has a review'); }
}

/* ------------------------------------------------------------ toolbar */
$('#heatSel').addEventListener('change', (e) => { SR.solo = null; S.heat = e.target.value; setHeat(); if (SR.data) renderSR(); });
$('#heatOpacity').addEventListener('input', (e) => {
  S.opacity = +e.target.value;
  const it = viewer.world.getItemAt(1);
  if (it) it.setOpacity(S.opacity);
});
$('#fitBtn').addEventListener('click', () => viewer.viewport.goHome());
$('#helpBtn').addEventListener('click', () => $('#help').classList.remove('hidden'));
$('#helpClose').addEventListener('click', () => $('#help').classList.add('hidden'));
$('#exportBtn').addEventListener('click', (e) => { e.stopPropagation(); $('#exportMenu').classList.toggle('hidden'); });
document.addEventListener('click', () => $('#exportMenu').classList.add('hidden'));

/* ------------------------------------------------------------ keyboard */
document.addEventListener('keydown', (e) => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
  if (e.key === 'Escape') {
    $('#help').classList.add('hidden');
    document.querySelectorAll('.modal:not(#help)').forEach((m) => m.remove());
    if (typing) document.activeElement.blur();
    return;
  }
  if (MODE !== 'rois' || typing || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  const acts = {
    m: () => setSubtype('monophasic'), b: () => setSubtype('biphasic'), p: () => setSubtype('poorly_differentiated'),
    n: () => setSubtype('not_ss'), q: () => setSubtype('cannot_assess'),
    g: () => togglePattern('glandular'), c: () => togglePattern('corded'),
    Enter: saveSR, ArrowDown: () => stepSlide(1), ArrowUp: () => stepSlide(-1),
    f: () => viewer.viewport.goHome(),
    h: () => { SR.solo = null; setOverlay(OVERLAYS[(OVERLAYS.indexOf(S.heat) + 1) % OVERLAYS.length]); if (SR.data) renderSR(); },
    '?': () => $('#help').classList.remove('hidden'),
  };
  if (/^[1-9]$/.test(k) && SR.data) {
    const segs = [...SR.data.segments].sort((a, b) => b.area_mm2 - a.area_mm2);
    if (segs[+k - 1]) acts[k] = () => solo(segs[+k - 1].segment);
  }
  if (acts[k]) { e.preventDefault(); e.stopPropagation(); acts[k](); }
}, true);

/* ------------------------------------------------------------ segmentation run */
async function loadRuns() {
  const r = await api('api/clusterings'), el = $('#segRun');
  if (!r.runs.length) { el.classList.add('hidden'); return; }
  const enc = { conch15: 'CONCH v1.5', hoptimus0: 'H-optimus-0', hoptimus1: 'H-optimus-1', uni2h: 'UNI2-h', uni: 'UNI', virchow2: 'Virchow2' };
  el.innerHTML = r.runs.map((x) => `<option value="${esc(x.dir)}" ${x.dir === r.active ? 'selected' : ''} title="${esc(x.method)}">${esc(enc[x.encoder] || x.encoder)} · ${x.adaptive ? `K ≤ ${x.k}, per slide (median ${x.k_slide_median})` : `K = ${x.k}`}</option>`).join('');
  el.addEventListener('change', async () => {
    try { await post('api/clusterings/active', { dir: el.value }); location.reload(); } catch (e) { toast('Could not switch: ' + e.message, true); }
  });
}

/* ------------------------------------------------------------ start */
(async () => {
  await loadRuns();
  await loadSlides();
  const first = S.slides.find((s) => !S.rev[s.slide_id]) || S.slides[0];
  if (first) await openSlide(first.slide_id);
  if (!reviewerInput.value) reviewerInput.focus();
})().catch((err) => toast('Could not load: ' + err.message, true));
