/* Cluster review: label unsupervised token clusters from exemplar galleries. Loaded after app.js. */
'use strict';

const C = { data: null, sel: null, form: null, extraSeed: 1, marker: null, pick: new Set() };
const LABEL_KEYS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', '[', ']'];
const CLAB = {
  ss_spindle: 'SS spindle cell', ss_glandular: 'SS glandular', ss_corded: 'SS corded / solid epithelial',
  ss_round_cell: 'SS round cell (PD)', hemorrhage: 'Hemorrhage', vessel: 'Vessel', normal_tissue: 'Other normal tissue',
  tumor: 'Tumor', stroma: 'Stroma', necrosis: 'Necrosis', inflammation: 'Inflammation', benign_epithelium: 'Benign epithelium',
  fat: 'Fat', muscle: 'Muscle', blood: 'Blood / hemorrhage', artifact: 'Artifact', mixed: 'Mixed', other: 'Other',
  spindle: 'Spindle', glandular: 'Glandular', round_cell: 'Round cell', clean: 'Clean', mostly: 'Mostly', mixed_p: 'Mixed',
};
const clab = (k) => CLAB[k] ?? k ?? '';
const isSeg = () => C.data && C.data.clustering.kind === 'seg';
const cname = (k) => (isSeg() ? 'S' : 'C') + k;

/* ------------------------------------------------------------ mode switching */
function setMode(m) {
  MODE = m;
  $('#clusterMain').classList.toggle('hidden', m !== 'clusters');
  $('#roiMain').classList.toggle('hidden', m !== 'rois');
  $('#progress').classList.toggle('hidden', m !== 'rois');
  $('#slideTitle').classList.toggle('hidden', m !== 'rois');
  document.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('on', b.dataset.mode === m));
  try { localStorage.setItem('hss_mode', m); } catch (e) { /* ignore */ }
  if (m === 'rois' && typeof S !== 'undefined' && S.openedHidden) {   // a slide opened while its tab was hidden has no size yet
    requestAnimationFrame(() => {
      if (!S.openedHidden) return;   // another slide was opened in the meantime
      S.openedHidden = false;
      if (viewer.forceResize) viewer.forceResize();
      viewer.viewport.goHome(true);
    });
  }
}
document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));

/* ------------------------------------------------------------ data */
async function loadClusters() {
  try { C.data = await api('api/clusters'); } catch (e) { return false; }
  renderCList();
  addHeatOptions();
  if (C.sel == null) {
    const first = C.data.clusters.find((c) => !c.review) || C.data.clusters[0];
    selectCluster(first.cluster);
  } else selectCluster(C.sel, true);
  return true;
}
const cl = (k) => C.data.clusters.find((c) => c.cluster === k);

function addHeatOptions() {
  let g = document.querySelector('#heatSel optgroup[data-g="clusters"]');
  if (!g) {
    g = document.createElement('optgroup');
    g.label = `Token clusters (${C.data.clustering.token_um} µm)`; g.dataset.g = 'clusters';
    g.innerHTML = '<option value="labels@clusters">Reviewed labels</option><option value="all@clusters">All clusters</option>';
    $('#heatSel').appendChild(g);
  }
}

/* ------------------------------------------------------------ list */
function warnings(c) {
  const w = [];
  if (c.max_slide_share > 0.4) w.push(`${Math.round(c.max_slide_share * 100)}% of its tokens come from one slide`);
  if (c.cohort_enrichment > 1.8) {
    const fr = Object.entries(c).filter(([k]) => k.startsWith('frac_') && !['frac_tokens'].includes(k)).sort((a, b) => b[1] - a[1])[0];
    w.push(`enriched for ${fr[0].replace('frac_', '')} (${Math.round(fr[1] * 100)}% of tokens)`);
  }
  return w;
}
function renderCList() {
  const cs = C.data.clusters, done = cs.filter((c) => c.review);
  const cover = done.reduce((a, c) => a + c.frac_tokens, 0);
  $('#cProgress').textContent = `${done.length}/${cs.length} labelled · ${Math.round(cover * 100)}% of tissue`;
  const m = C.data.clustering;
  $('#cTitle').textContent = isSeg() ? 'Segment classes' : 'Token clusters';
  $('#cMeta').innerHTML = isSeg()
    ? `Self-supervised segmentation: ${m.k} segment classes, ${m.area_mm2} mm² of tissue, ${m.n_slides} slides (tokens ${m.token_um} µm, map ${m.out_um} µm/px)<div id="subtypeLine"></div>`
    : `${m.n_tokens.toLocaleString()} tokens of ${m.token_um} µm from ${m.n_slides} slides`;
  if (isSeg()) loadSubtypes();
  $('#cList').innerHTML = cs.map((c) => {
    const r = c.review, w = warnings(c), into = c.merged_into != null ? cl(c.merged_into) : null;
    return `<li data-k="${c.cluster}" class="${c.cluster === C.sel ? 'active' : ''}${into ? ' merged' : ''}">
      <div class="sl-row">${isSeg() ? `<input type="checkbox" class="cpick" data-pick="${c.cluster}" title="Select to merge" ${C.pick.has(c.cluster) ? 'checked' : ''}>` : ''}<span class="swatch" style="background:${esc(into ? into.color : c.color)}"></span>
        <span class="sl-name">${cname(c.cluster)}</span>
        ${into ? `<span class="mchip" title="Merged on all slides">in ${cname(c.merged_into)}</span>` : ''}
        ${c.group ? `<span class="mchip" title="Merged on all slides">+ ${c.group.filter((k) => k !== c.cluster).map(cname).join(', ')} <button class="linkbtn" data-unmerge="${c.cluster}">unmerge</button></span>` : ''}
        ${w.length ? `<span class="badge second" title="${esc(w.join('; '))}">batch?</span>` : ''}
        ${r ? `<span class="lchip" style="border-color:${esc(C.data.label_colors[r.label])}">${esc(clab(r.label))}</span>` : '<span class="muted small">unlabelled</span>'}</div>
      <div class="sl-meta">${isSeg() ? c.area_mm2.toFixed(1) + ' mm² · ' : ''}${(c.frac_tokens * 100).toFixed(1)}% · ${c.n_slides} slides · ${c.n_patients} pts · hint ${esc(clab(c.conch_hint))}${isSeg() && c.conch_hint === 'tumor' ? ' / ' + esc(clab(c.conch_pattern_hint)) : ''}</div></li>`;
  }).join('');
  $('#cMergeBar').classList.toggle('hidden', !isSeg());
  $('#cMergeBar').innerHTML = `<span class="small muted">${C.pick.size ? `${C.pick.size} selected` : 'Tick classes to merge'}</span>
    <button id="cMergeBtn" ${C.pick.size < 2 ? 'disabled' : ''} title="Merge these classes into one on every slide of this run">Merge on all slides</button>`;
  $('#cMergeBtn').addEventListener('click', mergeClasses);
}
async function loadSubtypes() {
  try {
    const r = await api('api/seg/subtypes');
    const v = Object.values(r.slides).filter((s) => s.tumor_mm2 > 0);
    const n = (re) => v.filter((s) => re.test(s.call)).length;
    $('#subtypeLine').innerHTML = v.length
      ? `Derived from labels: <b>${n(/^biphasic/)}</b> biphasic, <b>${n(/^monophasic/)}</b> monophasic, <b>${n(/poorly/)}</b> with ≥${Math.round(r.pd_min_round_cell * 100)}% round cell (of ${Object.keys(r.slides).length} slides; biphasic = epithelial ≥${Math.round(r.biphasic_min_epithelial * 100)}% of tumour)`
      : 'Slide subtypes appear once SS components are labelled.';
  } catch (e) { /* ignore */ }
}
$('#cList').addEventListener('click', (e) => {
  const un = e.target.closest('[data-unmerge]');
  if (un) { unmergeClass(+un.dataset.unmerge); return; }
  if (e.target.closest('.cpick')) return;
  const li = e.target.closest('li');
  if (li) selectCluster(+li.dataset.k);
});
$('#cList').addEventListener('change', (e) => {
  const el = e.target.closest('.cpick');
  if (!el) return;
  if (el.checked) C.pick.add(+el.dataset.pick); else C.pick.delete(+el.dataset.pick);
  renderCList();
});
async function mergeClasses() {
  if (C.pick.size < 2) return;
  const ks = [...C.pick].flatMap((k) => { const c = cl(k); return c.group || [k]; });
  const names = [...C.pick].map(cname).join(' + ');
  if (await sendMerge('*', ks, 'merge')) { C.pick = new Set(); toast(`${names} merged on all slides`); }
}
async function unmergeClass(k) {
  const c = cl(k);
  if (c && c.group && await sendMerge('*', c.group, 'unmerge')) toast(`${cname(k)} unmerged on all slides`);
}

/* ------------------------------------------------------------ gallery */
function tileHtml(t, kind) {
  const src = t.name ? `api/clusters/ex/${encodeURIComponent(t.name)}.jpg`
    : `api/token_crop.jpg?sid=${encodeURIComponent(t.slide_id)}&x0=${t.x0}&y0=${t.y0}&size=${t.size}`;
  const sl = C.slidesById[t.slide_id];
  return `<figure class="tile" data-sid="${esc(t.slide_id)}" data-x="${t.x0}" data-y="${t.y0}" data-s="${t.size}" title="Open on the slide">
    <img loading="lazy" src="${src}" alt=""><figcaption><span>${esc(sl ? sl.patient : t.slide_id)}</span><span class="muted">${kind}</span></figcaption></figure>`;
}
function selectCluster(k, keepForm) {
  C.sel = k;
  const c = cl(k);
  document.querySelectorAll('#cList li').forEach((li) => li.classList.toggle('active', +li.dataset.k === k));
  const act = document.querySelector(`#cList li[data-k="${k}"]`);
  if (act) act.scrollIntoView({ block: 'nearest' });
  const coh = Object.entries(c).filter(([n]) => n.startsWith('frac_') && n !== 'frac_tokens');
  const COL = { LUND: '#e00000', MSK: '#ffd400', Utah: '#00b050' };
  const w = warnings(c);
  $('#cHead').innerHTML = `
    <div class="ch-title"><span class="swatch big" style="background:${esc(c.color)}"></span><b>${isSeg() ? 'Segment class' : 'Cluster'} ${cname(k)}</b>
      <span class="muted">${isSeg() ? c.area_mm2.toFixed(1) + ' mm²' : c.n_tokens.toLocaleString() + ' tokens'} (${(c.frac_tokens * 100).toFixed(1)}% of tissue) · ${c.n_slides} slides · ${c.n_patients} patients</span></div>
    <div class="ch-row">
      <div class="cohort-bar" title="Cohort make-up of this cluster">${coh.map(([n, v]) => `<i style="width:${v * 100}%;background:${COL[n.replace('frac_', '')] || '#888'}"></i>`).join('')}</div>
      <span class="muted">${coh.map(([n, v]) => `${n.replace('frac_', '')} ${Math.round(v * 100)}%`).join(' · ')}</span>
    </div>
    <div class="ch-row muted">Patch-level CONCH call: <b class="hint">${esc(clab(c.conch_hint))} ${Math.round(c.conch_hint_frac * 100)}%</b>${c.conch_hint_2 ? `, then ${esc(clab(c.conch_hint_2))}` : ''}${c.conch_pattern_hint ? `; pattern ${esc(clab(c.conch_pattern_hint))} ${Math.round(c.conch_pattern_frac * 100)}%` : ''} (hints, not labels)</div>
    ${w.length ? `<div class="ch-row warnline">Check for a batch effect: ${esc(w.join('; '))}.</div>` : ''}
    <div class="ch-row muted small">${isSeg() ? 'White outline = this segment class inside a 256 µm field (typical = largest regions, at most 3 per slide).' : `White box = the token (${C.data.clustering.token_um} µm); the image shows its surroundings.`} Click a tile to open it on the slide.</div>`;
  const ex = c.group ? c.group.flatMap((m) => cl(m).exemplars) : c.exemplars;
  $('#cGallery').innerHTML = ex.filter((t) => t.kind === 'central').map((t) => tileHtml(t, 'typical')).join('') +
    ex.filter((t) => t.kind === 'random').map((t) => tileHtml(t, 'random')).join('');
  C.extraSeed = 1;
  if (!keepForm || !C.form) {
    const r = c.review;
    C.form = r ? { label: r.label, pattern: r.pattern, purity: r.purity || 'clean', batch: !!r.batch_artifact, comment: '' }
      : { label: null, pattern: null, purity: 'clean', batch: false, comment: '' };
  }
  renderCPanel();
}
$('#cMore').addEventListener('click', moreTiles);
async function moreTiles() {
  if (C.sel == null) return;
  const extra = await api(`api/clusters/${C.sel}/sample?n=24&seed=${C.extraSeed++}`);
  $('#cGallery').insertAdjacentHTML('beforeend', extra.map((t) => tileHtml(t, 'random')).join(''));
}
$('#cGallery').addEventListener('click', (e) => {
  const f = e.target.closest('.tile');
  if (f) openTokenOnSlide(f.dataset.sid, +f.dataset.x, +f.dataset.y, +f.dataset.s, C.sel);
});

async function openTokenOnSlide(sid, x0, y0, size, k) {
  setMode('rois');
  await openSlide(sid);
  let opt = document.querySelector('#heatSel option[data-sel="1"]');
  if (!opt) {
    opt = document.createElement('option'); opt.dataset.sel = '1';
    document.querySelector('#heatSel optgroup[data-g="clusters"]').appendChild(opt);
  }
  opt.value = `${k}@clusters`; opt.textContent = `Only ${isSeg() ? 'segment' : 'cluster'} ${cname(k)}`;
  S.heat = opt.value; $('#heatSel').value = S.heat; setHeat();
  if (C.marker) { viewer.removeOverlay(C.marker); C.marker = null; }
  if (!isSeg()) {
    C.marker = document.createElement('div'); C.marker.className = 'token-marker';
    viewer.addOverlay({ element: C.marker, location: imgRect({ x: x0, y: y0, w: size, h: size }) });
  }
  const m = isSeg() ? 0.75 : 6;
  viewer.viewport.fitBounds(imgRect({ x: x0 - size * m, y: y0 - size * m, w: size * (2 * m + 1), h: size * (2 * m + 1) }));
}

/* ------------------------------------------------------------ label panel */
function renderCPanel() {
  const c = cl(C.sel), f = C.form, r = c.review, isT = f.label === 'tumor';
  $('#cPanel').innerHTML = `
    <div class="sec" style="margin-top:0;border-top:0;padding-top:0"><h3>Name ${isSeg() ? 'segment class' : 'cluster'} ${cname(c.cluster)}</h3>
      ${r ? `<div class="decision"><b>${esc(clab(r.label))}</b>${r.pattern ? ' · ' + esc(clab(r.pattern)) : ''} · ${esc(r.purity)} ${r.batch_artifact ? '· batch-driven' : ''}<br>
        <span class="muted">${esc(r.reviewer)} · ${esc((r.ts || '').replace('T', ' ').slice(0, 16))}</span>${r.comment ? `<div class="muted">“${esc(r.comment)}”</div>` : ''}</div>` : ''}
      <div class="lgrid">${C.data.labels.map((l, i) => `<button data-l="${l}" class="${f.label === l ? 'on' : ''}">
        <span><span class="swatch" style="background:${esc(C.data.label_colors[l])}"></span>${esc(clab(l))}</span><kbd>${LABEL_KEYS[i]}</kbd></button>`).join('')}</div>
      ${C.data.patterns.length ? `<div class="seg"><span class="seg-lab">Pattern <kbd>P</kbd></span>
        ${C.data.patterns.map((p) => `<button data-p="${p}" ${isT ? '' : 'disabled'} class="${isT && f.pattern === p ? 'on' : ''}">${esc(clab(p))}</button>`).join('')}</div>` : ''}
      <div class="seg"><span class="seg-lab">Purity <kbd>U</kbd></span>
        ${C.data.purity.map((p) => `<button data-u="${p}" class="${f.purity === p ? 'on' : ''}">${esc(clab(p === 'mixed' ? 'mixed_p' : p))}</button>`).join('')}</div>
      <label class="check"><input type="checkbox" id="cBatch" ${f.batch ? 'checked' : ''}> Driven by one slide or stain batch <kbd>B</kbd></label>
      <textarea id="cComment" placeholder="Comment (optional)">${esc(f.comment)}</textarea>
      <div class="save-row"><button id="cSave" class="primary" ${f.label ? '' : 'disabled'}>Save label <kbd>Enter</kbd></button>
        <span class="muted">Moves to the next unlabelled ${isSeg() ? 'class' : 'cluster'}</span></div>
    </div>
    <div class="sec"><h3>History</h3><div id="cHist" class="hist muted">Loading…</div></div>
    <div class="sec muted small">${isSeg() ? 'Every region of this class on every slide takes this label. Give several classes the same label to merge them. Monophasic / biphasic / poorly differentiated are derived per slide from the labelled SS component areas.' : 'Every patch token in the cluster inherits this label. Give several clusters the same label to merge them.'} Use “Mixed” when the gallery shows more than one tissue type, so it can be split later.</div>`;
  const P = $('#cPanel');
  P.querySelectorAll('[data-l]').forEach((b) => b.addEventListener('click', () => setForm('label', b.dataset.l)));
  P.querySelectorAll('[data-p]').forEach((b) => b.addEventListener('click', () => setForm('pattern', b.dataset.p)));
  P.querySelectorAll('[data-u]').forEach((b) => b.addEventListener('click', () => setForm('purity', b.dataset.u)));
  $('#cBatch').addEventListener('change', (e) => { C.form.batch = e.target.checked; });
  $('#cComment').addEventListener('input', (e) => { C.form.comment = e.target.value; });
  $('#cSave').addEventListener('click', saveCluster);
  api(`api/clusters/${c.cluster}/history`).then((h) => {
    if (C.sel !== c.cluster) return;
    $('#cHist').innerHTML = h.length ? h.map((x) => `<div>${esc((x.ts || '').replace('T', ' ').slice(0, 16))} · <b>${esc(x.reviewer)}</b> · ${esc(clab(x.label))}${x.pattern ? ', ' + esc(clab(x.pattern)) : ''} · ${esc(x.purity)}${x.batch_artifact ? ' · batch' : ''}</div>`).join('') : 'No labels yet.';
  });
}
function setForm(k, v) {
  C.form[k] = v;
  if (k === 'label' && v === 'tumor' && !C.form.pattern) C.form.pattern = 'spindle';
  renderCPanel();
}
async function saveCluster() {
  const who = reviewer();
  if (!who || !C.form.label) return;
  const f = C.form;
  try {
    await post(`api/clusters/${C.sel}/review`, { reviewer: who, label: f.label, pattern: f.label === 'tumor' ? f.pattern : null,
      purity: f.purity, batch_artifact: f.batch, comment: f.comment });
  } catch (e) { toast('Not saved: ' + e.message, true); return; }
  toast(`${cname(C.sel)}: ${clab(f.label)} saved`);
  if (isSeg()) loadSubtypes();
  const cur = C.sel;
  C.data = await api('api/clusters');
  renderCList();
  const cs = C.data.clusters, i = cs.findIndex((c) => c.cluster === cur);
  const next = cs.slice(i + 1).concat(cs.slice(0, i + 1)).find((c) => !c.review);
  selectCluster(next ? next.cluster : cur);
}

/* ------------------------------------------------------------ keyboard (clusters mode only) */
document.addEventListener('keydown', (e) => {
  if (MODE !== 'clusters' || !C.data) return;
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
  if (typing || e.metaKey || e.ctrlKey || e.altKey || e.key === 'Escape') return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  const cs = C.data.clusters, i = cs.findIndex((c) => c.cluster === C.sel);
  const cyc = (arr, v) => arr[(arr.indexOf(v) + 1) % arr.length];
  const acts = {
    ArrowDown: () => selectCluster(cs[Math.min(i + 1, cs.length - 1)].cluster),
    ArrowUp: () => selectCluster(cs[Math.max(i - 1, 0)].cluster),
    p: () => C.data.patterns.length && C.form.label === 'tumor' && setForm('pattern', cyc(C.data.patterns, C.form.pattern)),
    u: () => setForm('purity', cyc(C.data.purity, C.form.purity)),
    b: () => setForm('batch', !C.form.batch),
    m: moreTiles, Enter: saveCluster, '?': () => $('#help').classList.remove('hidden'),
  };
  const li = LABEL_KEYS.indexOf(k);
  if (li >= 0 && li < C.data.labels.length) acts[k] = () => setForm('label', C.data.labels[li]);
  if (acts[k]) { e.preventDefault(); e.stopPropagation(); acts[k](); }
}, true);

/* ------------------------------------------------------------ start */
(async () => {
  C.slidesById = {};
  try { (await api('api/slides')).forEach((s) => { C.slidesById[s.slide_id] = s; }); } catch (e) { /* ignore */ }
  const ok = await loadClusters();
  refreshLegend();
  let saved = null;
  try { saved = localStorage.getItem('hss_mode'); } catch (e) { /* ignore */ }
  setMode(ok ? (saved || 'clusters') : 'rois');
  if (!ok) document.querySelector('.tabs button[data-mode="clusters"]').disabled = true;
})();
