# Pipeline and review details

Model proposes, pathologist decides, decisions retrain the model.

Review is slide by slide: self-supervised smooth segmentation (s1, s2) gives the segments, and the pathologist corrects their labels and calls the subtype on each slide.

```
H&E slide -> QC + tissue mask -> tissue probability map -> candidate ROIs (about 1 mm2)
   -> TITAN ROI embeddings -> ROI labels + confidence + OOD -> review UI -> decisions (audit log)
   -> a3_refine (linear probe on TITAN) -> new ROI table -> review UI ...
```

## Steps

| Step | What | Run |
|---|---|---|
| a1_qc | Patch QC on the 512 px / 20x grid: blur, pen ink, folds, weak stain, low tissue | `sbatch deploy/chpc/run_annot.slurm` (runs a1 + a2) |
| a2_tissue_rois | TITAN zero-shot tissue map (each patch scored from its 5 x 5 neighbourhood), 4 x 4 patch ROIs, ROI embeddings, zero-shot labels, triage (model `zs-v0`) | same job |
| a2b_conch_patch_map | CONCH v1 image + text encoder, zero-shot per 512 px patch (sharper map); adds conch_* columns to rois_v000 and map_agreement.csv | same job |
| c0_dense_tokens | CONCH v1.5 final-layer ViT tokens per patch (28 x 28, about 9 um), pooled to 14 x 14 (about 18 um), with token tissue fraction | `STEPS="c0_dense_tokens c1_token_clusters" sbatch deploy/chpc/run_annot.slurm` |
| c1_token_clusters | Tokens pooled to TOKEN_GRID (default 7 x 7 per patch = 37 um), tissue >= 50 %, L2 + PCA64 + mini-batch k-means (N_CLUSTERS, default 20); exemplars, cohort / slide make-up, CONCH hint, viewer overlay, UMAP figure | same job |
| s1_dense_features | Seamless dense features: CONCH v1.5 28 x 28 tokens (9 um) from 512 px / 20x windows at 50 % overlap, Hann-blended into one canvas per slide, PCA to 64 dims | `STEPS="s1_dense_features s2_segment" sbatch deploy/chpc/run_annot.slurm` (about 25 min) |
| s2_segment | k-means (N_SEG = 16) on pooled tokens, soft assignment (TAU 0.05), tissue-normalised Gaussian, upsampled to 4 um/px and refined with an H&E-guided filter, specks < 4000 um2 absorbed; label PNGs, GeoJSON polygons, areas, exemplars with outlines | same job |
| a3_refine | Linear probes on frozen TITAN embeddings from the latest decisions, patient-grouped CV report, next `rois_vNNN.csv` (model `lp-vNNN`) | `STEPS=a3_refine sbatch deploy/chpc/run_annot.slurm` |
| app | Review UI (FastAPI + OpenSeadragon) | `sbatch deploy/chpc/serve.slurm`, then tunnel |



## Labels (MVP)

- Tumor vs non-tumor
- Pattern, for tumor ROIs: spindle (monophasic), glandular (biphasic), round cell (poorly differentiated)
- Necrosis present / absent
- Artifact / insufficient tissue (from QC; reviewer can also mark it)

## Review policy

| State | Rule (zero-shot v0) | Rule (after refinement) | UI |
|---|---|---|---|
| prelabel | confidence rank > 2/3 | probability >= 0.9 | green dashed, confirm with A |
| review | 1/3 to 2/3 | 0.6 to 0.9 | amber dashed |
| needs_review | rank < 1/3, or OOD (bottom 5 % similarity to other slides) | < 0.6 or OOD | red dashed, first in the filmstrip |
| blocked | > 50 % of the ROI's patches fail QC, or slide QC fail | same, or predicted artifact | grey, never auto-labelled |

Zero-shot probabilities are close to uniform, so v0 confidence is the rank of the top-1 / top-2 margin within the cohort. It is a review order, not a calibrated probability.

## Review UI

- Left: slides with QC dot, unreviewed needs-review count, second-review count, progress bar.
- Centre: whole-slide viewer, ROI outlines coloured by state (solid blue once decided), heatmap overlay from CONCH (per patch) or TITAN (5 x 5 neighbourhood): tumor, necrosis, inflammation, top tissue class, with opacity; draw a missed region.
- Right: prediction with confidence and pattern alternatives, Accept / Not tumor / Artifact / Second review, label editor, the most similar regions in TITAN space (reviewed ones once at least three exist), and the decision history.
- Bottom: filmstrip, unreviewed first, then needs_review > blocked > review > prelabel, most uncertain first. Can show this slide or all slides.
- Keys: A, N, X, S, T, 1/2/3, E, Enter, left/right arrows, Z, F, H, O, D, ?.
- Audit: `reviews` in `results/annot/annotations.sqlite` is append-only (reviewer, UTC time, decision, labels, comment, model version, prediction snapshot, ROI geometry). Export from the UI: GeoJSON (QuPath-compatible, level-0 pixels) and the full audit CSV.

## Access

`serve.slurm` prints the node. From the Mac that holds the CHPC ControlMaster connection:

```
ssh -O forward -L 8765:<node>:8765 chpc-login
open "http://localhost:8765/?token=<token>"
```

The token is in `~/.hss_annot_token` (mode 600); the first visit sets a cookie. For a pathologist without CHPC ssh, run the same app through Open OnDemand or another host that can read the slides.

## Caveats

- The tissue map and v0 labels are zero-shot TITAN on small regions (5 x 5 patches = 1.3 mm for the map, 4 x 4 patches for ROIs), well below the 8192 px regions TITAN was trained on. Treat them as a starting triage that pathologist decisions replace.
- No supervised segmentation model exists for SS here. Once enough ROIs are reviewed, the decisions are the training set for one (patch-level labels can be taken from ROI decisions).
- ROIs are a fixed 4 x 4 patch grid; pathologists can add rectangles for missed areas. These get an embedding at the next refinement.

## Class-level labels (no longer in the UI)

The clusters tab, which named each segment class once across all slides, was removed from the app. Its endpoints and the `cluster_reviews` table still exist. Labels already saved there keep applying for their run (tagged "from class" on a slide) until a reviewer overrides them per slide; the audit stays available under Export.

## Self-supervised segmentation

- Labels: SS spindle cell, SS glandular, SS corded / solid epithelial, SS round cell (poorly differentiated), stroma, necrosis, inflammation, hemorrhage, fat, muscle, vessel, other normal tissue, artifact, mixed.
- Slide-level subtype is derived, not segmented: biphasic if glandular + corded >= 5 % of labelled tumour area, otherwise monophasic; poorly differentiated flag if round cell >= 20 % (`BIPHASIC_MIN_EPI`, `PD_MIN_ROUND` in ann_common.py). Shown in the slide header and the segment panel.
- Export: `api/export/segments.geojson?slide=<id>` gives the polygons with the reviewed label as QuPath classification.

## Slide review

The app shows the slide with the smooth segmentation overlay only; ROI squares, the ROI filmstrip and the
256 um patch heatmaps were removed from the view (the ROI tables and endpoints still exist).

- Overlay: segments coloured per class, labels saved on this slide, or model-predicted labels. Click a segment
  swatch (or 1 to 9) to show only that segment; the target button zooms to its largest piece.
- Model label per segment: CONCH v1 zero-shot patch scores (a2b tissue classes, a2c SS patterns with corded, and
  stroma types fibrous / hyalinized / myxoid / calcified) averaged over the segment. Pattern and stroma type are
  called relative to the cohort baseline (tissue_map_ss_prior.json, floor 0.02), spindle / fibrous unless another
  class is 1.25 x higher. Model subtype uses the same area rule as the derived subtype.
- Reviewer: subtype (monophasic, biphasic, poorly differentiated, not SS, cannot assess), patterns (glandular,
  corded) and a label per segment. Segment labels saved here override the class-level label on that slide only.
  Tables slide_reviews and slide_seg_reviews, append-only; export: Slide reviews audit (CSV).

## Reviewing and editing a segmentation

Reviewers have a step-by-step guide in the app: the Tutorial link in the header opens `app/static/tutorial.html`.

- Review: click a region or press J / K to select a segment; A accepts its label and moves on; T comments. Saving a slide confirms only the checked segments. Unchecked ones stay model suggestions and are not written to `slide_seg_reviews`. Comments are stored at once in `segment_comments`.
- Tools on the viewer: Lines (move an outline), Polygon, Lasso, Brush, Erase, Undo, Redo. Tools add to a chosen segment or start a new one. A new segment exists on its slide only, with an id from the run's k upward, shown as N1, N2.
- The model's `labels/<slide>.png` is never changed. Each edit is a row in `seg_edits` (append-only, level-0 coordinates, undo and redo are rows too), and the current map is the model's map with the edits in force replayed (`app/segedit.py`). Overlay, areas, model labels, derived subtype and the GeoJSON export all read that map.
- Moving a line: area gained is taken from whatever was there, and area given up goes to the nearest other segment or to background, so neighbours stay in contact. Undo is per slide, not per reviewer.
- Model labels are recomputed over the edited shape from the same CONCH patch scores (512 px patches), so they are coarse for small shapes. Encoder features are not recomputed.

## Expert ROIs

The **Pick ROIs** tab lets an expert outline an area with a polygon, square or oval and label it; **My ROIs** shows that expert's ROIs as a gallery (`app/expertroi.py`, `app/static/rois.js`). Labels: glandular, spindle cells, poorly differentiated (round cells), myxoid, osteoid (bone forming), collagenous, and normal tissue (vessels, muscle, nerve, fat). A ROI can have several.

- Each reviewer has a workspace: the app lists and changes only the ROIs drawn under their initials (`LL` and `ll` are the same person), and the server refuses changes from anyone else. Experts therefore annotate independently. Initials are the only identity, so this keeps people apart but is not access control: anyone typing someone's initials opens that workspace.
- `/api/export/expert_rois.csv` without `?reviewer=` holds every reviewer's ROIs, to compare them. It is available to anyone with the app's link.
- Stored in `expert_roi_events` (append-only, level-0 coordinates). Each event holds the full label set and comment; the latest event of a ROI is its current state, and delete and restore are events too. An outline never changes after it is drawn. An oval is stored as a 72-point polygon.
- Separate from the older `manual_rois` / `reviews` tables and from the segmentation run, so switching runs does not affect ROIs, and slides without a segmentation can have them.
- Moving to another slide or tab saves an open ROI if it has a label, and otherwise drops it with a message. An unfinished polygon is dropped too.
- Other exports: `/api/export/expert_rois.geojson` (`?reviewer=`, `?slide=`), `/api/export/expert_rois_audit.csv`.
- The tab is kept per browser. A link can name it: `…/#rois` or `…/#my-rois`.

## Export menu

The menu has two levels: Individual review (the initials entered) or All reviews, then Complete patches (`patch_masks.zip`) or Operation log (`patch_activity.csv`). The other exports are no longer in the menu but work by URL: `slide_reviews.csv`, `segments.geojson?slide=`, `cluster_labels.csv`, `expert_rois.csv`, `expert_rois.geojson`, `expert_rois_audit.csv`, `patches.csv`, `patches_audit.csv` and `patch_visits.csv`, all under `/api/export/`. The My ROIs tab has its own export buttons.

## Patch annotation

The **Patches** tab is for model training data: a grid over each slide, annotated patch by patch and pixel by pixel (`app/patches.py`, `app/static/patches.js`).

- Grid: patches of 512 × 512 pixels at 20×. The slides are scanned at 40× (0.2525 µm/pixel), so `PATCH_SIZE` = 1024 level-0 pixels from the slide's top left corner, or 259 µm, with masks of `MASK_SIZE` = 512 at 0.505 µm/pixel. Only whole patches are in the grid; a narrower strip at the right and bottom edges is left out. Patches saved at one size are not listed at another.
- A patch holds regions drawn with brush, wand, polygon or freehand. The wand runs in the browser on `/api/patches/<slide>/<x>/<y>/image`, the patch at 512 × 512 on the grid of its masks, after a small blur: it fills outward from the click through pixels within an RGB distance (sensitivity) of the colour clicked, within a fixed distance on screen (not drawn), so zoom sets its reach; enclosed holes of up to 256 pixels are closed. Each is a mask with one or more labels. Labels: those of the expert ROIs, plus `background` and `other`, because every pixel has to get one. A reviewer can also type a class, stored as `custom:<text>` (up to 40 characters); a workspace's typed classes are those in its patches now. Typing the name of a label on the list selects that label, in any case. Regions may overlap, so the annotation is multi-label. Regions with the same label set are one region. Choosing a label goes to the region waiting for one, or to one opened from the list; otherwise it finishes the last region and goes to everything still uncovered (`label_rest` in the log). Shift adds a second label to the last region.
- A patch is `complete` when every pixel is in a region, `partial` otherwise, and `empty` once its regions are removed. The server works this out from the masks. A region with no label isn't stored.
- Each reviewer has a workspace, as with expert ROIs: keyed by initials, which keeps people apart but is not access control.
- Stored in `patch_events` (append-only). Each save holds the whole patch, with masks run-length encoded in row order as in COCO; the latest row of a patch in a workspace is its current state.
- The browser saves shortly after each change and queues what fails, retrying until it reaches the server.
- `/api/export/patch_masks.zip` (`?reviewer=`, `?include_partial=1`): `manifest.csv`, `labels.json` and `masks/<reviewer>/<slide>/x<X>_y<Y>/<label>.png`, 512 × 512, 1-bit, 255 where the label applies. A typed class has the file `custom_<slug>_<hash>.png`, and `custom_labels` in `labels.json` gives the class of each file. The image of a patch is `read_region((x, y), 0, (size_l0, size_l0))` resized to `mask_size`, with the numbers in the manifest; the slide files have no level at 20×. Without `?reviewer=` it holds everyone's.
- `/api/export/patches.csv` has one row per patch with each label's share, with a `frac_custom:<text>` column for each typed class; without `?reviewer=` it holds everyone's, to compare them. `/api/export/patches_audit.csv` lists every save, without masks. These are available to anyone with the app's link.
- Experts choose the patches, so how they work is logged in `patch_activity` (append-only): every operation in order, sent in batches, with a repeated batch ignored. A visit runs from opening a patch to leaving it. Each visit has a clock time and an active time, which leaves out time with the browser tab hidden or with no input for 2 minutes. Times use the reviewer's clock. A closing page sends what it has with `sendBeacon`; if the browser crashes, a visit keeps the time of its last tick, at most 15 s old, and its end shows as `no close recorded`.
- `/api/export/patch_visits.csv` (`?reviewer=`) has one row per visit: `visit_order`, `pick_order` (first visits only), how the patch was reached (`via`), times, coverage and a count of each operation. A patch reached by arrow key or Enter is not a free pick, so `via` separates those from clicks. `/api/export/patch_activity.csv` (`?reviewer=`) is the operation log: one row per step in order, with `pick_order`, `annotation_order` (the step's place among those that changed this patch's annotation, across visits), the stay's times and the patch's total times. `patches.csv` includes each patch's pick order and total times. All are available to anyone with the app's link.
- A link can name the tab: `…/#patches`.

## Layout

The side panels resize and fold, the panels on the viewer move and fold, the minimap also resizes, and the sections of the review panel fold (`app/static/panels.js`; kept per browser in `localStorage`). If both side panels do not fit the window, the one opened less recently folds to its rail, so the review panel cannot end up off screen.

`/` and `/static/*` are served with `Cache-Control: no-cache`, so browsers revalidate and a deploy reaches them at once. When `style.css`, `app.js`, `panels.js`, `editor.js` or `rois.js` change in a way that must not mix with an older copy, raise the `?v=` in `index.html` too.

## Encoders and per-slide K (s1 / s2)

`s1_dense_features.py` takes `ENCODER=conch15|hoptimus0|hoptimus1|uni2h|uni|virchow2` (CONCH v1.5, H-optimus-0, H-optimus-1, UNI2-h, UNI, Virchow2). CONCH v1.5
features stay in `results/annot/seg/feat`; the others go to `results/annot/feat_<encoder>`. H-optimus-0 and UNI2-h
are gated on Hugging Face; the account whose token is in `~/.cache/huggingface` needs access to both.

`s2_segment.py` fits a cohort codebook of `K_MAX` classes (default 10) and then picks the number of segments per
slide (`ADAPT=1`, default): classes covering < `MIN_FRAC` (1 %) of the slide's tissue are folded into the nearest
present class, and the two least separable classes on the slide are merged while their separation d' is below
`D_MIN` (2.0). A merged segment keeps the id of its largest member class, so saved class labels still apply.
Output goes to `results/annot/seg_<encoder>_kmax<K>/` (`slide_k.csv` lists segments and merges per slide).
`tools/calib_dprime.py` prints the segments-per-slide distribution for a range of `D_MIN` values.

    ENCODER=uni2h STEPS=s1_dense_features sbatch --export=ALL run_annot.slurm           # GPU
    ENCODER=uni2h K_MAX=10 python s2_segment.py                                       # CPU

The review app lists every `results/annot/seg*` run in the header; switching it changes the run for every
open session, and labels are stored per run. The choice is kept in `results/annot/active_run.txt`, so it survives a restart; `SEG_DIR`, if set, still takes precedence at startup.
