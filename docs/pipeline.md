# Pipeline and review details

Model proposes, pathologist decides, decisions retrain the model.

Step 1 of review is segment-level: self-supervised smooth segmentation (s1, s2) gives 16 segment classes that are named once each (tab "1 · Clusters"; it shows the segmentation when results/annot/seg exists, otherwise the c1 token clusters),
then ROIs are reviewed on the slides (tab "2 · Slides & ROIs").

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

## Cluster review (step 1)

- Each cluster: 16 typical tokens (nearest the centroid, at most 3 per slide) and 8 random ones, shown as a white box inside 4x its size of surrounding tissue; more random tokens on demand; click opens the slide there with that cluster overlaid.
- Labels: tumor, stroma, necrosis, inflammation, benign epithelium, fat, muscle, blood, artifact, mixed, other; tumour pattern; purity (clean / mostly / mixed); "driven by one slide or stain batch" flag. Stored append-only in `cluster_reviews`, keyed by clustering id so a new clustering never inherits old labels.
- "batch?" marks clusters where one slide holds > 40 % of the tokens or one cohort is enriched > 1.8x.

## Self-supervised segmentation (step 1, current)

- Labels: SS spindle cell, SS glandular, SS corded / solid epithelial, SS round cell (poorly differentiated), stroma, necrosis, inflammation, hemorrhage, fat, muscle, vessel, other normal tissue, artifact, mixed.
- Slide-level subtype is derived, not segmented: biphasic if glandular + corded >= 5 % of labelled tumour area, otherwise monophasic; poorly differentiated flag if round cell >= 20 % (`BIPHASIC_MIN_EPI`, `PD_MIN_ROUND` in ann_common.py). Shown in the slide header and the segment panel.
- Export: `api/export/segments.geojson?slide=<id>` gives the polygons with the reviewed label as QuPath classification.

## Slide review (slides tab)

The slides tab shows the slide with the smooth segmentation overlay only; ROI squares, the ROI filmstrip and the
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

## Encoders and per-slide K (s1 / s2)

`s1_dense_features.py` takes `ENCODER=conch15|hoptimus0|hoptimus1|uni2h|uni|virchow2` (CONCH v1.5, H-optimus-0, H-optimus-1, UNI2-h, UNI, Virchow2). CONCH v1.5
features stay in `results/annot/seg/feat`; the others go to `results/annot/feat_<encoder>`. H-optimus-0 and UNI2-h
are gated on Hugging Face; the account whose token is in `~/.cache/huggingface` needs access to both.

`s2_segment.py` fits a cohort codebook of `K_MAX` classes (default 10) and then picks the number of segments per
slide (`ADAPT=1`, default): classes covering < `MIN_FRAC` (1 %) of the slide's tissue are folded into the nearest
present class, and the two least separable classes on the slide are merged while their separation d' is below
`D_MIN` (2.0). A merged segment keeps the id of its largest member class, so class labels from tab 1 still apply.
Output goes to `results/annot/seg_<encoder>_kmax<K>/` (`slide_k.csv` lists segments and merges per slide).
`tools/calib_dprime.py` prints the segments-per-slide distribution for a range of `D_MIN` values.

    ENCODER=uni2h STEPS=s1_dense_features sbatch --export=ALL run_annot.slurm           # GPU
    ENCODER=uni2h K_MAX=10 python s2_segment.py                                       # CPU

The review app lists every `results/annot/seg*` run in the header; switching it changes the run for every
open session, and labels are stored per run.
