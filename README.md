# easypath

Whole-slide H&E segmentation and a pathologist review app. Foundation-model tokens (CONCH v1.5, UNI, UNI2-h,
H-optimus, Virchow2) are clustered into smooth tissue segments, and a pathologist labels, merges, redraws and comments on
them in the browser. Every decision is kept in an append-only audit log.

Built for synovial sarcoma (subtypes monophasic, biphasic, poorly differentiated), but the label sets in
`ann_common.py` can be changed for other tumours.

## Layout

| Path | What |
|---|---|
| `s1_dense_features.py`, `s2_segment.py` | Dense encoder features per slide, then cohort k-means with the number of segments chosen per slide |
| `a1_*`, `a2*`, `a3_refine.py`, `c0_*`, `c1_*` | QC, zero-shot tissue maps, ROI triage, token clusters, linear-probe refinement |
| `app/` | Review app: FastAPI backend, OpenSeadragon viewer (`app/static`) |
| `ann_common.py`, `ann_db.py` | Paths, label vocabularies, SQLite audit tables |
| `deploy/chpc/` | SLURM scripts for the pipeline and for serving on a cluster node |
| `tools/` | D' calibration for segment merging, API smoke test |
| `docs/pipeline.md` | Pipeline steps, review policy and UI details |
| `app/static/tutorial.html` | Step-by-step guide for reviewers, linked from the app header |

## Data (not in this repo)

Slides, results and the review database never go into git. The app reads one folder, `EASYPATH_RESULTS`:

```
results/
  slide_manifest.csv          slide_id, file (path to the WSI), width, height, mpp, cohort, ...
  annot/
    seg_<encoder>_kmax<K>/    s2 output: clustering.json, per-slide label PNGs, slide_k.csv, exemplars
    tissue_map*/ qc_slides.csv rois_v*.csv ...
    annotations.sqlite        reviews, labels, merges (created on first start, written by the app)
```

The `file` column holds the slide path where the pipeline ran. On another host set
`EASYPATH_SLIDE_MAP=<old prefix>=<new prefix>` to point it at the copied slides.

## Run the app

Locally or on any server with Python 3.11:

```
pip install -r requirements-app.txt
export EASYPATH_RESULTS=/path/to/results EASYPATH_TOKEN=<long random string>
cd app && uvicorn server:app --host 0.0.0.0 --port 8765
```

Open `http://<host>:8765/?token=<token>`. The first visit sets a cookie.

With Docker:

```
cp .env.example .env        # set EASYPATH_TOKEN and EASYPATH_SLIDE_MAP
mkdir -p data && ln -s /path/to/results data/results && ln -s /path/to/slides data/slides
docker compose up -d --build
```

## Deploying as a service

- Put it behind HTTPS (Caddy, nginx or a cloud load balancer). The app sets the cookie `secure` when it sees
  `X-Forwarded-Proto: https`.
- Access is one shared token. For named reviewers use the proxy's login (OAuth2 proxy, Cloudflare Access) in
  front of it; reviewer initials are still recorded per decision.
- Run one worker: the app keeps open slides and a run selection in memory, and SQLite takes the writes.
- Back up `results/annot/annotations.sqlite`; it holds every review decision.
- Slides are patient material. Check the data use agreement before copying them off the cluster.

## Run the pipeline (GPU)

On CHPC, from the repo root (`mkdir -p logs` first):

```
pip install -r requirements-pipeline.txt
ENCODER=virchow2 STEPS=s1_dense_features sbatch --export=ALL deploy/chpc/run_annot.slurm
ENCODER=virchow2 K_MAX=10 D_MIN=1.5 python s2_segment.py
sbatch deploy/chpc/serve.slurm
```

UNI2-h, H-optimus and Virchow2 weights are gated on Hugging Face; log in with an account that has access.
See `docs/pipeline.md` for every step and setting.

## License

MIT, see `LICENSE`. `app/static/vendor/openseadragon.min.js` is OpenSeadragon (BSD 3-Clause). Model weights (CONCH, UNI, H-optimus, Virchow2) are not included and keep their own licenses.
