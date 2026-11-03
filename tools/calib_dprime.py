"""Calibrate D_MIN for s2's per-slide merging: pairwise d' between classes present on each slide, and the
segments-per-slide distribution that each threshold would give. Run from annot/ with the s2 env vars."""
import sys

import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, ".")
import s2_segment as s2  # noqa: E402

rng = np.random.default_rng(s2.SEED)
files = sorted(s2.FEAT.glob("*.npz"))
samp, slides = [], []
for f in files:
    z = np.load(f)
    feat, tis = s2.unit(z["feat"].astype(np.float32)), z["tissue"].astype(np.float32)
    X = feat.reshape(-1, feat.shape[-1])[tis.reshape(-1) >= 0.5]
    samp.append(X[rng.choice(len(X), min(6000, len(X)), replace=False)])
    slides.append((f.stem, feat, tis))
cent = s2.unit(MiniBatchKMeans(s2.K, batch_size=8192, n_init=10, max_iter=300, random_state=s2.SEED)
               .fit(np.concatenate(samp)).cluster_centers_)
pairs, per = [], {}
for sid, feat, tis in slides:
    P = np.exp((feat @ cent.T - 1) / s2.TAU)
    P /= P.sum(-1, keepdims=True)
    tw = (tis > 0.2).astype(np.float32)
    P = np.stack([ndi.gaussian_filter(P[..., k] * tw, 1.0) for k in range(s2.K)], -1) / (ndi.gaussian_filter(tw, 1.0)[..., None] + 1e-6)
    T = tis >= 0.5
    X, a = feat[T], P[T].argmax(-1)
    frac = np.bincount(a, minlength=s2.K) / len(a)
    pres = [k for k in range(s2.K) if frac[k] >= s2.MIN_FRAC]
    for i in range(len(pres)):
        for j in range(i + 1, len(pres)):
            pairs.append(s2.dprime(X[a == pres[i]], X[a == pres[j]]))
    per[sid] = (X, a, len(pres))
pairs = np.array(pairs)
print("present classes per slide:", np.bincount([v[2] for v in per.values()]).tolist())
print("pairwise d' quantiles 5/25/50/75/95:", np.percentile(pairs, [5, 25, 50, 75, 95]).round(2).tolist())
for d in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    s2.D_MIN = d
    ks = [len(s2.adapt_groups(X, a, cent, np.random.default_rng(0))[0]) for X, a, _ in per.values()]
    print(f"D_MIN {d}: segments per slide median {np.median(ks)}, min {min(ks)}, max {max(ks)}, counts {np.bincount(ks).tolist()}")
