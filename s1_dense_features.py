"""S1: seamless dense ViT token features per slide, input to self-supervised segmentation.

Windows at 20x with 50 % overlap tile each slide's tissue, and every window gives a grid of final-layer
ViT tokens. Overlapping windows are blended with a Hann weight into one slide canvas, so the feature map
has no patch seams and every token sees context on all sides. Features are projected to 64 dims with a
PCA fitted on tokens of the same encoder, which keeps the canvases small.

Env ENCODER (default conch15)
  conch15    CONCH v1.5 (TITAN patch encoder), 512 px windows -> 28 x 28 tokens (about 9 um)
  hoptimus0  H-optimus-0 (Bioptimus, ViT-g/14), 224 px windows -> 16 x 16 tokens (about 7 um)
  hoptimus1  H-optimus-1 (Bioptimus, ViT-g/14), 224 px windows -> 16 x 16 tokens (about 7 um)
  uni2h      UNI2-h (MahmoodLab, ViT-H/14), 224 px windows -> 16 x 16 tokens (about 7 um)
  uni        UNI (MahmoodLab, ViT-L/16), 224 px windows -> 14 x 14 tokens (about 8 um)
  virchow2   Virchow2 (Paige, ViT-H/14), 224 px windows -> 16 x 16 tokens (about 7 um)

Outputs (results/annot/)
  conch15: seg/pca.npz, seg/feat/<slide_id>.npz     others: feat_<encoder>_pca.npz, feat_<encoder>/<slide_id>.npz
  feat [h, w, 64] fp16, tissue [h, w] fp16 (token tissue fraction), x0, y0 (level-0 origin), ts (level-0 px per token)
"""
import importlib.util
import os

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ann_common import ANN, PATCH_DIR, RES, ROOT, SEED, TITAN_REV

ENCODER = os.environ.get("ENCODER", "conch15")
WIN, NTOK = {"conch15": (512, 28), "hoptimus0": (224, 16), "hoptimus1": (224, 16), "uni2h": (224, 16), "uni": (224, 14), "virchow2": (224, 16)}[ENCODER]   # window px at 20x, tokens per side
FEAT_DIR = ANN / "seg" / "feat" if ENCODER == "conch15" else ANN / f"feat_{ENCODER}"
PCA_PATH = ANN / "seg" / "pca.npz" if ENCODER == "conch15" else ANN / f"feat_{ENCODER}_pca.npz"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("s01", ROOT / "scripts/01_manifest_tiling.py")
s01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s01)


def load_encoder():
    """(tokens: images [B, 3, H, W] -> patch tokens [B, NTOK * NTOK, D], image transform)"""
    if ENCODER == "conch15":
        from transformers import AutoModel
        titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True, revision=TITAN_REV)
        conch, tf = titan.return_conch()
        trunk = conch.trunk.cuda().eval()
        return (lambda x: trunk(x, return_all_tokens=True)[:, 1:]), tf
    import timm
    from torchvision import transforms as T
    if ENCODER in ("hoptimus0", "hoptimus1"):
        m = timm.create_model(f"hf-hub:bioptimus/H-optimus-{ENCODER[-1]}", pretrained=True, init_values=1e-5, dynamic_img_size=False)
        norm = T.Normalize((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517))
    elif ENCODER == "virchow2":
        m = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=timm.layers.SwiGLUPacked,
                              act_layer=torch.nn.SiLU)
        norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    elif ENCODER == "uni":
        m = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        m = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, img_size=224, patch_size=14, depth=24,
                              num_heads=24, init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
                              no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                              reg_tokens=8, dynamic_img_size=True)
        norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    m = m.cuda().eval()
    n0 = m.num_prefix_tokens   # class + register tokens
    return (lambda x: m.forward_features(x)[:, n0:]), T.Compose([T.ToTensor(), norm])


def windows(slide):
    """Window grid (level 0) over the tissue bounding box, stride = half a window."""
    W, H = slide.dimensions
    mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
    ds = s01.THUMB_UM / mpp
    thumb = np.asarray(slide.get_thumbnail((int(W / ds), int(H / ds))).convert("RGB"))
    mask = s01.tissue_mask(thumb)
    sx, sy = W / mask.shape[1], H / mask.shape[0]
    step = int(round(WIN * 0.5 / mpp))
    step -= step % 2
    stride = step // 2
    ys, xs = np.nonzero(mask)
    bx0, by0 = max(int(xs.min() * sx) - stride, 0), max(int(ys.min() * sy) - stride, 0)
    bx1, by1 = min(int((xs.max() + 1) * sx) + stride, W), min(int((ys.max() + 1) * sy) + stride, H)
    nx, ny = max((bx1 - bx0 - step) // stride + 1, 1), max((by1 - by0 - step) // stride + 1, 1)
    keep = []
    for j in range(ny):
        for i in range(nx):
            x, y = bx0 + i * stride, by0 + j * stride
            if mask[int(y / sy):int((y + step) / sy) + 1, int(x / sx):int((x + step) / sx) + 1].mean() >= 0.02:
                keep.append((i, j, x, y))
    return keep, step, (bx0, by0), (nx, ny)


class Windows(Dataset):
    def __init__(self, path, wins, step, tf):
        self.path, self.wins, self.step, self.tf, self.s = path, wins, step, tf, None

    def __len__(self):
        return len(self.wins)

    def __getitem__(self, k):
        if self.s is None:
            self.s = openslide.OpenSlide(self.path)
        _, _, x, y = self.wins[k]
        im = self.s.read_region((int(x), int(y)), 0, (self.step, self.step)).convert("RGB").resize((WIN, WIN), Image.BILINEAR)
        od = -np.log(np.clip(np.asarray(im, np.float32), 1, 255) / 255.0).mean(-1)
        tis = F.adaptive_avg_pool2d(torch.from_numpy((od > 0.12).astype(np.float32))[None, None], NTOK)[0, 0]
        return self.tf(im), tis, k


def encode(tok, xb):
    with torch.autocast("cuda", torch.float16):
        t = tok(xb.cuda())
    assert t.shape[1] == NTOK * NTOK, t.shape
    return F.normalize(t.float(), dim=-1)


def fit_pca(tok, tf, man):
    """64-dim PCA of unit-norm tissue tokens: CONCH v1.5 from the c0 dense tokens, other encoders from
    40 random windows per slide."""
    if PCA_PATH.exists():
        z = np.load(PCA_PATH)
        return z["mean"], z["comp"]
    rng = np.random.default_rng(SEED)
    X = []
    if ENCODER == "conch15":
        for h5 in sorted((ANN / "dense_tokens").glob("*.h5")):
            with h5py.File(h5) as f:
                t, tis = f["tokens"][:], f["tissue"][:]
            t = t.reshape(-1, t.shape[-1])[tis.reshape(-1) > 0.5]
            X.append(t[rng.choice(len(t), min(2000, len(t)), replace=False)].astype(np.float32))
    else:
        for h5 in sorted(PATCH_DIR.glob("*.h5")):
            path = man.loc[h5.stem, "file"]
            wins, step, _, _ = windows(openslide.OpenSlide(path))
            sel = [wins[i] for i in rng.choice(len(wins), min(40, len(wins)), replace=False)]
            for xb, tb, _ in DataLoader(Windows(path, sel, step, tf), batch_size=40, num_workers=8):
                t = encode(tok, xb).cpu().numpy()
                t = t.reshape(-1, t.shape[-1])[tb.reshape(-1).numpy() > 0.5]
                X.append(t[rng.choice(len(t), min(2000, len(t)), replace=False)])
    X = np.concatenate(X).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    mean = X.mean(0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    np.savez(PCA_PATH, mean=mean, comp=vt[:64])
    print(ENCODER, "PCA fitted on", len(X), "tokens", flush=True)
    return mean, vt[:64]


@torch.inference_mode()
def main():
    tok, tf = load_encoder()
    man = pd.read_csv(RES / "slide_manifest.csv").set_index("slide_id")
    mean, comp = fit_pca(tok, tf, man)
    mean_t, comp_t = torch.from_numpy(mean).cuda(), torch.from_numpy(comp.astype(np.float32)).cuda()
    hann = np.hanning(NTOK + 2)[1:-1]
    wgt = (np.outer(hann, hann) + 0.05).astype(np.float32)
    half = NTOK // 2

    for h5 in sorted(PATCH_DIR.glob("*.h5")):
        sid = h5.stem
        out = FEAT_DIR / f"{sid}.npz"
        if out.exists():
            continue
        slide = openslide.OpenSlide(man.loc[sid, "file"])
        wins, step, (bx0, by0), (nx, ny) = windows(slide)
        h, w = (ny + 1) * half, (nx + 1) * half
        acc = np.zeros((h, w, comp.shape[0]), np.float32)
        tacc = np.zeros((h, w), np.float32)
        wacc = np.zeros((h, w), np.float32)
        for xb, tb, kb in DataLoader(Windows(man.loc[sid, "file"], wins, step, tf), batch_size=32, num_workers=10):
            t = encode(tok, xb)
            z = ((t - mean_t) @ comp_t.T).reshape(len(t), NTOK, NTOK, -1).cpu().numpy()
            for b, k in enumerate(kb.numpy()):
                i, j = wins[k][0], wins[k][1]
                r0, c0 = j * half, i * half
                acc[r0:r0 + NTOK, c0:c0 + NTOK] += z[b] * wgt[..., None]
                tacc[r0:r0 + NTOK, c0:c0 + NTOK] += tb[b].numpy() * wgt
                wacc[r0:r0 + NTOK, c0:c0 + NTOK] += wgt
        ok = wacc > 0
        acc[ok] /= wacc[ok][:, None]
        tacc[ok] /= wacc[ok]
        np.savez_compressed(out, feat=acc.astype(np.float16), tissue=tacc.astype(np.float16),
                            x0=bx0, y0=by0, ts=step / NTOK)
        print(sid, len(wins), "windows", f"{h}x{w} tokens", flush=True)


if __name__ == "__main__":
    main()
