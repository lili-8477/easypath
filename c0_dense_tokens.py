"""C0: dense (sub-patch) tokens from CONCH v1.5 for high-resolution clustering.

The slides are small biopsies, so one embedding per 256 um patch is too coarse. CONCH v1.5 is a
ViT-L/16: every 512 px / 20x patch (resized to 448 px) already yields a 28 x 28 grid of final-layer
tokens (about 9 um each, contextualised within the patch). These are average-pooled to a 14 x 14
grid (about 18 um, 1 to 2 cells) and stored with a per-token tissue fraction. c1 can pool further.

Outputs (results/annot/dense_tokens/)
  <slide_id>.h5   tokens [N_patch, 14, 14, 1024] fp16, tissue [N_patch, 14, 14], coords [N_patch, 2],
                  attrs patch_size_level0, grid
"""
import h5py
import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from ann_common import ANN, PATCH_DIR, RES, TITAN_REV, load_patches

GRID = 14
OUT = ANN / "dense_tokens"
OUT.mkdir(exist_ok=True)


class Patches(Dataset):
    def __init__(self, path, coords, step, tf):
        self.path, self.coords, self.step, self.tf, self.s = path, coords, step, tf, None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        if self.s is None:
            self.s = openslide.OpenSlide(self.path)
        x, y = self.coords[i]
        im = self.s.read_region((int(x), int(y)), 0, (self.step, self.step)).convert("RGB").resize((512, 512), Image.BILINEAR)
        a = np.asarray(im, np.float32)
        od = -np.log(np.clip(a, 1, 255) / 255.0).mean(-1)          # same OD tissue rule as a1_qc
        tissue = torch.from_numpy((od > 0.12).astype(np.float32))[None, None]
        tissue = F.adaptive_avg_pool2d(tissue, GRID)[0, 0]
        return self.tf(im), tissue


@torch.inference_mode()
def main():
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True, revision=TITAN_REV)
    conch, tf = titan.return_conch()
    trunk = conch.trunk.cuda().eval()
    del titan
    man = pd.read_csv(RES / "slide_manifest.csv").set_index("slide_id")
    for h5 in sorted(PATCH_DIR.glob("*.h5")):
        sid = h5.stem
        out = OUT / f"{sid}.h5"
        if out.exists():
            continue
        _, coords, step = load_patches(sid)
        toks, tis = [], []
        for xb, tb in DataLoader(Patches(man.loc[sid, "file"], coords, step, tf), batch_size=32, num_workers=10):
            with torch.autocast("cuda", torch.float16):
                t = trunk(xb.cuda(), return_all_tokens=True)[:, 1:]          # drop CLS -> [B, 784, 1024]
            n = int(t.shape[1] ** 0.5)
            t = t.float().transpose(1, 2).reshape(len(t), -1, n, n)
            t = F.adaptive_avg_pool2d(t, GRID).permute(0, 2, 3, 1)          # [B, 14, 14, 1024]
            toks.append(t.half().cpu().numpy())
            tis.append(tb.numpy())
        with h5py.File(out, "w") as f:
            f.create_dataset("tokens", data=np.concatenate(toks), compression="lzf")
            f["tissue"] = np.concatenate(tis).astype(np.float16)
            f["coords"] = coords
            f.attrs["patch_size_level0"] = step
            f.attrs["grid"] = GRID
        print(sid, len(coords), "patches", flush=True)


if __name__ == "__main__":
    main()
