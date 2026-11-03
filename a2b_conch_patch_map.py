"""A2b: patch-level zero-shot tissue map with CONCH v1 (image + its own text encoder).

Same 512 px / 20x grid as the TITAN map, but each patch is classified on its own (about 256 um),
so the map is sharper than the TITAN 5 x 5 neighbourhood map. CONCH v1.5 (used for TITAN) has no
text tower, so patches are re-encoded with the CONCH v1 image encoder, which is aligned to the text.

Outputs (results/annot/)
  conch_v1/<slide_id>.h5            CONCH v1 image embeddings [N, 512] (normalised) + coords
  tissue_map/<slide_id>_conch.npz   gx, gy, step, probs [N, tissue], pattern_probs [N, pattern]
  tissue_map_scale_conch.json       per-class 5th / 95th percentiles used for display
  map_agreement.csv                 per-slide agreement between the TITAN and CONCH tumour maps
  rois_v000.csv                     gains conch_* columns (ROI means of the patch probabilities)
"""
import json

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

from ann_common import ANN, MAP_DIR, PATCH_DIR, PATTERN, RES, ROI_PATCHES, TISSUE, load_patches

CONCH_REV = "f9ca9f877171a28ade80228fb195ac5d79003357"
TEMPLATES = ["CLASSNAME.", "a photomicrograph showing CLASSNAME.", "an image of CLASSNAME.",
             "a histopathological image showing CLASSNAME.", "an H&E stained image of CLASSNAME.",
             "this is CLASSNAME.", "an example of CLASSNAME."]
OUT = ANN / "conch_v1"
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
        return self.tf(im)


@torch.inference_mode()
def text_classifier(model, tok, classes):
    from conch.open_clip_custom import tokenize
    W = []
    for prompts in classes.values():
        texts = [t.replace("CLASSNAME", p) for p in prompts for t in TEMPLATES]
        e = model.encode_text(tokenize(texts=texts, tokenizer=tok).cuda())
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        W.append(torch.nn.functional.normalize(e, dim=-1))
    return torch.stack(W, 1)  # [512, n_classes]


@torch.inference_mode()
def main():
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    ckpt = hf_hub_download("MahmoodLab/CONCH", "pytorch_model.bin", revision=CONCH_REV)
    model, tf = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=ckpt)
    model = model.cuda().eval()
    tok = get_tokenizer()
    Wt, Wp = text_classifier(model, tok, TISSUE), text_classifier(model, tok, PATTERN)
    scale = model.logit_scale.exp()
    tcls = list(TISSUE)

    man = pd.read_csv(RES / "slide_manifest.csv").set_index("slide_id")
    allp, agree, roi_rows = [], [], []
    for h5 in sorted(PATCH_DIR.glob("*.h5")):
        sid = h5.stem
        _, coords, step = load_patches(sid)
        out = OUT / f"{sid}.h5"
        if out.exists():
            with h5py.File(out) as f:
                E = torch.from_numpy(f["features"][:]).cuda()
        else:
            E = []
            for xb in DataLoader(Patches(man.loc[sid, "file"], coords, step, tf), batch_size=128, num_workers=10):
                with torch.autocast("cuda", torch.float16):
                    E.append(model.encode_image(xb.cuda(), proj_contrast=True, normalize=True).float())
            E = torch.cat(E)
            with h5py.File(out, "w") as f:
                f["features"] = E.cpu().numpy()
                f["coords"] = coords
                f["coords"].attrs["patch_size_level0"] = step
        pt = (scale * E @ Wt).softmax(-1).cpu().numpy()
        pp = (scale * E @ Wp).softmax(-1).cpu().numpy()
        gx, gy = coords[:, 0] // step, coords[:, 1] // step
        np.savez(MAP_DIR / f"{sid}_conch.npz", gx=gx, gy=gy, step=step, probs=pt, pattern_probs=pp,
                 classes=np.array(tcls), pattern_classes=np.array(list(PATTERN)))
        allp.append(pt)

        # agreement with the TITAN neighbourhood map
        tm = np.load(MAP_DIR / f"{sid}.npz")
        k = tcls.index("tumor")
        rho = spearmanr(tm["probs"][:, k], pt[:, k]).statistic if len(pt) > 3 else np.nan
        agree.append(dict(slide_id=sid, n=len(pt), rho_tumor=rho,
                          titan_tumor_top=float((tm["probs"].argmax(1) == k).mean()),
                          conch_tumor_top=float((pt.argmax(1) == k).mean()),
                          top_class_agreement=float((tm["probs"].argmax(1) == pt.argmax(1)).mean())))

        # ROI summaries (same 4 x 4 blocks as a2)
        bx, by = gx // ROI_PATCHES, gy // ROI_PATCHES
        for i, j in set(zip(bx, by)):
            m = (bx == i) & (by == j)
            top = pt[m].argmax(1)
            tum = top == k
            row = dict(roi_id=f"{sid}__{i}_{j}", conch_p_tumor=float(pt[m, k].mean()),
                       conch_tumor_frac=float(tum.mean()),
                       conch_necrosis_frac=float((top == tcls.index("necrosis")).mean()),
                       conch_top_class=tcls[np.bincount(top, minlength=len(tcls)).argmax()])
            src = pp[m][tum] if tum.any() else pp[m]
            row.update({f"conch_pattern:{c}": float(v) for c, v in zip(PATTERN, src.mean(0))})
            roi_rows.append(row)
        print(sid, len(pt), "patches, conch tumor-top", round(agree[-1]["conch_tumor_top"], 2), "rho vs titan", round(rho, 2), flush=True)

    allp = np.concatenate(allp)
    json.dump({c: [float(np.percentile(allp[:, i], 5)), float(np.percentile(allp[:, i], 95))] for i, c in enumerate(tcls)},
              open(ANN / "tissue_map_scale_conch.json", "w"), indent=1)
    A = pd.DataFrame(agree)
    A.to_csv(ANN / "map_agreement.csv", index=False)
    print(A.describe().round(3).to_string())
    print("patch top class (CONCH):", pd.Series(np.array(tcls)[allp.argmax(1)]).value_counts().to_dict())

    rp = ANN / "rois_v000.csv"
    if rp.exists():
        R = pd.read_csv(rp)
        R = R[[c for c in R.columns if not c.startswith("conch_")]].merge(pd.DataFrame(roi_rows), on="roi_id", how="left")
        R.to_csv(rp, index=False)
        print(pd.crosstab(R.pred_tumor, R.conch_top_class).to_string())


if __name__ == "__main__":
    main()
