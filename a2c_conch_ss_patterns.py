"""A2c: CONCH v1 zero-shot scores for the synovial sarcoma patterns and stroma types, per patch.

Reuses the CONCH v1 patch embeddings from a2b (no re-encoding). Adds the corded pattern, which the
a2b pattern prompts lack, and splits stroma into fibrous / hyalinized / myxoid / calcified. The slide
review panel averages these over each segment to predict a label for it.

Outputs (results/annot/)
  tissue_map/<slide_id>_ss.npz   gx, gy, step, pattern_probs [N, 4], stroma_probs [N, 4], class names
  tissue_map_ss_prior.json       cohort mean pattern probs over tumour patches and stroma-type probs over
                                 stroma patches; the panel divides by these so a class is called when it
                                 is above its cohort baseline, not when it is the raw argmax
"""
import json

import h5py
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from a2b_conch_patch_map import CONCH_REV, OUT as CONCH_DIR, text_classifier
from ann_common import ANN, MAP_DIR, PATCH_DIR, SS_PATTERN, STROMA_TYPE


@torch.inference_mode()
def main():
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    ckpt = hf_hub_download("MahmoodLab/CONCH", "pytorch_model.bin", revision=CONCH_REV)
    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=ckpt)
    model = model.cuda().eval()
    tok = get_tokenizer()
    Wp, Ws = text_classifier(model, tok, SS_PATTERN), text_classifier(model, tok, STROMA_TYPE)
    scale = model.logit_scale.exp()

    pat_t, str_s = [], []
    for h5 in sorted(PATCH_DIR.glob("*.h5")):
        sid = h5.stem
        p = CONCH_DIR / f"{sid}.h5"
        if not p.exists():
            continue
        with h5py.File(p) as f:
            E, coords = torch.from_numpy(f["features"][:]).cuda(), f["coords"][:]
            step = int(f["coords"].attrs["patch_size_level0"])
        pp = (scale * E @ Wp).softmax(-1).cpu().numpy()
        ps = (scale * E @ Ws).softmax(-1).cpu().numpy()
        np.savez(MAP_DIR / f"{sid}_ss.npz", gx=coords[:, 0] // step, gy=coords[:, 1] // step, step=step,
                 pattern_probs=pp, stroma_probs=ps, pattern_classes=np.array(list(SS_PATTERN)),
                 stroma_classes=np.array(list(STROMA_TYPE)))
        z = np.load(MAP_DIR / f"{sid}_conch.npz")
        top = np.array([str(c) for c in z["classes"]])[z["probs"].argmax(1)]
        pat_t.append(pp[top == "tumor"])
        str_s.append(ps[top == "stroma"])
        print(sid, len(E), "patches", flush=True)
    P, Sm = np.concatenate(pat_t), np.concatenate(str_s)
    prior = dict(pattern=dict(zip(SS_PATTERN, P.mean(0).round(4).tolist())),
                 stroma=dict(zip(STROMA_TYPE, Sm.mean(0).round(4).tolist())),
                 n_tumor_patches=len(P), n_stroma_patches=len(Sm),
                 pattern_argmax=dict(zip(SS_PATTERN, np.bincount(P.argmax(1), minlength=len(SS_PATTERN)).tolist())),
                 stroma_argmax=dict(zip(STROMA_TYPE, np.bincount(Sm.argmax(1), minlength=len(STROMA_TYPE)).tolist())))
    json.dump(prior, open(ANN / "tissue_map_ss_prior.json", "w"), indent=1)
    print(json.dumps(prior, indent=1))


if __name__ == "__main__":
    main()
