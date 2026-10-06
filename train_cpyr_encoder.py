#!/usr/bin/env python3
"""
Train CPYR's convolutional encoder from the real repo code and export the artifacts
that the secure-boot demo encrypts, places on the dm-verity partition and measures.

Source of each piece (github.com/moustafa991982/Cpyr):
  * BaseModel                -> basemodel.py                      (imported as-is)
  * split / Sequence_dl      -> data_handler.py                   (imported as-is)
  * conv_enocder             -> "Integration script.ipynb" cell 21 (copied verbatim; it isn't in a .py file)
  * binary loss (bits 120:500) -> "Integration script.ipynb" cell 19 (copied verbatim)
  * data                     -> data/bin/imag_*.npy (1,201 hex->binary frames)
Changes from the notebook: CPU instead of .cuda(), num_workers=0, a plain training loop
(learner.fit hard-codes .cuda()), Adam lr 1e-2 as in notebook cell 23, bs 8 / seq_len 20 / 112x112
as in docs/methodology.md, and more epochs because the public data gives only 4 batches per epoch.

The repo has no labelled attack capture, so ATTACK frames are synthesised from held-out
real frames: first 120 bits (time/type/header) kept, the rest of the frame randomised
(payload fuzzing). Say so when you present the numbers.

Usage: python3 train_cpyr_encoder.py --repo /path/to/Cpyr [--epochs 40]
Output: cpyr_artifacts/  (basemodel.py, conv_encoder.py, model.pt, manifest + eval traffic)
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpyr_artifacts"))
args = ap.parse_args()
repo = os.path.abspath(args.repo)
sys.path.insert(0, repo)
torch.manual_seed(0)
np.random.seed(0)

from basemodel import BaseModel            # noqa: E402  (real repo file)
from data_handler import split, Sequence_dl  # noqa: E402  (real repo file)

# ---- verbatim from Integration script.ipynb, cells 14, 19 and 21 ---------------------------
CONV_ENCODER_SRC = '''
bs = 8
seq_len = 20
overlap=0

class binary:
    def __call__(self,ip,tgt):
        bs = tgt.size(0)
        seq_len = tgt.size(1)
        #ip is what comes out of the model

        ip = ip.view(bs,seq_len,-1).contiguous()[:,:,120:500] #customized setup
        tgt = tgt.view(bs,seq_len,-1).contiguous()[:,:,120:500] #customized setup

        loss = nn.functional.binary_cross_entropy(ip,tgt)
        return loss # returned loss

class conv_enocder(BaseModel):
    def __init__(self):
        super().__init__()

        self.encode = nn.Conv2d(seq_len,1,(1,1))
        self.decode = nn.ConvTranspose2d(1,seq_len,(1,1))
        self.relu = nn.PReLU()

    def forward(self,x):
        bs = x.size(0)
        ts = x.size(1)
        encode = self.encode(x)
        decode = self.decode(encode.view(bs,1,112,112))



        img = torch.sigmoid(decode)
        return img#torch.sigmoid(op)
'''
ns = {"torch": torch, "nn": nn, "BaseModel": BaseModel}
exec(CONV_ENCODER_SRC, ns)
conv_enocder, binary, bs, seq_len = ns["conv_enocder"], ns["binary"], ns["bs"], ns["seq_len"]

# ---- data: same split call as the notebook -------------------------------------------------
data_dir = os.path.join(repo, "data", "bin") + "/"
trn_, val_, test = split(data_dir, prc=0.2, dataset_pct=1, test_pcg=0.2, dim=(112, 112),
                         word_type=torch.float)
trn_dl = Sequence_dl(trn_, bs=bs, seq_len=seq_len, overlap=0, num_workers=0, drop_last=True,
                     dim=[112, 112])
print(f"frames: train {len(trn_)}, val {len(val_)}, test {len(test)}; "
      f"train batches {len(trn_dl)} (bs {bs} x seq {seq_len})")


def windows(ds):
    """Consecutive non-overlapping 20-frame windows -> tensor (n, 20, 112, 112)."""
    x = torch.stack([ds[i] for i in range(len(ds))])
    n = len(x) // seq_len
    return x[: n * seq_len].view(n, seq_len, 112, 112)


def fuzz(win, rng):
    """Payload fuzzing: keep first 120 bits of each frame, randomise the rest of its real length."""
    out = win.clone().view(win.size(0), win.size(1), -1)
    for i in range(out.size(0)):
        for j in range(out.size(1)):
            nz = torch.nonzero(out[i, j]).flatten()
            end = int(nz[-1]) + 1 if len(nz) else 500
            end = max(end, 500)
            out[i, j, 120:end] = torch.from_numpy(rng.integers(0, 2, end - 120)).float()
    return out.view_as(win)


def window_losses(model, win):
    crit = binary()
    with torch.no_grad():
        return [float(crit(model(w.unsqueeze(0)), w.unsqueeze(0))) for w in win]


# ---- train ---------------------------------------------------------------------------------
m = conv_enocder()
opt = torch.optim.Adam(m.parameters(), args.lr)
crit = binary()
for ep in range(1, args.epochs + 1):
    m.train()
    tot = 0.0
    for data in trn_dl:
        opt.zero_grad()
        loss = crit(m(data), data)
        loss.backward()
        opt.step()
        tot += loss.item()
    if ep == 1 or ep % 25 == 0 or ep == args.epochs:
        print(f"epoch {ep:>3}: train loss {tot / len(trn_dl):.4f}")
m.eval()

# ---- threshold on validation, evaluate on test + fuzzed test --------------------------------
rng = np.random.default_rng(1)
val_w, test_w = windows(val_), windows(test)
fuzz_w = fuzz(test_w, rng)
val_l, test_l, fuzz_l = (window_losses(m, w) for w in (val_w, test_w, fuzz_w))
threshold = max(val_l) * 1.15
fp = sum(v > threshold for v in test_l)
tp = sum(v > threshold for v in fuzz_l)
print(f"val   windows {len(val_l):>2}: loss {min(val_l):.4f}–{max(val_l):.4f}  → threshold {threshold:.4f}")
print(f"test  windows {len(test_l):>2}: loss {min(test_l):.4f}–{max(test_l):.4f}  flagged {fp}/{len(test_l)}")
print(f"fuzz  windows {len(fuzz_l):>2}: loss {min(fuzz_l):.4f}–{max(fuzz_l):.4f}  flagged {tp}/{len(fuzz_l)}")

# ---- export --------------------------------------------------------------------------------
os.makedirs(args.out, exist_ok=True)
buf = io.BytesIO()
torch.save(m.state_dict(), buf)
weights = buf.getvalue()
open(os.path.join(args.out, "model.pt"), "wb").write(weights)
shutil.copy(os.path.join(repo, "basemodel.py"), os.path.join(args.out, "basemodel.py"))
open(os.path.join(args.out, "conv_encoder.py"), "w").write(
    "# Verbatim from Cpyr 'Integration script.ipynb' (cells 14, 19, 21)\n"
    "import torch\nimport torch.nn as nn\n" + CONV_ENCODER_SRC)
np.save(os.path.join(args.out, "eval_normal.npy"), test_w.numpy().astype(np.uint8))
np.save(os.path.join(args.out, "eval_fuzz.npy"), fuzz_w.numpy().astype(np.uint8))
n_params = sum(p.numel() for p in m.parameters())
meta = {"threshold": round(threshold, 6), "window": seq_len, "params": n_params,
        "weights_bytes": len(weights), "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "epochs": args.epochs, "lr": args.lr, "train_frames": len(trn_), "val_frames": len(val_),
        "test_frames": len(test), "val_loss_max": max(val_l),
        "test_flagged": f"{fp}/{len(test_l)}", "fuzz_flagged": f"{tp}/{len(fuzz_l)}"}
json.dump(meta, open(os.path.join(args.out, "train_meta.json"), "w"), indent=2)
print(f"\nexported {args.out}/: model.pt = {len(weights)} B state_dict ({n_params} parameters), "
      f"sha256 {meta['weights_sha256'][:16]}…")
