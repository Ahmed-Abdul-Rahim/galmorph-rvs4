#!/usr/bin/env python3
"""
Regenerate everything the C build validates against, from the trained checkpoint.
Run from the repository ROOT:

    python scripts/generate_data.py

Produces (deterministically):
  weights.bin                     - flat weight blob the C program loads (from model.pth)
  test_data/sample_{0..4}_*.bin   - input images + per-layer + softmax PyTorch references
  test_data/variety_{class}.bin   - one synthetic input per output class (demo)

Requirements: see requirements.txt (torch, numpy, einops). The PyTorch model
lives in the pymodel/ package. This lets the .bin files be reproduced
instead of committed.
"""
import os, sys
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)   # so `import pymodel` finds the pymodel/ package
from pymodel import GalaxyClassifierS4D

CKPT    = os.path.join(REPO, "model.pth")
WEIGHTS = os.path.join(REPO, "weights.bin")
TEST    = os.path.join(REPO, "test_data")

def dump(t, path):
    np.asarray(t.detach().cpu().numpy(), dtype=np.float32).flatten().tofile(path)

def main():
    os.makedirs(TEST, exist_ok=True)
    if not os.path.isfile(CKPT):
        sys.exit(f"ERROR: checkpoint not found at {os.path.relpath(CKPT, REPO)}")
    print(f"[*] checkpoint: {os.path.relpath(CKPT, REPO)}")
    model = GalaxyClassifierS4D(colored=False)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    # 1) sync the flat C weight blob (same key order the C forward pass expects)
    keys = ['hilbert_scan.indices','uproject.weight','uproject.bias',
            's4_1.log_dt','s4_1.log_A_real','s4_1.A_imag','s4_1.C','s4_1.D',
            's4_2.log_dt','s4_2.log_A_real','s4_2.A_imag','s4_2.C','s4_2.D',
            'fc.weight','fc.bias']
    with open(WEIGHTS, "wb") as f:
        for k in keys:
            t = model.state_dict()[k].cpu().detach()
            if t.is_complex(): t = torch.view_as_real(t)
            if t.dtype == torch.int64:   t = t.to(torch.int32)
            elif t.dtype == torch.float64: t = t.to(torch.float32)
            f.write(t.numpy().flatten().tobytes())
    print("[*] wrote weights.bin")

    # 2) per-layer + softmax references for samples 0..4 (deterministic torch seeds)
    acts = {}
    def hook(name):
        def h(m,i,o): acts[name] = (o[0] if isinstance(o,tuple) else o).detach()
        return h
    model.hilbert_scan.register_forward_hook(hook('hilbert'))
    model.uproject.register_forward_hook(hook('uproject'))
    model.s4_1.register_forward_hook(hook('s4_1'))
    model.act1.register_forward_hook(hook('gelu_1'))
    model.s4_2.register_forward_hook(hook('s4_2'))
    model.act2.register_forward_hook(hook('gelu_2'))
    model.take_last.register_forward_hook(hook('takelast'))
    model.fc.register_forward_hook(hook('fc'))
    for i in range(5):
        torch.manual_seed(i)
        img = torch.randn(1,1,64,64)
        with torch.no_grad():
            acts['softmax'] = model(img, return_logits=False).detach()
        pfx = os.path.join(TEST, f"sample_{i}")
        dump(img, f"{pfx}_img.bin")
        for name,t in acts.items():
            dump(t, f"{pfx}_{name}.bin")
        print(f"[*] sample {i}: {['Round','In-between','Cigar','Edge-on'][int(acts['softmax'].argmax())]}")

    # 3) one synthetic input per class (numpy seeds chosen to hit each label)
    for name, seed in {'round':35, 'inbetween':45, 'cigar':6, 'edgeon':0}.items():
        np.random.seed(seed)
        np.random.randn(1,64,64).astype(np.float32).tofile(os.path.join(TEST, f"variety_{name}.bin"))
    print("[*] wrote test_data/variety_{round,inbetween,cigar,edgeon}.bin")
    print("\nDone. Now validate:  make && ./build/main --validate")

if __name__ == "__main__":
    main()
