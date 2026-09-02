#!/usr/bin/env python3
"""Validate a compiled S4D variant against an independent NumPy forward pass
built from the same checkpoint. Confirms the C math is correct.

Usage (run from a variant folder after bake_weights.py):
  python3 ../../scripts/validate_oracle.py --config d108_seq256 --bin ./build/host
where ./build/host is a NON-baked host build (make all) that reads weights.bin
and takes an image path argument.
"""
import os, sys, argparse, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from bake_weights import load_ckpt, CONFIGS
np.seterr(all="ignore")

def oracle(arrs, dm, half, nl, patch_dim, seq_len, img):
    idx=np.asarray(arrs["hilbert_scan.indices"]).astype(int)
    up_w=arrs["uproject.weight"].reshape(dm,patch_dim); up_b=arrs["uproject.bias"]
    grid=int(round((seq_len)**0.5)); ps=64//grid
    seq=np.zeros((seq_len,patch_dim))
    for t in range(seq_len):
        g=int(idx[t]); gh,gw=g//grid,g%grid
        k=0
        for c in range(3):
            for pr in range(ps):
                for pc in range(ps):
                    seq[t,k]=img[c,gh*ps+pr,gw*ps+pc]; k+=1
    x=seq@up_w.T+up_b
    gelu=lambda z:0.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+0.044715*z**3)))
    def s4d(inp,ldt,lar,ai,Cc,D):
        dt=np.exp(ldt); lam=(-np.exp(lar))+1j*ai
        Abar=np.exp(lam*dt[:,None]); Bbar=(Abar-1)/lam
        Cx=Cc[...,0]+1j*Cc[...,1]; Cbar=2*Cx*Bbar
        out=np.zeros_like(inp); st=np.zeros((dm,half),dtype=complex)
        for t in range(inp.shape[0]):
            u=inp[t]; st=Abar*st+u[:,None]; out[t]=D*u+np.real(np.sum(Cbar*st,axis=1))
        return out
    for L in range(nl):
        ldt=arrs[f"s4_layers.{L}.log_dt"]; lar=arrs[f"s4_layers.{L}.log_A_real"].reshape(dm,half)
        ai=arrs[f"s4_layers.{L}.A_imag"].reshape(dm,half); Cc=arrs[f"s4_layers.{L}.C"].reshape(dm,half,2)
        D=arrs[f"s4_layers.{L}.D"]; x=gelu(s4d(x,ldt,lar,ai,Cc,D))
    pooled=x[-1]; fc_w=arrs["fc.weight"].reshape(4,dm); fc_b=arrs["fc.bias"]
    logits=fc_w@pooled+fc_b; e=np.exp(logits-logits.max()); return e/e.sum()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--bin", default="./build/host")
    ap.add_argument("--ckpt-dir", default="../../checkpoints")
    a=ap.parse_args()
    C=CONFIGS[a.config]; arrs=load_ckpt(os.path.join(a.ckpt_dir,C["ckpt"]))
    np.random.seed(7); img=(np.random.randn(3,64,64).astype("<f4"))*0.15
    img.tofile("/tmp/_val.bin")
    run=subprocess.run([a.bin,"/tmp/_val.bin"],capture_output=True,text=True)
    cp=[float(l.split(":")[1].replace("%",""))/100 for l in run.stdout.splitlines() if "%" in l and ":" in l][:4]
    cp=np.array(cp); orc=oracle(arrs,C["d_model"],C["half"],C["n_layers"],C["patch_dim"],C["seq_len"],img.astype(np.float64))
    ok=len(cp)==4 and cp.argmax()==orc.argmax() and np.max(np.abs(cp-orc))<2e-3
    print(f"{a.config}: oracle {np.round(orc,5)} | C {np.round(cp,5)} | maxdiff {np.max(np.abs(cp-orc)):.2e} | {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)

if __name__=="__main__": main()
