#!/usr/bin/env python3
"""Bake a trained S4D checkpoint into the C runtime headers, WITHOUT torch.

Writes, in the current directory:
  weights.bin  flat blob the non-baked C reads (int32 indices + float weights)
  weights.h    same blob baked in, for the RISC-V / QEMU / gem5 build (-DBAKED)
  image.h      a deterministic sample image baked in (energy is data independent)

Usage:
  python3 bake_weights.py --config d108_seq256 [--ckpt-dir ../../checkpoints]
Configs: d64_native, d64_seq256, d108_seq256
"""
import os, sys, argparse, zipfile, pickle, tempfile, numpy as np

CONFIGS = {
  "d64_native":  dict(d_model=64,  half=32, n_layers=2, seq_len=4096, patch_dim=3,
                      ckpt="d64_seq4096__main__seed30485.pt"),
  "d64_seq256":  dict(d_model=64,  half=32, n_layers=2, seq_len=256,  patch_dim=48,
                      ckpt="d64_seq256__main__seed30485.pt"),
  "d108_seq256": dict(d_model=108, half=54, n_layers=3, seq_len=256,  patch_dim=48,
                      ckpt="d108_seq256__main__seed30485.pt"),
}

def load_ckpt(pt):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(pt) as z: z.extractall(tmp)
    root = dpkl = None
    for dp,_,fs in os.walk(tmp):
        if "data.pkl" in fs: dpkl=os.path.join(dp,"data.pkl"); root=dp; break
    def rebuild(storage,off,size,stride,*a): return {"key":storage[0],"off":off,"size":tuple(size)}
    class Any:
        def __init__(self,*a,**k): pass
        def __setstate__(self,s): pass
    def factory(m,n):
        if n=="_rebuild_tensor_v2": return rebuild
        if n=="OrderedDict":
            import collections; return collections.OrderedDict
        class C(Any): pass
        return C
    class U(pickle.Unpickler):
        def find_class(self,m,n): return factory(m,n)
        def persistent_load(self,pid): return (pid[2], pid[1])
    sd = U(open(dpkl,"rb")).load()
    arrs={}
    for k,v in sd.items():
        key=v["key"]; size=v["size"]; off=v["off"]
        dt="<i8" if "indices" in k else "<f4"
        raw=np.fromfile(os.path.join(root,"data",str(key)),dtype=dt)
        n=int(np.prod(size)) if size else 1
        arrs[k]=raw[off:off+n].reshape(size) if size else raw[off:off+1]
    return arrs

def emit_h(binpath, arr, macro, outpath):
    b=open(binpath,"rb").read()
    with open(outpath,"w") as f:
        f.write(f"#ifndef {macro}\n#define {macro}\n")
        f.write(f"static const unsigned char {arr}[{len(b)}] = {{\n")
        for i in range(0,len(b),20):
            f.write("  "+",".join(map(str,b[i:i+20]))+",\n")
        f.write("};\n#endif\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--ckpt-dir", default="../../checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    a=ap.parse_args()
    C=CONFIGS[a.config]; dm=C["d_model"]; half=C["half"]; nl=C["n_layers"]
    pt=os.path.join(a.ckpt_dir, C["ckpt"])
    if not os.path.exists(pt):
        sys.exit(f"checkpoint not found: {pt}\nRun scripts/fetch_checkpoints.sh first, "
                 f"or drop {C['ckpt']} into {a.ckpt_dir}/")
    arrs=load_ckpt(pt)
    out=open("weights.bin","wb")
    wf=lambda x: out.write(np.asarray(x,dtype="<f4").ravel().tobytes())
    wi=lambda x: out.write(np.asarray(x,dtype="<i4").ravel().tobytes())
    wi(np.asarray(arrs["hilbert_scan.indices"],dtype="<i8").astype("<i4"))
    wf(arrs["uproject.weight"]); wf(arrs["uproject.bias"])
    for L in range(nl):
        for t in ["log_dt","log_A_real","A_imag","C","D"]:
            wf(arrs[f"s4_layers.{L}.{t}"])
    wf(arrs["fc.weight"]); wf(arrs["fc.bias"])
    out.close()
    slots=os.path.getsize("weights.bin")//4
    exp=C["seq_len"]+(dm*C["patch_dim"]+dm)+nl*(dm+dm*half+dm*half+dm*half*2+dm)+(4*dm+4)
    assert slots==exp, f"weight size {slots} != expected {exp}"
    np.random.seed(a.seed); np.random.randn(3,64,64).astype("<f4").tofile("image.bin")
    emit_h("weights.bin","WEIGHTS_BLOB","WEIGHTS_H","weights.h")
    emit_h("image.bin","SAMPLE_IMAGE","IMAGE_H","image.h")
    print(f"[ok] {a.config}: wrote weights.h ({slots} slots) and image.h")

if __name__=="__main__": main()
