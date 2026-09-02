#!/usr/bin/env python3
"""Convert a linear_d108 checkpoint (.zip / .pth, torch save) into the C
runtime files WITHOUT needing torch installed:
    weights.bin   -- flat blob the C loader reads (indices int32 + float weights)
    weights.h     -- same blob baked in, for the RISC-V/QEMU `make bench` build
    image.h       -- a baked RGB sample image (edit SAMPLE below to use a real one)
Usage:  python3 make_d108_bins.py linear_d108_seed2_best.zip
"""
import sys, os, zipfile, pickle, tempfile, numpy as np

ckpt = sys.argv[1] if len(sys.argv) > 1 else "linear_d108_seed2_best.zip"

# --- read the torch save (a zip of a pickle + raw storage files), no torch ---
tmp = tempfile.mkdtemp()
with zipfile.ZipFile(ckpt) as z: z.extractall(tmp)
root = os.path.join(tmp, os.listdir(tmp)[0])
def rebuild(storage, off, size, stride, *a):
    key, dt = storage; return {'key': key, 'size': tuple(size)}
class Any:
    def __init__(self,*a,**k): pass
    def __setstate__(self,s): pass
def factory(mod,name):
    if name=='_rebuild_tensor_v2': return rebuild
    if name=='OrderedDict':
        import collections; return collections.OrderedDict
    class C(Any): pass
    return C
class U(pickle.Unpickler):
    def find_class(self,m,n): return factory(m,n)
    def persistent_load(self,pid): return (pid[2], None)   # storage key
sd = U(open(os.path.join(root,'data.pkl'),'rb')).load()
def store(name, dtype):
    key = sd[name]['key']
    return np.fromfile(os.path.join(root,'data',str(key)), dtype=dtype)

out = open('weights.bin','wb')
def wf(a): out.write(np.asarray(a,dtype='<f4').tobytes())
def wi(a): out.write(np.asarray(a,dtype='<i4').tobytes())
wi(store('hilbert_scan.indices','<i8').astype('<i4'))          # 256 idx
wf(store('uproject.weight','<f4')); wf(store('uproject.bias','<f4'))
for L in range(3):
    for t in ['log_dt','log_A_real','A_imag','C','D']:
        wf(store(f's4_layers.{L}.{t}','<f4'))
wf(store('fc.weight','<f4')); wf(store('fc.bias','<f4'))
out.close()
nfloats = os.path.getsize('weights.bin')//4
print(f"[*] wrote weights.bin ({nfloats} slots)")
assert nfloats == 76616, f"unexpected size {nfloats} (expected 76616)"

def emit(src, arr, macro, path):
    b=open(src,'rb').read()
    with open(path,'w') as f:
        f.write(f"#ifndef {macro}\n#define {macro}\n")
        f.write(f"static const unsigned char {arr}[{len(b)}] = {{\n")
        for i in range(0,len(b),20):
            f.write("  "+",".join(map(str,b[i:i+20]))+",\n")
        f.write("};\n#endif\n")
    print(f"[*] wrote {path}")
emit('weights.bin','WEIGHTS_BLOB','WEIGHTS_H','weights.h')
# baked sample image: deterministic RGB placeholder (energy/instret are data-independent).
# Replace with a real preprocessed galaxy image for a meaningful prediction.
if not os.path.exists('image.bin'):
    np.random.seed(0); np.random.randn(3,64,64).astype('<f4').tofile('image.bin')
emit('image.bin','SAMPLE_IMAGE','IMAGE_H','image.h')
print("[*] done. Build:  make bench CC=riscv32-unknown-elf-gcc CFLAGS=\"-O2\"")
