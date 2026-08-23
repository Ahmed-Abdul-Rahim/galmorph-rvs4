# ==================================================================
#  KAGGLE CELL — d108 accuracy test, checkpoint loaded from an INPUT
#  data source.  (Right panel -> Add Input -> your d108 checkpoint dataset.)
#  Run this AFTER the notebook's cells that define the model classes,
#  test_dataset, and DEVICE.  Paste into a new cell in the browser, then Run.
# ==================================================================
import os, glob, zipfile, tempfile, numpy as np, torch

# ---------- 1. find + load the checkpoint from /kaggle/input ----------
def load_state():
    for f in glob.glob('/kaggle/input/**/*', recursive=True):
        if os.path.isfile(f) and 'linear_d108' in os.path.basename(f):
            try: return torch.load(f, map_location='cpu')
            except Exception: pass
    # fallback: Kaggle unzipped the torch-save; re-zip the folder holding data.pkl
    for d in glob.glob('/kaggle/input/**/data.pkl', recursive=True):
        root = os.path.dirname(d); base = os.path.dirname(root)
        z = tempfile.mktemp(suffix='.zip')
        with zipfile.ZipFile(z, 'w') as zf:
            for r, _, fs in os.walk(root):
                for fn in fs:
                    fp = os.path.join(r, fn)
                    zf.write(fp, os.path.relpath(fp, base))
        return torch.load(z, map_location='cpu')
    raise FileNotFoundError("no d108 checkpoint under /kaggle/input -- Add Input -> your dataset")

state = load_state()
print("checkpoint loaded:", len(state), "tensors")

# ---------- 2. rebuild the d108 model + load weights (PyTorch reference) ----------
model = GalaxyClassifierS4DFast(s4_state=108, d_model=108, num_classes=4, colored=True,
                                num_layers=3, patch_size=4, pooling="last",
                                patch_embed="linear").to(DEVICE).eval()
model.load_state_dict(state)
print("model built + weights loaded OK")

# ---------- 3. weights as numpy (the C model's math) ----------
sd   = {k: v.detach().cpu().numpy() for k, v in state.items()}
idx  = sd['hilbert_scan.indices'].astype(int)
up_w = sd['uproject.weight']; up_b = sd['uproject.bias']
Ls   = [(sd[f's4_layers.{L}.log_dt'], sd[f's4_layers.{L}.log_A_real'],
         sd[f's4_layers.{L}.A_imag'], sd[f's4_layers.{L}.C'], sd[f's4_layers.{L}.D'])
        for L in range(3)]
fc_w = sd['fc.weight']; fc_b = sd['fc.bias']

# ---------- 4. test set (from the notebook's test_dataset) ----------
ds   = test_dataset
imgs = torch.stack([ds[i][0].float() for i in range(len(ds))]).numpy()   # (N,3,64,64)
labs = np.array([int(ds[i][1]) for i in range(len(ds))])
N = len(imgs); print("test samples:", N)

# patchify + hilbert -> (N,256,48)  (matches the C hilbert_scan)
seq = np.zeros((N, 256, 48), np.float64)
for t in range(256):
    g = int(idx[t]); gh, gw = g // 16, g % 16
    for c in range(3):
        for pr in range(4):
            for pc in range(4):
                seq[:, t, c*16 + pr*4 + pc] = imgs[:, c, gh*4+pr, gw*4+pc]
x = seq @ up_w.T + up_b

def gelu(z):
    k = np.sqrt(2/np.pi); return 0.5*z*(1 + np.tanh(k*(z + 0.044715*z**3)))
def s4d(inp, ldt, lar, ai, C, D):
    dt = np.exp(ldt); lam = (-np.exp(lar)) + 1j*ai
    Abar = np.exp(lam*dt[:, None]); Bbar = (Abar - 1)/lam
    Cbar = 2*(C[..., 0] + 1j*C[..., 1])*Bbar
    B, Ln, H = inp.shape; out = np.empty_like(inp)
    st = np.zeros((B, H, ai.shape[1]), complex)
    for t in range(Ln):
        u = inp[:, t, :]
        st = Abar*st + u[..., None]
        out[:, t, :] = D*u + np.real(np.sum(Cbar*st, axis=2))
    return out
for (ldt, lar, ai, C, D) in Ls:
    x = gelu(s4d(x, ldt, lar, ai, C, D))
c_pred = (x[:, -1, :] @ fc_w.T + fc_b).argmax(1)

# ---------- 5. PyTorch predictions ----------
tp = []
with torch.no_grad():
    for i in range(0, N, 64):
        tp.append(model(torch.tensor(imgs[i:i+64]).float().to(DEVICE)).argmax(-1).cpu().numpy())
t_pred = np.concatenate(tp)

print("\n==================  RESULTS  ==================")
print(f"PyTorch accuracy        : {(t_pred==labs).mean()*100:.2f}%")
print(f"C-model (NumPy) accuracy: {(c_pred==labs).mean()*100:.2f}%")
print(f"C vs PyTorch agreement  : {(c_pred==t_pred).mean()*100:.2f}%   (want ~100%)")
