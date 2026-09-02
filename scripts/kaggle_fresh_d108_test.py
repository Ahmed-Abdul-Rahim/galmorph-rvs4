# =====================================================================
#  KAGGLE — FRESH NOTEBOOK, self-contained d108 accuracy + agreement test
#  Setup:  right panel -> Add Input -> your s4d-108 checkpoint dataset
#          (the file linear_d108_seed2_best).  Enable Internet (for the
#          galaxy dataset download).  Paste this whole thing in one cell, Run.
# =====================================================================
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q",
    "git+https://github.com/mwalmsley/galaxy_mnist.git","einops"])

import os, glob, math, zipfile, tempfile, numpy as np, torch
import torch.nn as nn
from einops import repeat
from galaxy_mnist import GalaxyMNIST

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------ model definition (from the training notebook) ------------------
class HilbertScan(nn.Module):
    def __init__(self, image_size=64, patch_size=1):
        super().__init__()
        self.image_size=image_size; self.patch_size=patch_size
        self.grid_size=image_size//patch_size; self.num_patches=self.grid_size**2
        self.register_buffer("indices", self._get_hilbert_indices(self.grid_size))
    @staticmethod
    def _rot(s,x,y,rx,ry):
        if ry==0:
            if rx==1: x=s-1-x; y=s-1-y
            x,y=y,x
        return x,y
    def _d2xy(self,n,d):
        x=y=0; t=d; s=1
        while s<n:
            rx=(t//2)&1; ry=(t^rx)&1; x,y=self._rot(s,x,y,rx,ry); x+=s*rx; y+=s*ry; t//=4; s*=2
        return x,y
    def _get_hilbert_indices(self,g):
        return torch.LongTensor([self._d2xy(g,d)[1]*g+self._d2xy(g,d)[0] for d in range(g*g)])
    def forward(self,x):
        B,C,H,W=x.shape; p=self.patch_size
        pt=x.unfold(2,p,p).unfold(3,p,p).permute(0,2,3,1,4,5).contiguous().view(B,self.num_patches,C*p*p)
        return pt[:,self.indices,:]
class TakeLastTimestep(nn.Module):
    def forward(self,x): return x[:,-1,:]
class S4DConv(nn.Module):
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1, transposed=True, lr=None):
        super().__init__(); self.h=d_model; self.n=d_state; self.transposed=transposed
        log_dt=torch.rand(self.h)*(math.log(dt_max)-math.log(dt_min))+math.log(dt_min)
        log_A_real=torch.log(0.5*torch.ones(self.h,self.n//2))
        A_imag=math.pi*repeat(torch.arange(self.n//2),'n -> h n',h=self.h)
        C_init=torch.randn(self.h,self.n//2,dtype=torch.cfloat)
        self.register("log_dt",log_dt,lr); self.register("log_A_real",log_A_real,lr); self.register("A_imag",A_imag,lr)
        self.C=nn.Parameter(torch.view_as_real(C_init)); self.D=nn.Parameter(torch.randn(self.h))
    def register(self,name,tensor,lr=None):
        if lr==0.0: self.register_buffer(name,tensor)
        else:
            self.register_parameter(name,nn.Parameter(tensor)); opt={"weight_decay":0.0}
            if lr is not None: opt["lr"]=lr
            setattr(getattr(self,name),"_optim",opt)
    def forward(self,u):
        if not self.transposed: u=u.transpose(-1,-2)
        L=u.size(-1); dt=torch.exp(self.log_dt); C=torch.view_as_complex(self.C)
        A=-torch.exp(self.log_A_real)+1j*self.A_imag; dtA=A*dt.unsqueeze(-1)
        K=torch.exp(dtA.unsqueeze(-1)*torch.arange(L,device=u.device))
        Ct=C*(torch.exp(dtA)-1.)/A; k=2*torch.einsum('hn,hnl->hl',Ct,K).real
        kf=torch.fft.rfft(k,n=2*L); uf=torch.fft.rfft(u,n=2*L)
        y=torch.fft.irfft(uf*kf,n=2*L)[...,:L]+u*self.D.unsqueeze(-1)
        if not self.transposed: y=y.transpose(-1,-2)
        return y,None
class ConvPatchStem(nn.Module):
    def __init__(self,in_channels,d_model,patch_size):
        super().__init__(); mid=max(in_channels*8,32)
        self.net=nn.Sequential(nn.Conv2d(in_channels,mid,3,1,1),nn.GELU(),nn.Conv2d(mid,d_model,patch_size,patch_size))
    def forward(self,x): return self.net(x)
class GalaxyClassifierS4DFast(nn.Module):
    def __init__(self,s4_state=64,d_model=64,num_classes=4,colored=True,num_layers=2,patch_size=1,
                 pooling="last",use_norm=False,use_residual=False,dropout=0.0,patch_embed="linear"):
        super().__init__(); self.hilbert_channels=3 if colored else 1
        self.patch_size=patch_size; self.pooling=pooling; self.use_norm=use_norm
        self.use_residual=use_residual; self.patch_embed=patch_embed
        if patch_embed=="linear":
            self.hilbert_scan=HilbertScan(64,patch_size)
            self.uproject=nn.Linear(self.hilbert_channels*patch_size*patch_size,d_model); self.conv_stem=None
        else:
            self.conv_stem=ConvPatchStem(self.hilbert_channels,d_model,patch_size)
            self.hilbert_scan=HilbertScan(64//patch_size,1); self.uproject=nn.Identity()
        self.s4_layers=nn.ModuleList([S4DConv(d_model=d_model,d_state=s4_state,transposed=False) for _ in range(num_layers)])
        self.acts=nn.ModuleList([nn.GELU() for _ in range(num_layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)]) if use_norm else None
        self.drop=nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.take_last=TakeLastTimestep() if pooling=="last" else None
        self.fc=nn.Linear(d_model,num_classes); self.softmax=nn.Softmax(dim=-1)
    def forward(self,x,return_logits=True):
        if self.patch_embed=="conv": h=self.uproject(self.hilbert_scan(self.conv_stem(x)))
        else: h=self.uproject(self.hilbert_scan(x))
        for i,(s4,act) in enumerate(zip(self.s4_layers,self.acts)):
            res=h; hin=self.norms[i](h) if self.use_norm else h
            ho,_=s4(hin); ho=self.drop(act(ho)); h=res+ho if self.use_residual else ho
        pooled=h.mean(1) if self.take_last is None else self.take_last(h)
        lg=self.fc(pooled); return lg if return_logits else self.softmax(lg)

# ------------------ load checkpoint from /kaggle/input ------------------
def load_state():
    for f in glob.glob('/kaggle/input/**/*',recursive=True):
        if os.path.isfile(f) and 'linear_d108' in os.path.basename(f):
            try: return torch.load(f,map_location='cpu')
            except Exception: pass
    for d in glob.glob('/kaggle/input/**/data.pkl',recursive=True):
        root=os.path.dirname(d); base=os.path.dirname(root); z=tempfile.mktemp(suffix='.zip')
        with zipfile.ZipFile(z,'w') as zf:
            for r,_,fs in os.walk(root):
                for fn in fs: zf.write(os.path.join(r,fn),os.path.relpath(os.path.join(r,fn),base))
        return torch.load(z,map_location='cpu')
    raise FileNotFoundError("no d108 checkpoint under /kaggle/input -- Add Input -> your dataset")
state=load_state(); print("checkpoint loaded:",len(state),"tensors")

model=GalaxyClassifierS4DFast(s4_state=108,d_model=108,num_classes=4,colored=True,
        num_layers=3,patch_size=4,pooling="last",patch_embed="linear").to(DEVICE).eval()
model.load_state_dict(state); print("model built + weights loaded")

# ------------------ test data (canonical GalaxyMNIST test split) ------------------
test=GalaxyMNIST(root='/kaggle/working/gmnist',download=True,train=False)
Xraw=test.data.float(); y=test.targets.numpy(); N=len(y); print("test samples:",N)

# auto-detect the preprocessing the model was trained with (highest subset accuracy)
cands={'div255':Xraw/255.0,'raw':Xraw,'[-1,1]':Xraw/127.5-1.0,
       'standardize':(Xraw-Xraw.mean())/(Xraw.std()+1e-6)}
best=None
for nm,Xc in cands.items():
    with torch.no_grad():
        pr=model(Xc[:400].to(DEVICE)).argmax(-1).cpu().numpy()
    acc=(pr==y[:400]).mean(); print(f"  preproc {nm:11s}: subset acc {acc*100:5.1f}%")
    if best is None or acc>best[1]: best=(nm,acc,Xc)
print("--> using preprocessing:",best[0]); imgs=best[2].numpy()

# ------------------ C model's math (NumPy) ------------------
sd={k:v.detach().cpu().numpy() for k,v in state.items()}
idx=sd['hilbert_scan.indices'].astype(int); up_w=sd['uproject.weight']; up_b=sd['uproject.bias']
Ls=[(sd[f's4_layers.{L}.log_dt'],sd[f's4_layers.{L}.log_A_real'],sd[f's4_layers.{L}.A_imag'],
     sd[f's4_layers.{L}.C'],sd[f's4_layers.{L}.D']) for L in range(3)]
fc_w=sd['fc.weight']; fc_b=sd['fc.bias']
seq=np.zeros((N,256,48),np.float64)
for t in range(256):
    g=int(idx[t]); gh,gw=g//16,g%16
    for c in range(3):
        for pr in range(4):
            for pc in range(4): seq[:,t,c*16+pr*4+pc]=imgs[:,c,gh*4+pr,gw*4+pc]
x=seq@up_w.T+up_b
def gelu(z): k=np.sqrt(2/np.pi); return 0.5*z*(1+np.tanh(k*(z+0.044715*z**3)))
def s4d(inp,ldt,lar,ai,C,D):
    dt=np.exp(ldt); lam=(-np.exp(lar))+1j*ai; Abar=np.exp(lam*dt[:,None]); Bbar=(Abar-1)/lam
    Cbar=2*(C[...,0]+1j*C[...,1])*Bbar; B,Ln,H=inp.shape; out=np.empty_like(inp)
    st=np.zeros((B,H,ai.shape[1]),complex)
    for t in range(Ln):
        u=inp[:,t,:]; st=Abar*st+u[...,None]; out[:,t,:]=D*u+np.real(np.sum(Cbar*st,axis=2))
    return out
for (ldt,lar,ai,C,D) in Ls: x=gelu(s4d(x,ldt,lar,ai,C,D))
c_pred=(x[:,-1,:]@fc_w.T+fc_b).argmax(1)

tp=[]
with torch.no_grad():
    for i in range(0,N,64): tp.append(model(torch.tensor(imgs[i:i+64]).float().to(DEVICE)).argmax(-1).cpu().numpy())
t_pred=np.concatenate(tp)

print("\n==================  RESULTS  ==================")
print(f"PyTorch accuracy        : {(t_pred==y).mean()*100:.2f}%")
print(f"C-model (NumPy) accuracy: {(c_pred==y).mean()*100:.2f}%")
print(f"C vs PyTorch agreement  : {(c_pred==t_pred).mean()*100:.2f}%   (want ~100%)")
