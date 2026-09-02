#!/usr/bin/env python3
"""Run the C model on the exported test set and report accuracy + PyTorch agreement.
Usage:  (from the folder containing build/main and weights.bin)
        python3 accuracy_d108.py
Needs test_d108/ (from export_testset_cell.py) sitting here."""
import numpy as np, subprocess, csv, os, sys
D="test_d108"
rows=list(csv.DictReader(open(f"{D}/labels.csv")))
N=len(rows); cc=tc=ag=0
for k,r in enumerate(rows):
    i=r["idx"]
    out=subprocess.run(["./build/main", f"{D}/img_{i}.bin"], capture_output=True, text=True).stdout
    p=[]
    for ln in out.splitlines():
        if '%' in ln and ':' in ln:
            try: p.append(float(ln.split(':')[1].replace('%',''))/100)
            except: pass
    if len(p)<4: print("parse fail on", i); continue
    cp=int(np.argmax(p[:4])); true=int(r["true"]); tp=int(r["torch_pred"])
    cc+=cp==true; tc+=tp==true; ag+=cp==tp
    if (k+1)%200==0: print(f"  ...{k+1}/{N}", file=sys.stderr)
print(f"N = {N}")
print(f"PyTorch accuracy         : {tc/N*100:.2f}%")
print(f"C (this repo) accuracy    : {cc/N*100:.2f}%")
print(f"C vs PyTorch agreement    : {ag/N*100:.2f}%   (want ~100%)")
