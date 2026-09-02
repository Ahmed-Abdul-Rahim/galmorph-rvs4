# ==== paste this as a NEW CELL at the END of the s4d-108d notebook and run it ====
# It dumps the real test images + labels + PyTorch predictions for the C to check.
import os, csv, numpy as np, torch

model   = best_repeat_d108_model     # <-- the loaded d108 model (adjust name if different)
ds      = test_dataset               # <-- the augment=False test set from cell 3
device  = DEVICE
outdir  = "test_d108"; os.makedirs(outdir, exist_ok=True)

model.eval(); rows=[]
with torch.no_grad():
    for i in range(len(ds)):
        img, lab = ds[i]                              # img: (3,64,64) preprocessed tensor
        img = img.float()
        probs = model(img.unsqueeze(0).to(device))[0].detach().cpu().numpy().astype('<f4')
        pred  = int(probs.argmax())
        np.ascontiguousarray(img.cpu().numpy(), dtype='<f4').tofile(f"{outdir}/img_{i}.bin")  # [c][y][x]
        probs.tofile(f"{outdir}/soft_{i}.bin")
        rows.append((i, int(lab), pred))
with open(f"{outdir}/labels.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["idx","true","torch_pred"]); w.writerows(rows)
a=np.array(rows)
print(f"exported {len(rows)} test samples to {outdir}/")
print(f"PyTorch test accuracy: {(a[:,1]==a[:,2]).mean()*100:.2f}%")
# ==== then: copy the test_d108/ folder next to build/main and run accuracy_d108.py ====
