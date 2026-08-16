import argparse,csv
from pathlib import Path
import numpy as np
from skimage.metrics import peak_signal_noise_ratio,structural_similarity

p=argparse.ArgumentParser(); p.add_argument("--pred_dir",required=True); p.add_argument("--gt_dir",required=True); p.add_argument("--csv_path",default="./results/metrics.csv"); a=p.parse_args()
rows=[]
for f in sorted(Path(a.pred_dir).glob("*.npy")):
    g=Path(a.gt_dir)/f.name
    if not g.exists(): continue
    x=np.load(f).astype(np.float32); y=np.load(g).astype(np.float32)
    if x.shape!=y.shape: continue
    dr=float(y.max()-y.min()) or 1.0
    rows.append((f.name,peak_signal_noise_ratio(y,x,data_range=dr),structural_similarity(y,x,data_range=dr)))
Path(a.csv_path).parent.mkdir(parents=True,exist_ok=True)
with open(a.csv_path,"w",newline="") as h:
    w=csv.writer(h); w.writerow(["file","PSNR","SSIM"]); w.writerows(rows)
print("Mean PSNR:",np.mean([r[1] for r in rows]) if rows else "N/A")
print("Mean SSIM:",np.mean([r[2] for r in rows]) if rows else "N/A")
