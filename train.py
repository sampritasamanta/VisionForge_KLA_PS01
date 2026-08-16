import argparse,os,random
import numpy as np,torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
from data.dataset import PairedNPYDataset
from models.model import ResidualUNet

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--noisy_dir",required=True); p.add_argument("--gt_dir",required=True)
    p.add_argument("--output_dir",default="./weights"); p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--batch_size",type=int,default=8); p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--base_channels",type=int,default=32); p.add_argument("--num_res_blocks",type=int,default=4)
    a=p.parse_args(); os.makedirs(a.output_dir,exist_ok=True)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:",dev)
    ds=PairedNPYDataset(a.noisy_dir,a.gt_dir)
    nv=max(1,int(.1*len(ds))); tr,va=random_split(ds,[len(ds)-nv,nv],generator=torch.Generator().manual_seed(42))
    tl=DataLoader(tr,batch_size=1,shuffle=True); vl=DataLoader(va,batch_size=1)
    m=ResidualUNet(a.base_channels,a.num_res_blocks).to(dev); opt=torch.optim.Adam(m.parameters(),lr=a.lr); loss_fn=nn.L1Loss()
    best=float("inf")
    for ep in range(1,a.epochs+1):
        m.train(); s=0
        for b in tqdm(tl,desc=f"Train {ep}",leave=False):
            x,y=b["input"].to(dev),b["target"].to(dev); opt.zero_grad()
            pred=m(x,target_size=y.shape[-2:]); loss=loss_fn(pred,y); loss.backward(); opt.step(); s+=loss.item()
        m.eval(); v=0
        with torch.no_grad():
            for b in vl:
                x,y=b["input"].to(dev),b["target"].to(dev)
                v+=loss_fn(m(x,target_size=y.shape[-2:]),y).item()
        v/=len(vl); print(f"epoch={ep} train={s/len(tl):.6f} val={v:.6f}")
        if v<best:
            best=v; torch.save({"model_state_dict":m.state_dict(),"base_channels":a.base_channels,"num_res_blocks":a.num_res_blocks},os.path.join(a.output_dir,"final_model.pt"))
if __name__=="__main__": main()
