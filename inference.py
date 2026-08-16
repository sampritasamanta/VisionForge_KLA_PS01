import argparse
from pathlib import Path
import numpy as np,torch
from models.model import ResidualUNet

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input_dir",required=True); p.add_argument("--output_dir",required=True); p.add_argument("--weights",required=True)
    a=p.parse_args(); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(a.weights,map_location=dev); m=ResidualUNet(ck.get("base_channels",32),ck.get("num_res_blocks",4))
    m.load_state_dict(ck["model_state_dict"]); m.to(dev).eval()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    fs=sorted(Path(a.input_dir).glob("*.npy"))
    if not fs: raise RuntimeError("No .npy files found.")
    with torch.no_grad():
        for f in fs:
            x=np.load(f).astype(np.float32)
            if x.ndim!=2: raise ValueError(f"Expected 2-D array: {f}")
            y=m(torch.from_numpy(x)[None,None].to(dev)).squeeze().cpu().numpy().astype(np.float32)
            np.save(out/f.name,y)
    print(f"Processed {len(fs)} files.")
if __name__=="__main__": main()
