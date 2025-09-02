import torch, pandas as pd
from PIL import Image
from torchvision import transforms as T

class NRDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_size=512, sigma_source="per_image"):
        self.df = pd.read_csv(csv_path)
        self.tr = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])
        self.sigma_source = sigma_source
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        x = self.tr(img)
        mos = float(row["MOS"])
        sd = float(row["SD"]) if "SD" in row and not pd.isna(row["SD"]) else None
        sigma = sd if self.sigma_source=="per_image" and sd is not None else float(row.get("SIGMA", 0.5))
        bin_id = int(row["bin_id"])
        bin_mid = float(row["bin_mid"])
        return {"pixel_values": x, "mos": mos, "sigma": sigma, "bin_id": bin_id, "bin_mid": bin_mid, "path": row["path"]}
