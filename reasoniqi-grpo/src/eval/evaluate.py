import yaml, torch, numpy as np, pandas as pd
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from src.data.datasets import NRDataset
from src.eval.metrics import classification_metrics, window_hit, plcc_srocc
from src.utils.bins import Binner, BinSpec

@torch.no_grad()
def predict_bins_and_mos(model, processor, ds, binner, T=8):
    y_true_cls, y_pred_cls, mos_true, mos_pred, sigma = [], [], [], [], []
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    for batch in dl:
        sigma.append(float(batch["sigma"]))
        mos_true.append(float(batch["mos"]))
        # Placeholder: random predicts — replace with actual inference/parsing of FinalBin
        pred_cls = int(batch["bin_id"])  # stub to be replaced
        y_true_cls.append(int(batch["bin_id"]))
        y_pred_cls.append(pred_cls)
        # MC MOS
        mos_pred.append(float(batch["bin_mid"]))
    return np.array(y_true_cls), np.array(y_pred_cls), np.array(mos_true), np.array(mos_pred), np.array(sigma)

def main(config_path, ckpt_dir, data_csv):
    cfg = yaml.safe_load(open(config_path))
    binner = Binner(BinSpec(cfg["bins"]["min"], cfg["bins"]["max"], cfg["bins"]["step"]))
    ds = NRDataset(data_csv, image_size=cfg["train"]["image_size"], sigma_source=cfg["sigma_source"])
    bnb = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, quantization_config=bnb, device_map="auto")
    processor = AutoProcessor.from_pretrained(ckpt_dir)
    y_true_cls, y_pred_cls, mos_true, mos_pred, sigma = predict_bins_and_mos(model, processor, ds, binner, T=8)
    cls = classification_metrics(y_true_cls, y_pred_cls)
    wh = window_hit(mos_true, mos_pred, sigma)
    reg = plcc_srocc(mos_true, mos_pred)
    print({"classification": cls, "window_hit": wh, "regression": reg})

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--data_csv", required=True)
    main(**vars(ap.parse_args()))
