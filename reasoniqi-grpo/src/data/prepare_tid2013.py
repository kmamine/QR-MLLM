import pandas as pd, os, argparse, numpy as np
from src.utils.bins import Binner, BinSpec

def main(args):
    df = pd.read_csv(args.input_csv)  # expects columns: filename, MOS
    df["path"] = df["filename"].apply(lambda x: os.path.join(args.image_root, x))
    binner = Binner(BinSpec(args.min_mos, args.max_mos, args.step))
    df["bin_id"] = df["MOS"].apply(binner.to_class)
    df["bin_mid"] = df["bin_id"].apply(binner.to_mos)

    # split per distortion type if available; else random split
    train = df.sample(frac=0.8, random_state=42)
    remain = df.drop(train.index)
    val = remain.sample(frac=0.5, random_state=42)
    test = remain.drop(val.index)

    # estimate global sigma if requested
    if args.estimate_sigma:
        sigma = float(train["MOS"].std())
    else:
        sigma = args.sigma_default
    for split in (train, val, test):
        split["SIGMA"] = sigma

    os.makedirs(os.path.dirname(args.out_meta), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    df.to_csv(args.out_meta, index=False)
    train.to_csv(args.out_train, index=False)
    val.to_csv(args.out_val, index=False)
    test.to_csv(args.out_test, index=False)
    print("Saved:", args.out_meta, "Global σ:", sigma)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--min_mos", type=float, default=0.0)
    ap.add_argument("--max_mos", type=float, default=9.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--estimate_sigma", action="store_true")
    ap.add_argument("--sigma_default", type=float, default=0.5)
    ap.add_argument("--out_meta", default="data/processed/tid2013_meta.csv")
    ap.add_argument("--out_train", default="data/splits/tid_train.csv")
    ap.add_argument("--out_val", default="data/splits/tid_val.csv")
    ap.add_argument("--out_test", default="data/splits/tid_test.csv")
    main(ap.parse_args())
