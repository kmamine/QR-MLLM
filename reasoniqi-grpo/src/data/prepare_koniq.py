import pandas as pd, os, argparse
from src.utils.bins import Binner, BinSpec

def main(args):
    df = pd.read_csv(args.input_csv)  # expects columns: image_name, MOS, SD
    df["path"] = df["image_name"].apply(lambda x: os.path.join(args.image_root, x))
    binner = Binner(BinSpec(args.min_mos, args.max_mos, args.step))
    df["bin_id"] = df["MOS"].apply(binner.to_class)
    df["bin_mid"] = df["bin_id"].apply(binner.to_mos)
    # basic split: stratify by bin_id
    train = df.sample(frac=0.8, random_state=42)
    remain = df.drop(train.index)
    val = remain.sample(frac=0.5, random_state=42)
    test = remain.drop(val.index)
    os.makedirs(os.path.dirname(args.out_meta), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    df.to_csv(args.out_meta, index=False)
    train.to_csv(args.out_train, index=False)
    val.to_csv(args.out_val, index=False)
    test.to_csv(args.out_test, index=False)
    print("Saved:", args.out_meta)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--min_mos", type=float, default=1.0)
    ap.add_argument("--max_mos", type=float, default=5.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--out_meta", default="data/processed/koniq10k_meta.csv")
    ap.add_argument("--out_train", default="data/splits/koniq_train.csv")
    ap.add_argument("--out_val", default="data/splits/koniq_val.csv")
    ap.add_argument("--out_test", default="data/splits/koniq_test.csv")
    main(ap.parse_args())
