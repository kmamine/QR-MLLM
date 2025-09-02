#!/usr/bin/env bash
set -e
python -u src/train/sft_train.py --config configs/config_koniq.yaml --data_csv data/splits/koniq_train.csv --out_dir outputs/sft_koniq
python -u src/train/sft_train.py --config configs/config_tid2013.yaml --data_csv data/splits/tid_train.csv --out_dir outputs/sft_tid
