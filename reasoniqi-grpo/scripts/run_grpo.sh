#!/usr/bin/env bash
set -e
python -u src/train/grpo_train.py --config configs/config_koniq.yaml --sft_ckpt outputs/sft_koniq/sft_ckpt --data_csv data/splits/koniq_train.csv --out_dir outputs/grpo_koniq
python -u src/train/grpo_train.py --config configs/config_tid2013.yaml --sft_ckpt outputs/sft_tid/sft_ckpt --data_csv data/splits/tid_train.csv --out_dir outputs/grpo_tid
