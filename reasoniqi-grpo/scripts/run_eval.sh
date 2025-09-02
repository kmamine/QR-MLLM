#!/usr/bin/env bash
set -e
python -u src/eval/evaluate.py --config configs/config_koniq.yaml --ckpt_dir outputs/grpo_koniq/grpo_ckpt --data_csv data/splits/koniq_test.csv
python -u src/eval/evaluate.py --config configs/config_tid2013.yaml --ckpt_dir outputs/grpo_tid/grpo_ckpt --data_csv data/splits/tid_test.csv
