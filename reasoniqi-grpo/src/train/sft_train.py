import os, math, json, yaml, torch, pandas as pd, numpy as np
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from src.data.datasets import NRDataset
from src.utils.bins import Binner, BinSpec
from src.utils.seed import set_seed
from src.utils.logging import get_logger

logger = get_logger()

def build_model(cfg_model):
    quant = cfg_model.get("quant","int4")
    bnb = BitsAndBytesConfig(load_in_4bit=True) if quant=="int4" else None
    model = AutoModelForCausalLM.from_pretrained(cfg_model["name"], quantization_config=bnb, torch_dtype=getattr(torch, cfg_model.get("dtype","bfloat16")))
    peft_cfg = LoraConfig(r=cfg_model["lora"]["r"], lora_alpha=cfg_model["lora"]["alpha"], lora_dropout=cfg_model["lora"]["dropout"], target_modules=cfg_model["lora"]["target_modules"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_cfg)
    processor = AutoProcessor.from_pretrained(cfg_model["name"])
    return model, processor

def collate_fn(batch, processor, binner):
    images = [b["pixel_values"] for b in batch]
    # Build text with targets
    texts = []
    labels = []
    for b in batch:
        bin_mid = b["bin_mid"]
        low = bin_mid - binner.step/2 if hasattr(binner,"step") else bin_mid - 0.25
        high = bin_mid + binner.step/2 if hasattr(binner,"step") else bin_mid + 0.25
        text = ("You are an image quality expert.\n"
                "Briefly describe the dominant degradations in one sentence.\n"
                "Then pick ONE quality bin in steps of 0.5.\n\n"
                "[reasoning]\n"
                f"FinalBin: {low:.1f}-{high:.1f}")
        texts.append(text)
        labels.append(b["bin_id"])
    enc = processor(text=texts, images=images, return_tensors="pt", padding=True)
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    return enc

def main(config_path, data_csv, out_dir):
    set_seed(42)
    os.makedirs(out_dir, exist_ok=True)
    cfg = yaml.safe_load(open(config_path))
    model, processor = build_model(yaml.safe_load(open("configs/model_qwen3b.yaml")))
    ds = NRDataset(data_csv, image_size=cfg["train"]["image_size"], sigma_source=cfg["sigma_source"])
    binner = Binner(BinSpec(cfg["bins"]["min"], cfg["bins"]["max"], cfg["bins"]["step"]))
    # Trainer expects a dataset object; for brevity we'll wrap collator in lambda
    args = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=cfg["train"]["batch_size"], num_train_epochs=cfg["train"]["epochs_sft"], learning_rate=cfg["train"]["lr_sft"], logging_steps=50, save_steps=1000, fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=lambda b: collate_fn(b, processor, binner))
    trainer.train()
    model.save_pretrained(os.path.join(out_dir,"sft_ckpt"))
    processor.save_pretrained(os.path.join(out_dir,"sft_ckpt"))
    logger.info("SFT training done.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--out_dir", default="outputs/sft")
    main(**vars(ap.parse_args()))
