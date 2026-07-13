#!/usr/bin/env python3
"""Finetune Whisper-medium on our Sanskrit train set — the encoder-decoder
external comparison. Full FT, Devanagari targets, language=sanskrit."""
import json, argparse, os
from dataclasses import dataclass
import torch, soundfile as sf
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
ROOT="/home/ece/BigDisk/Prathosh/ASR"
ap=argparse.ArgumentParser()
ap.add_argument("--train", default=f"{ROOT}/data/train_ft5.jsonl")
ap.add_argument("--base", default="openai/whisper-medium")
ap.add_argument("--outdir", default=f"{ROOT}/exp/whisper_med_v5")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--grad-accum", type=int, default=2)
ap.add_argument("--maxdur", type=float, default=28.0)
ap.add_argument("--smoke", action="store_true")
A=ap.parse_args()
proc=WhisperProcessor.from_pretrained(A.base, language="sanskrit", task="transcribe")
rows=[json.loads(l) for l in open(A.train)]
rows=[r for r in rows if 0.4<r["duration"]<=A.maxdur]
if A.smoke: rows=rows[:48]
print(f"train utts {len(rows)}")
class DS(torch.utils.data.Dataset):
    def __init__(s, rows): s.rows=rows
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        r=s.rows[i]; x,sr=sf.read(r["audio_filepath"], dtype="float32")
        if x.ndim>1: x=x.mean(1)
        feat=proc.feature_extractor(x, sampling_rate=16000).input_features[0]
        labels=proc.tokenizer(r["text"]).input_ids
        return {"input_features": feat, "labels": labels}
@dataclass
class Collator:
    processor: object
    def __call__(s, feats):
        b=s.processor.feature_extractor.pad([{"input_features":f["input_features"]} for f in feats], return_tensors="pt")
        lb=s.processor.tokenizer.pad([{"input_ids":f["labels"]} for f in feats], return_tensors="pt")
        labels=lb["input_ids"].masked_fill(lb.attention_mask.ne(1), -100)
        if (labels[:,0]==s.processor.tokenizer.bos_token_id).all().cpu().item():
            labels=labels[:,1:]
        b["labels"]=labels
        return b
model=WhisperForConditionalGeneration.from_pretrained(A.base)
model.generation_config.language="sanskrit"; model.generation_config.task="transcribe"
model.generation_config.forced_decoder_ids=None; model.config.forced_decoder_ids=None
model.config.suppress_tokens=[]
targs=Seq2SeqTrainingArguments(output_dir=A.outdir, per_device_train_batch_size=A.bs,
    gradient_accumulation_steps=A.grad_accum, num_train_epochs=(1 if A.smoke else A.epochs),
    max_steps=(10 if A.smoke else -1), learning_rate=A.lr, warmup_steps=(2 if A.smoke else 200),
    bf16=True, save_strategy="epoch", save_total_limit=3, logging_steps=25,
    dataloader_num_workers=4, predict_with_generate=False, remove_unused_columns=False, report_to=[])
Seq2SeqTrainer(model=model, args=targs, train_dataset=DS(rows), data_collator=Collator(proc),
        tokenizer=proc.feature_extractor).train()
proc.save_pretrained(A.outdir); model.save_pretrained(A.outdir)
print("SAVED", A.outdir)
