#!/usr/bin/env python3
"""Finetune XLSR/wav2vec2 (CTC, char-level) on our Sanskrit train set — the
architecture-matched external comparison to our IndicConformer-CTC model.
Builds char vocab from train text; trains a fresh CTC head on frozen feature encoder."""
import json, os, argparse
from dataclasses import dataclass
import torch, soundfile as sf
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor,
                          Wav2Vec2ForCTC, TrainingArguments, Trainer)
ROOT="/home/ece/BigDisk/Prathosh/ASR"
ap=argparse.ArgumentParser()
ap.add_argument("--train", default=f"{ROOT}/data/train_ft5.jsonl")
ap.add_argument("--base", default="facebook/wav2vec2-xls-r-300m")
ap.add_argument("--outdir", default=f"{ROOT}/exp/w2v2_xlsr_v5")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--grad-accum", type=int, default=2)
ap.add_argument("--maxdur", type=float, default=20.0)
ap.add_argument("--smoke", action="store_true")
A=ap.parse_args()
rows=[json.loads(l) for l in open(A.train)]
rows=[r for r in rows if 0.4<r["duration"]<=A.maxdur]
if A.smoke: rows=rows[:48]
os.makedirs(A.outdir, exist_ok=True)
# --- vocab from train text ---
vocab=set()
for r in rows:
    for ch in r["text"]:
        if ch!=" ": vocab.add(ch)
vd={c:i for i,c in enumerate(sorted(vocab))}
vd["|"]=len(vd); vd["[UNK]"]=len(vd); vd["[PAD]"]=len(vd)
json.dump(vd, open(f"{A.outdir}/vocab.json","w"), ensure_ascii=False)
print(f"vocab size {len(vd)}, train utts {len(rows)}")
tok=Wav2Vec2CTCTokenizer(f"{A.outdir}/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
fe=Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
proc=Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
proc.save_pretrained(A.outdir)
class DS(torch.utils.data.Dataset):
    def __init__(s, rows): s.rows=rows
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        r=s.rows[i]; x,sr=sf.read(r["audio_filepath"], dtype="float32")
        if x.ndim>1: x=x.mean(1)
        iv=proc(x, sampling_rate=16000).input_values[0]
        labels=proc(text=r["text"]).input_ids
        return {"input_values": iv, "labels": labels}
@dataclass
class Collator:
    processor: object
    def __call__(s, feats):
        b=s.processor.pad([{"input_values":f["input_values"]} for f in feats], padding=True, return_tensors="pt")
        lb=s.processor.pad(labels=[{"input_ids":f["labels"]} for f in feats], padding=True, return_tensors="pt")
        b["labels"]=lb["input_ids"].masked_fill(lb["attention_mask"].ne(1), -100)
        return b
model=Wav2Vec2ForCTC.from_pretrained(A.base, ctc_loss_reduction="mean", ctc_zero_infinity=True,
        pad_token_id=proc.tokenizer.pad_token_id, vocab_size=len(vd))
model.freeze_feature_encoder()
targs=TrainingArguments(output_dir=A.outdir, per_device_train_batch_size=A.bs,
    gradient_accumulation_steps=A.grad_accum, num_train_epochs=(1 if A.smoke else A.epochs),
    max_steps=(10 if A.smoke else -1), learning_rate=A.lr, warmup_steps=(2 if A.smoke else 500),
    bf16=True, save_strategy="epoch", save_total_limit=4, logging_steps=20,
    dataloader_num_workers=4, group_by_length=False, report_to=[])
Trainer(model=model, args=targs, train_dataset=DS(rows), data_collator=Collator(proc),
        tokenizer=proc.feature_extractor).train()
proc.save_pretrained(A.outdir); model.save_pretrained(A.outdir)
print("SAVED", A.outdir)
