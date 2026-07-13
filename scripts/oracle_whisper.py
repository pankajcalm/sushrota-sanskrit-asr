#!/usr/bin/env python3
"""Complementarity: v5 (IndicConformer-CTC) vs Whisper-medium-ft on chant.
Per-utt CER each, oracle (per-utt min) = fusion ceiling, and who-rescues-whom."""
import json, glob, re, unicodedata
import numpy as np, torch, soundfile as sf
ROOT="/home/ece/BigDisk/Prathosh/ASR"; LOG=f"{ROOT}/data/eval_logits"
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip().replace(' ','')
def ed(a,b):
    n,m=len(a),len(b)
    if n==0:return m
    if m==0:return n
    p=list(range(m+1))
    for i in range(1,n+1):
        c=[i]+[0]*m;ai=a[i-1]
        for j in range(1,m+1):c[j]=min(p[j]+1,c[j-1]+1,p[j-1]+(ai!=b[j-1]))
        p=c
    return p[m]
labels=json.load(open(f"{LOG}/labels.json")); refs=json.load(open(f"{LOG}/refs.json"))
files=sorted(glob.glob(f"{LOG}/[0-9]*.npy")); BLANK=len(labels)-1
def v5greedy(lp):
    ids=lp.argmax(-1); out=[]; prev=-1
    for i in ids:
        if i!=prev and i!=BLANK: out.append(labels[i])
        prev=i
    return ''.join(out).replace('▁',' ').strip()
man=[json.loads(l) for l in open(f"{ROOT}/data/utts_fa/manifest_fa_eval_sk10.jsonl")]
assert len(man)==len(files)==len(refs)
from transformers import pipeline
asr=pipeline("automatic-speech-recognition", model=f"{ROOT}/exp/whisper_med_v5", device=0,
             torch_dtype=torch.float16, chunk_length_s=30)
gkw=dict(language="sanskrit", task="transcribe")
def load(fp):
    x,sr=sf.read(fp, dtype='float32')
    if x.ndim>1: x=x.mean(1)
    return x
wh=[]
for i in range(0,len(man),16):
    out=asr([load(r["audio_filepath"]) for r in man[i:i+16]], batch_size=16, generate_kwargs=gkw)
    wh+=[o["text"] for o in out]
e5=ew=eo=tn=0; v5win=whwin=tie=both0=0
for r,ref,f,h in zip(man,refs,files,wh):
    rc=norm(ref); v=norm(v5greedy(np.load(f))); w=norm(h)
    d5=ed(v,rc); dw=ed(w,rc); L=len(rc)
    e5+=d5; ew+=dw; eo+=min(d5,dw); tn+=L
    if d5==0 and dw==0: both0+=1
    elif d5<dw: v5win+=1
    elif dw<d5: whwin+=1
    else: tie+=1
N=len(man)
print(f"== v5(CTC) vs Whisper-med-ft complementarity on chant ({N} utts) ==")
print(f"  v5 (IndicConformer-CTC)  CER : {100*e5/tn:.2f}%")
print(f"  Whisper-medium-ft        CER : {100*ew/tn:.2f}%")
print(f"  ORACLE (per-utt min)     CER : {100*eo/tn:.2f}%   <-- best any v5+Whisper fusion could do")
print(f"  headroom vs v5               : {100*(e5-eo)/tn:.2f} CER points ({100*(e5-eo)/e5:.1f}% rel)")
print(f"  utts both perfect            : {both0}")
print(f"  utts v5 better               : {v5win}")
print(f"  utts Whisper rescues (better): {whwin}")
print(f"  ties                         : {tie}")
