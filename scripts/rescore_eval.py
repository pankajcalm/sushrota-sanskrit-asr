#!/usr/bin/env python3
"""Char-LM (sa_char6) rescoring/beam-decode over exported eval logits. Sweep alpha,
report micro content-only CER. alpha=0 beam ~= greedy (sanity)."""
import json, glob, re, unicodedata
import numpy as np
from pyctcdecode import build_ctcdecoder
ROOT="/home/ece/BigDisk/Prathosh/ASR"; LOG=f"{ROOT}/data/eval_logits"; LM=f"{ROOT}/lm/sa_char6.bin"
labels=json.load(open(f"{LOG}/labels.json")); refs=json.load(open(f"{LOG}/refs.json"))
files=sorted(glob.glob(f"{LOG}/[0-9]*.npy"))
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
def edist(a,b):
    n,m=len(a),len(b)
    if n==0: return m
    if m==0: return n
    p=list(range(m+1))
    for i in range(1,n+1):
        c=[i]+[0]*m; ai=a[i-1]
        for j in range(1,m+1): c[j]=min(p[j]+1,c[j-1]+1,p[j-1]+(ai!=b[j-1]))
        p=c
    return p[m]
idx=list(range(0,len(files),4))          # ~1/4 subset for a fast directional read
LP=[np.load(files[i]) for i in idx]; refs=[refs[i] for i in idx]
print(f"subset {len(LP)} utts (every 4th of {len(files)})")
for alpha in (0.0, 0.3, 0.6, 1.0, 1.5):
    dec=build_ctcdecoder(labels, kenlm_model_path=(None if alpha==0 else LM), alpha=alpha, beta=0.5)
    ce=cn=0
    for lp,ref in zip(LP,refs):
        h=norm(dec.decode(lp, beam_width=100)); r=norm(ref)
        ce+=edist(h.replace(" ",""), r.replace(" ","")); cn+=len(r.replace(" ",""))
    print(f"  alpha={alpha:<4} CER {100*ce/max(1,cn):.2f}%", flush=True)
