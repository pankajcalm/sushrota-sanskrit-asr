#!/usr/bin/env python3
"""Contextual biasing (pyctcdecode hotwords) on ft_ep20 eval logits — no GPU, no LM.
Two modes:
  general = full sastra glossary (deployment-realistic: you don't know which terms appear)
  oracle  = important words drawn from the eval refs themselves (ceiling / upper bound)
Content-only micro CER vs greedy-beam (weight 0)."""
import json, glob, re, unicodedata
import numpy as np
from pyctcdecode import build_ctcdecoder
ROOT="/home/ece/BigDisk/Prathosh/ASR"; LOG=f"{ROOT}/data/eval_logits"
labels=json.load(open(f"{LOG}/labels.json")); refs_all=json.load(open(f"{LOG}/refs.json"))
files=sorted(glob.glob(f"{LOG}/[0-9]*.npy"))
DEVA=re.compile(r'[ऀ-ॿ]+')
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
def edist(a,b):
    n,m=len(a),len(b)
    if n==0:return m
    if m==0:return n
    p=list(range(m+1))
    for i in range(1,n+1):
        c=[i]+[0]*m;ai=a[i-1]
        for j in range(1,m+1):c[j]=min(p[j]+1,c[j-1]+1,p[j-1]+(ai!=b[j-1]))
        p=c
    return p[m]
# subset for speed (beam+hotwords is CPU-heavy)
idx=list(range(0,len(files),2)); LP=[np.load(files[i]) for i in idx]; refs=[norm(refs_all[i]) for i in idx]
print(f"subset {len(LP)} utts")
# general glossary: clean to devanagari-only tokens, dedupe, len>=3
gloss=set()
for line in open(f"{ROOT}/corpus/glossary/all_hotwords.txt"):
    for w in DEVA.findall(unicodedata.normalize('NFC',line)):
        if len(w)>=3: gloss.add(w)
GEN=sorted(gloss)
# oracle: long-ish words from the eval refs (unique)
orc=set()
for r in refs:
    for w in r.split():
        if len(w)>=5: orc.add(w)
ORACLE=sorted(orc)
print(f"general hotwords: {len(GEN)} | oracle hotwords: {len(ORACLE)}")
dec=build_ctcdecoder(labels)   # no LM
def run(hw, w):
    ce=cn=0
    for lp,r in zip(LP,refs):
        h=norm(dec.decode(lp, beam_width=100, hotwords=hw, hotword_weight=w))
        ce+=edist(h.replace(" ",""), r.replace(" ","")); cn+=len(r.replace(" ",""))
    return 100*ce/max(1,cn)
print(f"\n  baseline (no hotwords)        CER {run([],0):.2f}%", flush=True)
for w in (5,10,20):
    print(f"  GENERAL glossary  w={w:<3}       CER {run(GEN,w):.2f}%", flush=True)
for w in (10,20):
    print(f"  ORACLE (ref-derived) w={w:<3}    CER {run(ORACLE,w):.2f}%  [ceiling]", flush=True)
