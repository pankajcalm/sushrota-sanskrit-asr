#!/usr/bin/env python3
"""Run a HF Whisper model on OUR eval manifests, scored with the identical
content-only normalization used for our AM baseline (strip dandas/digits/avagraha/
om/Vedic svaras; keep Devanagari only). Micro CER/WER + per-speaker."""
import json, re, unicodedata, argparse, sys
import numpy as np, soundfile as sf, torch
_ap=argparse.ArgumentParser()
_ap.add_argument("--model", default="openai/whisper-large-v3")
_ap.add_argument("--manifest", required=True)
_ap.add_argument("--tag", default="")
_ap.add_argument("--bs", type=int, default=16)
_ap.add_argument("--limit", type=int, default=0)
A=_ap.parse_args()
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
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
rows=[json.loads(l) for l in open(A.manifest)]
if A.limit: rows=rows[:A.limit]
from transformers import pipeline
dev=0 if torch.cuda.is_available() else -1
print(f"loading {A.model} on cuda:{dev} ...", file=sys.stderr)
asr=pipeline("automatic-speech-recognition", model=A.model, device=dev,
             torch_dtype=torch.float16, chunk_length_s=30)
gkw=dict(language="sanskrit", task="transcribe")
def load(fp):
    x,sr=sf.read(fp, dtype='float32')
    if x.ndim>1: x=x.mean(1)
    if sr!=16000:
        import librosa; x=librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x
hyps=[]
B=A.bs
for i in range(0,len(rows),B):
    batch=[load(r["audio_filepath"]) for r in rows[i:i+B]]
    out=asr(batch, batch_size=B, generate_kwargs=gkw)
    hyps+=[o["text"] for o in out]
    print(f"  {min(i+B,len(rows))}/{len(rows)}", file=sys.stderr)
tot_ce=tot_cn=tot_we=tot_wn=0
spk_ce={}; spk_cn={}
lat=0
for r,h in zip(rows,hyps):
    if re.search(r'[a-zA-Z]', h): lat+=1
    hn=norm(h); rn=norm(r["text"])
    hc,rc=hn.replace(' ',''), rn.replace(' ','')
    tot_ce+=ed(hc,rc); tot_cn+=len(rc)
    hw,rw=hn.split(), rn.split()
    tot_we+=ed(hw,rw); tot_wn+=len(rw)
    sp=r.get("speaker","all"); spk_ce[sp]=spk_ce.get(sp,0)+ed(hc,rc); spk_cn[sp]=spk_cn.get(sp,0)+len(rc)
print(f"\n== {A.tag or A.model} on {A.manifest.split('/')[-1]} ({len(rows)} utts) ==")
print(f"== OVERALL (micro) ==  CER {100*tot_ce/tot_cn:.2f}%  WER {100*tot_we/tot_wn:.2f}%")
print(f"   (utts with latin chars in raw hyp: {lat}/{len(rows)})")
if len(spk_ce)>1:
    for sp in sorted(spk_ce): print(f"   {sp:16} CER {100*spk_ce[sp]/max(1,spk_cn[sp]):.2f}%")
