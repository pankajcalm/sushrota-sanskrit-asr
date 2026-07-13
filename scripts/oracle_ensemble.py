#!/usr/bin/env python3
"""Complementarity / oracle analysis for v5 (IndicConformer-CTC) vs XLSR (wav2vec2-CTC)
on the chant eval. Decodes v5 from exported logits, XLSR from audio; computes per-utt
CER for each, the per-utterance ORACLE (min) CER, and agreement stats. Decides whether
an ensemble has headroom BEFORE we build one. CPU-friendly except XLSR forward (GPU)."""
import json, glob, re, unicodedata, argparse
import numpy as np, torch, soundfile as sf
ROOT="/home/ece/BigDisk/Prathosh/ASR"; LOG=f"{ROOT}/data/eval_logits"
ap=argparse.ArgumentParser()
ap.add_argument("--xlsr", default=f"{ROOT}/exp/w2v2_xlsr_v5")
ap.add_argument("--manifest", default=f"{ROOT}/data/utts_fa/manifest_fa_eval_sk10.jsonl")
A=ap.parse_args()
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
# --- v5 greedy from exported logits (aligned to refs.json / manifest order) ---
labels=json.load(open(f"{LOG}/labels.json")); refs=json.load(open(f"{LOG}/refs.json"))
files=sorted(glob.glob(f"{LOG}/[0-9]*.npy")); BLANK=len(labels)-1
def v5greedy(lp):
    ids=lp.argmax(-1); out=[]; prev=-1
    for i in ids:
        if i!=prev and i!=BLANK: out.append(labels[i])
        prev=i
    return ''.join(out).replace('▁',' ').strip()
man=[json.loads(l) for l in open(A.manifest)]
assert len(man)==len(files)==len(refs), f"len mismatch {len(man)} {len(files)} {len(refs)}"
# --- XLSR decode from audio ---
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
proc=Wav2Vec2Processor.from_pretrained(A.xlsr)
model=Wav2Vec2ForCTC.from_pretrained(A.xlsr).to("cuda").half().eval()
def load(fp):
    x,sr=sf.read(fp, dtype='float32')
    if x.ndim>1: x=x.mean(1)
    return x
xh=[]
for i in range(0,len(man),16):
    wavs=[load(r["audio_filepath"]) for r in man[i:i+16]]
    inp=proc(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        lg=model(inp.input_values.to("cuda").half(), attention_mask=inp.attention_mask.to("cuda")).logits
    xh+=proc.batch_decode(torch.argmax(lg,-1))
# --- per-utt CER + oracle ---
e5=en5=ex=enx=eo=0; comp_win=0; xlsr_win=0; tie=0; both0=0
for r,ref,f,h in zip(man,refs,files,xh):
    rc=norm(ref); v=norm(v5greedy(np.load(f))); x=norm(h)
    d5=ed(v,rc); dx=ed(x,rc); L=len(rc)
    e5+=d5; en5+=L; ex+=dx; enx+=L; eo+=min(d5,dx)
    if d5==0 and dx==0: both0+=1
    elif d5<dx: xlsr_win+=0; comp_win+=1  # v5 better
    elif dx<d5: xlsr_win+=1
    else: tie+=1
N=len(man)
print(f"== oracle / complementarity on chant ({N} utts) ==")
print(f"  v5 (IndicConformer-CTC)  CER : {100*e5/en5:.2f}%")
print(f"  XLSR (wav2vec2-CTC)      CER : {100*ex/enx:.2f}%")
print(f"  ORACLE (per-utt min)     CER : {100*eo/en5:.2f}%   <-- fusion ceiling")
print(f"  headroom vs v5               : {100*(e5-eo)/en5:.2f} CER points")
print(f"  utts both perfect            : {both0}")
print(f"  utts v5 strictly better      : {comp_win}")
print(f"  utts XLSR strictly better    : {xlsr_win}")
print(f"  utts tie (incl both wrong eq): {tie}")
print(f"  => XLSR rescues {xlsr_win}/{N} utts v5 gets worse; complementarity {'LIKELY' if 100*(e5-eo)/en5>0.4 else 'WEAK'}")
