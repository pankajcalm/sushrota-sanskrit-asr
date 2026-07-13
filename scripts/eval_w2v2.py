#!/usr/bin/env python3
"""Eval a finetuned wav2vec2-CTC model on our eval manifests, content-only norm
(identical to AM baseline). Greedy CTC decode. Micro CER/WER + per-speaker."""
import json, re, unicodedata, argparse, sys
import torch, soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
ap=argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--bs", type=int, default=16)
A=ap.parse_args()
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
proc=Wav2Vec2Processor.from_pretrained(A.model)
model=Wav2Vec2ForCTC.from_pretrained(A.model).to("cuda").half().eval()
rows=[json.loads(l) for l in open(A.manifest)]
def load(fp):
    x,sr=sf.read(fp, dtype='float32')
    if x.ndim>1: x=x.mean(1)
    if sr!=16000:
        import librosa; x=librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x
hyps=[]
for i in range(0,len(rows),A.bs):
    wavs=[load(r["audio_filepath"]) for r in rows[i:i+A.bs]]
    inp=proc(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits=model(inp.input_values.to("cuda").half(),
                     attention_mask=inp.attention_mask.to("cuda")).logits
    ids=torch.argmax(logits,dim=-1)
    hyps+=proc.batch_decode(ids)
    print(f"  {min(i+A.bs,len(rows))}/{len(rows)}", file=sys.stderr)
tce=tcn=twe=twn=0; sce={}; scn={}
for r,h in zip(rows,hyps):
    hn=norm(h); rn=norm(r["text"])
    hc,rc=hn.replace(' ',''), rn.replace(' ','')
    tce+=ed(hc,rc); tcn+=len(rc)
    twe+=ed(hn.split(),rn.split()); twn+=len(rn.split())
    sp=r.get("speaker","all"); sce[sp]=sce.get(sp,0)+ed(hc,rc); scn[sp]=scn.get(sp,0)+len(rc)
print(f"\n== {A.tag or A.model} on {A.manifest.split('/')[-1]} ({len(rows)} utts) ==")
print(f"== OVERALL (micro) ==  CER {100*tce/tcn:.2f}%  WER {100*twe/twn:.2f}%")
if len(sce)>1:
    for sp in sorted(sce): print(f"   {sp:16} CER {100*sce[sp]/max(1,scn[sp]):.2f}%")
