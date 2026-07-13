#!/usr/bin/env python3
"""3-way complementarity + REAL fusion on chant: v5 (IndicConformer-CTC),
XLSR (wav2vec2-CTC), Whisper-medium-ft. Reports individual CER, pairwise + 3-way
ORACLE (ceiling), and anchored-ROVER majority vote (reference-free = shippable number)."""
import json, glob, re, unicodedata, difflib
from collections import Counter
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
def load(fp):
    x,sr=sf.read(fp, dtype='float32')
    if x.ndim>1: x=x.mean(1)
    return x
# XLSR decode
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
xp=Wav2Vec2Processor.from_pretrained(f"{ROOT}/exp/w2v2_xlsr_v5")
xm=Wav2Vec2ForCTC.from_pretrained(f"{ROOT}/exp/w2v2_xlsr_v5").to("cuda").half().eval()
xl=[]
for i in range(0,len(man),16):
    inp=xp([load(r["audio_filepath"]) for r in man[i:i+16]], sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        lg=xm(inp.input_values.to("cuda").half(), attention_mask=inp.attention_mask.to("cuda")).logits
    xl+=xp.batch_decode(torch.argmax(lg,-1))
del xm; torch.cuda.empty_cache()
# Whisper decode
asr=pipeline("automatic-speech-recognition", model=f"{ROOT}/exp/whisper_med_v5", device=0,
             torch_dtype=torch.float16, chunk_length_s=30)
wh=[]
for i in range(0,len(man),16):
    out=asr([load(r["audio_filepath"]) for r in man[i:i+16]], batch_size=16,
            generate_kwargs=dict(language="sanskrit", task="transcribe"))
    wh+=[o["text"] for o in out]
# anchored ROVER: v5 backbone; override a char only if the other two AGREE on a different char
def align_to(ref, hyp):
    res=['']*len(ref)
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(a=ref,b=hyp,autojunk=False).get_opcodes():
        if tag=='equal':
            for k in range(i2-i1): res[i1+k]=hyp[j1+k]
        elif tag=='replace':
            for k in range(i2-i1): res[i1+k]=hyp[j1+k] if k<(j2-j1) else ''
    return res
def rover(a,b,c):
    ba=align_to(a,b); ca=align_to(a,c); out=[]
    for k in range(len(a)):
        cnt=Counter([a[k],ba[k],ca[k]]); best,bn=cnt.most_common(1)[0]
        if bn>=2 and best!='': out.append(best)
        elif bn>=2 and best=='': pass       # 2 vote delete
        else: out.append(a[k])              # tie -> backbone
    return ''.join(out)
e={"v5":0,"xlsr":0,"wh":0,"o2_vx":0,"o2_vw":0,"o3":0,"rover":0}; tn=0
resc_x=resc_w=0
for r,ref,f,h,x in zip(man,refs,files,wh,xl):
    rc=norm(ref); v=norm(v5greedy(np.load(f))); xx=norm(x); ww=norm(h)
    dv=ed(v,rc); dx=ed(xx,rc); dw=ed(ww,rc); L=len(rc); tn+=L
    e["v5"]+=dv; e["xlsr"]+=dx; e["wh"]+=dw
    e["o2_vx"]+=min(dv,dx); e["o2_vw"]+=min(dv,dw); e["o3"]+=min(dv,dx,dw)
    e["rover"]+=ed(norm(rover(v,xx,ww)),rc)
    if dx<dv: resc_x+=1
    if dw<dv: resc_w+=1
def P(k): return 100*e[k]/tn
print(f"== 3-way complementarity + fusion on chant ({len(man)} utts) ==")
print(f"  v5   (IndicConformer-CTC) CER : {P('v5'):.2f}%")
print(f"  XLSR (wav2vec2-CTC)       CER : {P('xlsr'):.2f}%")
print(f"  Whisper-medium-ft         CER : {P('wh'):.2f}%")
print(f"  --- oracle ceilings (cheat, use ref) ---")
print(f"  oracle v5+XLSR                : {P('o2_vx'):.2f}%")
print(f"  oracle v5+Whisper             : {P('o2_vw'):.2f}%")
print(f"  oracle v5+XLSR+Whisper (3way) : {P('o3'):.2f}%   <-- absolute ceiling")
print(f"  --- REAL fusion (reference-free, shippable) ---")
print(f"  anchored ROVER (v5+XLSR+Wh)   : {P('rover'):.2f}%   <-- vs v5 {P('v5'):.2f}%")
print(f"  XLSR rescues v5 on {resc_x} utts ; Whisper rescues v5 on {resc_w} utts")
