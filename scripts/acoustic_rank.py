#!/usr/bin/env python3
"""Acoustic (CTC-posterior) ranking of lexicon suggestions vs frequency ranking.
For each near-miss word, CTC-forward-score every edit-distance-<=2 lexicon candidate
against that word's frame span, rank by acoustic score. Measures top-5/top-1. CPU."""
import os; os.environ["CUDA_VISIBLE_DEVICES"]=""
import json, glob, re, unicodedata, difflib
from collections import Counter, defaultdict
import numpy as np
from rapidfuzz import process, distance as rfdist
import nemo.collections.asr as nemo_asr
ROOT="/home/ece/BigDisk/Prathosh/ASR"; LOG=f"{ROOT}/data/eval_logits"
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
LEVd=rfdist.Levenshtein.distance
m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location="cpu")
_tokcache={}
def toks(w):
    if w not in _tokcache: _tokcache[w]=m.tokenizer.text_to_ids(w,"sa")
    return _tokcache[w]
labels=json.load(open(f"{LOG}/labels.json")); refs=json.load(open(f"{LOG}/refs.json"))
files=sorted(glob.glob(f"{LOG}/[0-9]*.npy")); BLANK=len(labels)-1
# lexicon
lex=Counter()
for fn in glob.glob(f"{ROOT}/corpus/norm/*.txt"):
    if fn.endswith(('.prose.txt','.verse.txt')): continue
    for line in open(fn,encoding='utf-8',errors='ignore'):
        for w in norm(line).split():
            if w: lex[w]+=1
for gf in glob.glob(f"{ROOT}/corpus/glossary/rerank/*.hotwords.txt"):
    for line in open(gf,encoding='utf-8',errors='ignore'):
        if line.startswith('#') or not line.strip(): continue
        w=norm(line.split('\t')[0])
        if w: lex[w]+=50
LEX={w:c for w,c in lex.items() if c>=2}
bylen=defaultdict(list)
for w in LEX: bylen[len(w)].append(w)
print(f"lexicon {len(LEX)} forms; model+tokenizer loaded; measuring...", flush=True)
def cands(h):
    pool=[]
    for L in range(len(h)-2,len(h)+3): pool.extend(bylen.get(L,()))
    hits=process.extract(h,pool,scorer=LEVd,score_cutoff=2,limit=200) if pool else []
    return [(w,d) for w,d,_ in hits if w!=h]
def word_spans(lp):
    ids=lp.argmax(-1); emitted=[]; prev=-1
    for t,i in enumerate(ids):
        if i!=prev and i!=BLANK: emitted.append((t,int(i)))
        prev=int(i)
    W=[]; cur=[]
    for (t,i) in emitted:
        if labels[i].startswith('▁') and cur:
            s=''.join(labels[x] for _,x in cur).replace('▁',' ').strip()
            W.append((s,cur[0][0],t)); cur=[]
        cur.append((t,i))
    if cur:
        s=''.join(labels[x] for _,x in cur).replace('▁',' ').strip()
        W.append((s,cur[0][0],len(ids)))
    return W
def ctc_score(seq, P):
    T=P.shape[0]; L=len(seq)
    if L==0 or T==0: return -1e30
    ext=[BLANK]
    for s in seq: ext+=[s,BLANK]
    S=len(ext); NEG=-1e30
    a=np.full(S,NEG); a[0]=P[0,ext[0]]
    if S>1: a[1]=P[0,ext[1]]
    for t in range(1,T):
        na=np.full(S,NEG)
        for s in range(S):
            v=a[s]
            if s>0: v=np.logaddexp(v,a[s-1])
            if s>1 and ext[s]!=BLANK and ext[s]!=ext[s-2]: v=np.logaddexp(v,a[s-2])
            na[s]=v+P[t,ext[s]]
        a=na
    return float(np.logaddexp(a[S-1],a[S-2])) if S>1 else float(a[S-1])
Bf=defaultdict(lambda:{"n":0,"t5":0,"t1":0}); Ba=defaultdict(lambda:{"n":0,"t5":0,"t1":0})
for f,ref in zip(files,refs):
    P=np.load(f); W=word_spans(P); hw=[w[0] for w in W]; rw=norm(ref).split()
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(a=rw,b=hw,autojunk=False).get_opcodes():
        if tag!='replace': continue
        for k in range(min(i2-i1,j2-j1)):
            r=rw[i1+k]; h=hw[j1+k]; s,e=W[j1+k][1],W[j1+k][2]; d=LEVd(h,r)
            if d>2: 
                key=3
                Bf[key]["n"]+=1; Ba[key]["n"]+=1; continue
            key=d; cs=cands(h)
            if not cs: 
                Bf[key]["n"]+=1; Ba[key]["n"]+=1; continue
            # freq ranking
            byf=sorted(cs,key=lambda t:(t[1],-LEX[t[0]]))[:5]; fw=[w for w,_ in byf]
            Bf[key]["n"]+=1
            if fw and fw[0]==r: Bf[key]["t1"]+=1
            if r in fw: Bf[key]["t5"]+=1
            # acoustic ranking
            span=P[s:e] if e>s else P[s:s+1]
            sc=[(w, ctc_score(toks(w),span)) for w,_ in cs]
            bya=[w for w,_ in sorted(sc,key=lambda t:-t[1])[:5]]
            Ba[key]["n"]+=1
            if bya and bya[0]==r: Ba[key]["t1"]+=1
            if r in bya: Ba[key]["t5"]+=1
def show(B,name):
    lab={1:"1 edit",2:"2 edits",3:"3+ (structural)"}
    print(f"\n== {name} ==")
    print("  bucket             n     top5   top1")
    for k in sorted(B):
        b=B[k]; n=max(1,b["n"])
        print(f"  {lab[k]:16} {b['n']:5}  {100*b['t5']/n:5.0f}% {100*b['t1']/n:5.0f}%")
    near=B[1]["n"]+B[2]["n"]; t5=B[1]["t5"]+B[2]["t5"]; t1=B[1]["t1"]+B[2]["t1"]
    print(f"  addressable(d<=2): {near}  top5 {100*t5/max(1,near):.0f}%  top1 {100*t1/max(1,near):.0f}%")
show(Bf,"FREQUENCY ranking (baseline)")
show(Ba,"ACOUSTIC (CTC-posterior) ranking")
