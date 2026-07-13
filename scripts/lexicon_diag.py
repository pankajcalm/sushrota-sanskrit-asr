#!/usr/bin/env python3
"""Diagnose the 19%: decompose wrong-word recall by edit-distance(raw,correct).
Tells us if the bottleneck is candidate GENERATION (edit-dist too narrow) or
RANKING (frequency pulls to common words). CPU."""
import json, glob, re, unicodedata, difflib
from collections import Counter
import numpy as np
ROOT="/home/ece/BigDisk/Prathosh/ASR"; LOG=f"{ROOT}/data/eval_logits"
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
lex=Counter()
for fn in glob.glob(f"{ROOT}/corpus/norm/*.txt"):
    if fn.endswith(('.prose.txt','.verse.txt')): continue
    for line in open(fn, encoding='utf-8', errors='ignore'):
        for w in norm(line).split():
            if w: lex[w]+=1
for gf in glob.glob(f"{ROOT}/corpus/glossary/rerank/*.hotwords.txt"):
    for line in open(gf, encoding='utf-8', errors='ignore'):
        if line.startswith('#') or not line.strip(): continue
        w=norm(line.split('\t')[0])
        if w: lex[w]+=50
LEX={w:c for w,c in lex.items() if c>=2}
ALPH=sorted(set(''.join(LEX.keys())))
def edits1(w):
    sp=[(w[:i],w[i:]) for i in range(len(w)+1)]; out=set()
    for a,b in sp:
        if b: out.add(a+b[1:])
        if b:
            for c in ALPH: out.add(a+c+b[1:])
        for c in ALPH: out.add(a+c+b)
        if len(b)>1: out.add(a+b[1]+b[0]+b[2:])
    return out
def cand1(w): return [x for x in edits1(w) if x in LEX]
labels=json.load(open(f"{LOG}/labels.json")); refs=json.load(open(f"{LOG}/refs.json"))
files=sorted(glob.glob(f"{LOG}/[0-9]*.npy")); BLANK=len(labels)-1
def greedy(lp):
    ids=lp.argmax(-1); out=[]; prev=-1
    for i in ids:
        if i!=prev and i!=BLANK: out.append(labels[i])
        prev=i
    return ''.join(out).replace('▁',' ').strip()
# bucket wrong pairs by ed(h,r)
from collections import defaultdict
B=defaultdict(lambda:{"n":0,"inlex":0,"gen":0,"top5_freq":0,"top5_dist":0,"top1_freq":0})
for f,ref in zip(files,refs):
    rw=norm(ref).split(); hw=norm(greedy(np.load(f))).split()
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(a=rw,b=hw,autojunk=False).get_opcodes():
        if tag!='replace': continue
        for k in range(min(i2-i1,j2-j1)):
            r=rw[i1+k]; h=hw[j1+k]; d=ed(h,r); key=d if d<=2 else 3
            b=B[key]; b["n"]+=1
            if r in LEX: b["inlex"]+=1
            cs=cand1(h)                       # edit-distance-1 candidates in lexicon
            if r in cs: b["gen"]+=1           # is correct word even generated?
            byf=sorted(set(cs), key=lambda x:-LEX[x])[:5]
            if byf and byf[0]==r: b["top1_freq"]+=1
            if r in byf: b["top5_freq"]+=1
lab={1:"exactly 1 edit off",2:"exactly 2 edits off",3:"3+ edits (misaligned/far)"}
print("== recall decomposed by edit-distance(raw, correct) ==")
print("  bucket                     n     in-lex  gen@d1  top5(freq)  top1(freq)")
for k in sorted(B):
    b=B[k]; n=b["n"]
    print(f"  {lab[k]:26} {n:5}  {100*b['inlex']/n:5.0f}%  {100*b['gen']/n:5.0f}%   {100*b['top5_freq']/n:6.0f}%     {100*b['top1_freq']/n:5.0f}%")
