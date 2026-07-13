#!/usr/bin/env python3
"""Validate the top-5 suggestion feature: build a frequency-weighted Devanagari lexicon
from the multi-shastra corpus (+ glossary), then on sk10 measure whether the CORRECT word
is in the top-5 lexicon suggestions when the raw ASR word is wrong. Edit-distance-1
candidate generation (Norvig), ranked by corpus frequency. CPU, no GPU."""
import json, glob, re, unicodedata, difflib
from collections import Counter
import numpy as np
ROOT="/home/ece/BigDisk/Prathosh/ASR"; LOG=f"{ROOT}/data/eval_logits"
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
# ---- build lexicon from corpus ----
lex=Counter()
for fn in glob.glob(f"{ROOT}/corpus/norm/*.txt"):
    if fn.endswith(('.prose.txt','.verse.txt')): continue   # use the combined bucket only
    for line in open(fn, encoding='utf-8', errors='ignore'):
        for w in norm(line).split():
            if w: lex[w]+=1
for gf in glob.glob(f"{ROOT}/corpus/glossary/rerank/*.hotwords.txt"):
    for line in open(gf, encoding='utf-8', errors='ignore'):
        if line.startswith('#') or not line.strip(): continue
        w=norm(line.split('\t')[0])
        if w: lex[w]+=50   # glossary boost (curated technical terms)
LEX={w:c for w,c in lex.items() if c>=2}
ALPH=sorted(set(''.join(LEX.keys())))
print(f"lexicon: {len(lex)} raw forms, {len(LEX)} kept (freq>=2 ∪ glossary), alphabet {len(ALPH)} chars")
def edits1(w):
    sp=[(w[:i],w[i:]) for i in range(len(w)+1)]
    out=set()
    for a,b in sp:
        if b: out.add(a+b[1:])                      # delete
        if b:
            for c in ALPH: out.add(a+c+b[1:])       # substitute
        for c in ALPH: out.add(a+c+b)               # insert
        if len(b)>1: out.add(a+b[1]+b[0]+b[2:])     # transpose
    return out
def suggest(w,k=5):
    c=[x for x in edits1(w) if x in LEX and x!=w]
    return sorted(set(c), key=lambda x:-LEX[x])[:k]
# ---- decode v5 greedy from logits ----
labels=json.load(open(f"{LOG}/labels.json")); refs=json.load(open(f"{LOG}/refs.json"))
files=sorted(glob.glob(f"{LOG}/[0-9]*.npy")); BLANK=len(labels)-1
def greedy(lp):
    ids=lp.argmax(-1); out=[]; prev=-1
    for i in ids:
        if i!=prev and i!=BLANK: out.append(labels[i])
        prev=i
    return ''.join(out).replace('▁',' ').strip()
# ---- measure ----
n_ref=n_correct=n_wrong=0
cov=0; top5=0; top1=0; wrong_in_lex=0
for f,ref in zip(files,refs):
    rw=norm(ref).split(); hw=norm(greedy(np.load(f))).split()
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(a=rw,b=hw,autojunk=False).get_opcodes():
        if tag=='equal':
            n_ref+=i2-i1; n_correct+=i2-i1
        elif tag=='replace':
            for k in range(min(i2-i1,j2-j1)):
                r=rw[i1+k]; h=hw[j1+k]; n_ref+=1; n_wrong+=1
                if r in LEX: cov+=1
                sg=suggest(h)
                if r in LEX: wrong_in_lex+=1
                if sg and sg[0]==r: top1+=1
                if r in sg: top5+=1
            n_ref+=max(0,(i2-i1)-(j2-j1))   # ref words with no aligned hyp (deletions)
        elif tag=='delete':
            n_ref+=i2-i1
print(f"\n== suggestion-feature validation on sk10 ==")
print(f"  ref words                         : {n_ref}")
print(f"  raw ASR already correct           : {n_correct} ({100*n_correct/n_ref:.1f}%)  <- no suggestion needed")
print(f"  raw ASR wrong (substitution)      : {n_wrong}")
print(f"  --- of the wrong words ---")
print(f"  correct word IS in lexicon (ceil) : {100*cov/n_wrong:.1f}%")
print(f"  correct word in TOP-5 suggestions : {100*top5/n_wrong:.1f}%")
print(f"  correct word is TOP-1 suggestion  : {100*top1/n_wrong:.1f}%")
print(f"  => with raw + top-5 shown, user can reach the right word on")
print(f"     {100*(n_correct+top5)/n_ref:.1f}% of all words (raw-correct + top5-recoverable)")
