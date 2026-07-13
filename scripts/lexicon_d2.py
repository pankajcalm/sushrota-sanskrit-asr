#!/usr/bin/env python3
"""Measure top-5 recall with edit-distance-2 candidate generation (rapidfuzz),
ranked by (distance asc, corpus-frequency desc). Decomposed by ed(raw,correct).
Confirms the d=2 lift. CPU."""
import json, glob, re, unicodedata, difflib
from collections import Counter, defaultdict
import numpy as np
from rapidfuzz import process, distance as rfdist
ROOT="/home/ece/BigDisk/Prathosh/ASR"; LOG=f"{ROOT}/data/eval_logits"
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
LEVd=rfdist.Levenshtein.distance
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
# bucket words by length for fast length-filtered fuzzy search
bylen=defaultdict(list)
for w in LEX: bylen[len(w)].append(w)
print(f"lexicon {len(LEX)} forms; measuring...", flush=True)
def suggest(h,k=5):
    pool=[]
    for L in range(len(h)-2, len(h)+3):
        pool.extend(bylen.get(L,()))
    if not pool: return []
    hits=process.extract(h, pool, scorer=LEVd, score_cutoff=2, limit=200)
    # hits: (word, dist, idx); rank by (dist asc, freq desc)
    hits=sorted(hits, key=lambda t:(t[1], -LEX[t[0]]))
    seen=[]; 
    for w,d,_ in hits:
        if w!=h and w not in seen: seen.append(w)
        if len(seen)>=k: break
    return seen
labels=json.load(open(f"{LOG}/labels.json")); refs=json.load(open(f"{LOG}/refs.json"))
files=sorted(glob.glob(f"{LOG}/[0-9]*.npy")); BLANK=len(labels)-1
def greedy(lp):
    ids=lp.argmax(-1); out=[]; prev=-1
    for i in ids:
        if i!=prev and i!=BLANK: out.append(labels[i])
        prev=i
    return ''.join(out).replace('▁',' ').strip()
B=defaultdict(lambda:{"n":0,"top5":0,"top1":0})
alln=alltop5=0
for f,ref in zip(files,refs):
    rw=norm(ref).split(); hw=norm(greedy(np.load(f))).split()
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(a=rw,b=hw,autojunk=False).get_opcodes():
        if tag!='replace': continue
        for k in range(min(i2-i1,j2-j1)):
            r=rw[i1+k]; h=hw[j1+k]; d=LEVd(h,r); key=d if d<=2 else 3
            sg=suggest(h); b=B[key]; b["n"]+=1; alln+=1
            if sg and sg[0]==r: b["top1"]+=1
            if r in sg: b["top5"]+=1; alltop5+=1
lab={1:"1 edit off",2:"2 edits off",3:"3+ edits (structural)"}
print("== top-5 recall with EDIT-DISTANCE-2 generation, dist-then-freq rank ==")
print("  bucket                    n     top5    top1")
for k in sorted(B):
    b=B[k]; n=b["n"]
    print(f"  {lab[k]:22} {n:5}   {100*b['top5']/n:5.0f}%  {100*b['top1']/n:5.0f}%")
near=B[1]["n"]+B[2]["n"]; nt5=B[1]["top5"]+B[2]["top5"]
print(f"  --- addressable near-misses (d<=2): {near} words, top-5 recall {100*nt5/near:.0f}% ---")
print(f"  overall top-5 over ALL wrong words: {100*alltop5/alln:.0f}%")
