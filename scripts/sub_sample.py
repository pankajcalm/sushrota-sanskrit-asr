#!/usr/bin/env python3
"""Sample 'other-substitution' error contexts from exported ft_ep20 logits (CPU).
For each substitution op not in a phonetic category, print the ref-word vs hyp-word
so we can judge: rare terms (biasing helps) vs generic mishearing (floor)."""
import json, glob, re, unicodedata, difflib
import numpy as np
from collections import Counter
ROOT="/home/ece/BigDisk/Prathosh/ASR"; LOG=f"{ROOT}/data/eval_logits"
labels=json.load(open(f"{LOG}/labels.json")); refs=json.load(open(f"{LOG}/refs.json"))
files=sorted(glob.glob(f"{LOG}/[0-9]*.npy")); BLANK=len(labels)-1
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
def greedy(lp):
    ids=lp.argmax(-1); out=[]; prev=-1
    for i in ids:
        if i!=prev and i!=BLANK: out.append(labels[i])
        prev=i
    return ''.join(out).replace('▁',' ').strip()
NASAL=set('ंँङञणनम'); SIB=set('शषस')
RETRO_DENTAL=[set('नण'),set('तट'),set('थठ'),set('दड'),set('धढ'),set('सष')]
STOP=[set('कखगघ'),set('चछजझ'),set('टठडढ'),set('तथदध'),set('पफबभ')]
VOWEL=[set('अआ'),set('इई'),set('उऊ'),set('ऋॠ'),set('िी'),set('ुू'),set('ेै'),set('ोौ')]
def isphon(rc,hc):
    if rc in NASAL and hc in NASAL: return 1
    if rc in SIB and hc in SIB: return 1
    if any(rc in g and hc in g for g in RETRO_DENTAL): return 1
    if any(rc in g and hc in g for g in STOP): return 1
    if any(rc in g and hc in g for g in VOWEL): return 1
    return 0
# word-level: align ref/hyp word lists; for words that differ, count edit ops and
# whether the differing chars are 'other-substitution'. Print worst offenders.
def words(s): return norm(s).split()
examples=[]  # (ndiff_other, refword, hypword)
for f,ref in zip(files,refs):
    rw=words(ref); hw=words(greedy(np.load(f)))
    sm=difflib.SequenceMatcher(a=rw,b=hw,autojunk=False)
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=='replace':
            # pair up words positionally within the block
            for k in range(min(i2-i1,j2-j1)):
                rwd,hwd=rw[i1+k],hw[j1+k]
                # count 'other' substitution chars between the two words
                other=0
                for t,a1,a2,b1,b2 in difflib.SequenceMatcher(a=rwd,b=hwd,autojunk=False).get_opcodes():
                    if t=='replace':
                        for m in range(min(a2-a1,b2-b1)):
                            if not isphon(rwd[a1+m],hwd[b1+m]): other+=1
                if other>0 and rwd!=hwd:
                    examples.append((other,rwd,hwd))
examples.sort(reverse=True)
print(f"== {len(examples)} word-pairs with >=1 'other-substitution' char ==\n")
print("-- top 40 by #other-sub chars (ref -> hyp) --")
for o,r,h in examples[:40]:
    print(f"  {o}  {r:20} -> {h}")
# how many are 'close' (1 char off, likely near-miss) vs 'far' (big rewrite)?
close=sum(1 for o,r,h in examples if o<=1)
print(f"\n1-char-off pairs: {close}/{len(examples)} ({100*close/max(1,len(examples)):.0f}%)")
