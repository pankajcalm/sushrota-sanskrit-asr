#!/usr/bin/env python3
"""Error analysis on sk10 from exported ft_ep20 logits (CPU, no GPU).
Reports: standard content-only CER, PHONETIC-normalized CER (collapse anusvara/nasal
orthographic variants), and a breakdown of every edit op into linguistic categories."""
import json, glob, re, unicodedata, difflib
import numpy as np
ROOT="/home/ece/BigDisk/Prathosh/ASR"; LOG=f"{ROOT}/data/eval_logits"
labels=json.load(open(f"{LOG}/labels.json")); refs=json.load(open(f"{LOG}/refs.json"))
files=sorted(glob.glob(f"{LOG}/[0-9]*.npy")); BLANK=len(labels)-1
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
def phon(s):  # collapse truly-equivalent nasal orthography -> anusvara
    s=s.replace('ँ','ं')
    s=re.sub(r'[ङञणनम]्(?=[क-ह])','ं',s)   # nasal+virama before consonant -> ं
    s=re.sub(r'म्(?=\s|$)','ं',s)            # word-final म् -> ं
    return re.sub(r'\s+',' ',s).strip()
def greedy(lp):
    ids=lp.argmax(-1); out=[]; prev=-1
    for i in ids:
        if i!=prev and i!=BLANK: out.append(labels[i])
        prev=i
    return ''.join(out).replace('▁',' ').strip()
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
NASAL=set('ंँङञणनम'); SIB=set('शषस')
RETRO_DENTAL=[set('नण'),set('तट'),set('थठ'),set('दड'),set('धढ'),set('सष')]
STOP=[set('कखगघ'),set('चछजझ'),set('टठडढ'),set('तथदध'),set('पफबभ')]
VOWEL=[set('अआ'),set('इई'),set('उऊ'),set('ऋॠ'),set('िी'),set('ुू'),set('ेै'),set('ोौ')]
def cat(rc,hc):
    a,b=set([rc]),set([hc])
    if rc in NASAL and hc in NASAL: return 'nasal(orthographic)'
    if rc in SIB and hc in SIB: return 'sibilant ś/ṣ/s'
    if any(rc in g and hc in g for g in RETRO_DENTAL): return 'retroflex↔dental'
    if any(rc in g and hc in g for g in STOP): return 'voicing/aspiration'
    if any(rc in g and hc in g for g in VOWEL): return 'vowel-length'
    return 'other-substitution'
from collections import defaultdict, Counter
catc=Counter(); tot_ce=tot_cn=0; std_ce=0; phon_ce=0; phon_cn=0
for f,ref in zip(files,refs):
    h=norm(greedy(np.load(f))); r=norm(ref)
    hc,rc=h.replace(' ',''), r.replace(' ','')
    std_ce+=ed(hc,rc); tot_cn+=len(rc)
    phon_ce+=ed(phon(h).replace(' ',''), phon(r).replace(' ','')); phon_cn+=len(phon(r).replace(' ',''))
    # categorize edit ops on the un-phon-normalized chars
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(a=hc,b=rc,autojunk=False).get_opcodes():
        if tag=='equal': continue
        if tag=='replace':
            L=min(i2-i1,j2-j1)
            for k in range(L): catc[cat(rc[j1+k],hc[i1+k])]+=1
            catc['insertion']+= max(0,(i2-i1)-L); catc['deletion']+= max(0,(j2-j1)-L)
        elif tag=='insert': catc['insertion']+= i2-i1
        elif tag=='delete': catc['deletion']+= j2-j1
print(f"== error analysis (ft_ep20 greedy on sk10, {len(files)} utts) ==")
print(f"  standard content-only CER : {100*std_ce/tot_cn:.2f}%")
print(f"  PHONETIC-normalized CER   : {100*phon_ce/phon_cn:.2f}%   (nasal orthography collapsed)")
print(f"\n== edit-op breakdown (share of all {sum(catc.values())} ops) ==")
for k,v in catc.most_common():
    print(f"  {k:22} {v:5d}  {100*v/sum(catc.values()):5.1f}%")
