import json, glob, re, unicodedata, difflib
import numpy as np
from collections import Counter
ROOT='/home/ece/BigDisk/Prathosh/ASR'; LOG=f'{ROOT}/data/eval_logits'
labels=json.load(open(f'{LOG}/labels.json')); refs=json.load(open(f'{LOG}/refs.json'))
files=sorted(glob.glob(f'{LOG}/[0-9]*.npy')); BLANK=len(labels)-1
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
def words(s): return norm(s).split()
close=[]; chg=Counter()
for f,ref in zip(files,refs):
    rw=words(ref); hw=words(greedy(np.load(f)))
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(a=rw,b=hw,autojunk=False).get_opcodes():
        if tag!='replace': continue
        for k in range(min(i2-i1,j2-j1)):
            rwd,hwd=rw[i1+k],hw[j1+k]
            if len(rwd)!=len(hwd): continue
            diffs=[(m,rwd[m],hwd[m]) for m in range(len(rwd)) if rwd[m]!=hwd[m]]
            if len(diffs)==1 and not isphon(diffs[0][1],diffs[0][2]):
                m,rc,hc=diffs[0]; close.append((rwd,hwd,rc,hc)); chg[f'{rc} {hc}']+=1
print(f'== {len(close)} pure single-char other-substitutions (same-length words) ==')
print('-- 40 examples (ref -> hyp) [rc->hc] --')
for rwd,hwd,rc,hc in close[:40]:
    print(f'  {rwd:18} -> {hwd:18}  [{rc}->{hc}]')
print('-- most common char confusions (rc -> hc) --')
for k,v in chg.most_common(25): print(f'  {k}   {v}')
