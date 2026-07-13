import json, glob, re, unicodedata, difflib
import numpy as np
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
# whole-word: exact match, and "close" (word CER<=1 char) = readable-correct
wtot=wcorr=wclose=0
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
for f,ref in zip(files,refs):
    rw=norm(ref).split(); hw=greedy(np.load(f)).split()
    sm=difflib.SequenceMatcher(a=rw,b=hw,autojunk=False)
    wtot+=len(rw)
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=='equal': wcorr+=i2-i1; wclose+=i2-i1
        elif tag=='replace':
            for k in range(min(i2-i1,j2-j1)):
                if ed(rw[i1+k],hw[j1+k])<=1: wclose+=1
print(f'== whole-word accuracy on sk10 ({len(files)} utts, {wtot} ref words) ==')
print(f'  exact-word accuracy      : {100*wcorr/wtot:.1f}%   (word error rate {100*(1-wcorr/wtot):.1f}%)')
print(f'  within-1-char accuracy   : {100*wclose/wtot:.1f}%   (readable-correct)')
