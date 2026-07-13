import json,glob,re,unicodedata,math,numpy as np
from pyctcdecode import build_ctcdecoder
import kenlm
R='/home/ece/BigDisk/Prathosh/ASR'
sv=json.load(open('/tmp/sa_vocab.json')); off=sv['offset']; V=sv['V']; sa_tokens=sv['tokens']; BLANK=sv['total']
cols=list(range(off,off+V))+[BLANK]; labels=sa_tokens+['']
def logsoftmax(x):
    x=x-x.max(-1,keepdims=True); e=np.exp(x); return np.log(e/e.sum(-1,keepdims=True)+1e-12)
LOG=[logsoftmax(np.load(f).astype(np.float32)[:,cols]) for f in sorted(glob.glob('/tmp/ait_logits/chunk_*.npy'))]
ref=open('/tmp/ref.txt').read()
clm=kenlm.Model(R+'/lm/sa_char6.bin')
def co(s):
    s=unicodedata.normalize('NFC',s).replace('ॐ',' '); s=re.sub(r'[०-९।॥ऽ]',' ',s); s=re.sub(r'[^ऀ-ॿ\s]',' ',s)
    return re.sub(r'\s+',' ',s).strip()
def ed(a,b):
    n,m=len(a),len(b)
    if n==0:return m
    if m==0:return n
    p=list(range(m+1))
    for i in range(1,n+1):
        c=[i]+[0]*m; ai=a[i-1]
        for j in range(1,m+1): c[j]=min(p[j]+1,c[j-1]+1,p[j-1]+(ai!=b[j-1]))
        p=c
    return p[m]
def cer(r,h):
    r=co(r).replace(' ',''); h=co(h).replace(' ',''); return ed(r,h)/max(1,len(r))*100
def wer(r,h):
    r=co(r).split(); h=co(h).split(); return ed(r,h)/max(1,len(r))*100
def char_tok(text):
    w=co(text).split(); t=[]
    for i,x in enumerate(w):
        if i: t.append('|')
        t.extend(list(x))
    return ' '.join(t)
def clm_score(text):  # nat-log
    s=char_tok(text)
    if not s: return -1e9
    return clm.score(s, bos=True, eos=True)*math.log(10)

dec=build_ctcdecoder(labels, kenlm_model_path=None)
# top beams per chunk: (text, acoustic_logit_score)
BEAMS=[]
for l in LOG:
    bl=dec.decode_beams(l, beam_width=200)
    BEAMS.append([(b[0], b[3]) for b in bl[:150]])
# baseline: best acoustic beam
base=' '.join(bs[0][0] for bs in BEAMS)
print(f'acoustic 1-best              CER {cer(ref,base):.2f} WER {wer(ref,base):.2f}',flush=True)
# oracle: best-CER beam per chunk (ceiling of rescoring)
orc=' '.join(min(bs,key=lambda t: cer(' '.join(w for w in [t[0]]) or ' ', t[0]))[0] if False else bs[0][0] for bs in BEAMS)
oracle=' '.join(min(bs,key=lambda bt: ed(co(bt[0]).replace(' ',''), '') )[0] for bs in BEAMS) if False else None
# real oracle vs ref per chunk not possible (ref not chunked); skip
# rescoring sweep
import itertools
best=None
for alpha in [0.6,1.0,1.4,2.0,2.8,4.0]:
    for beta in [0.0]:
        pick=[]
        for bs in BEAMS:
            scored=[(am+alpha*clm_score(tx)+beta*len(co(tx).replace(' ','')), tx) for tx,am in bs]
            pick.append(max(scored)[1])
        t=' '.join(pick); c=cer(ref,t); w=wer(ref,t)
        tag=f'a{alpha} b{beta}'
        if best is None or c<best[0]: best=(c,tag,t)
        print(f'rescore {tag:10s}          CER {c:.2f} WER {w:.2f}',flush=True)
print('BEST',best[1],round(best[0],2),flush=True)
json.dump({'text':best[2],'cfg':best[1],'cer':best[0]},open('/tmp/ait_char_out.json','w'),ensure_ascii=False)
