import json,glob,re,unicodedata,numpy as np
from pyctcdecode import build_ctcdecoder
sv=json.load(open('/tmp/sa_vocab.json')); off=sv['offset']; V=sv['V']; sa_tokens=sv['tokens']
BLANK=sv['total']  # 5632
cols=list(range(off,off+V))+[BLANK]
labels=sa_tokens+['']              # 257, blank last
logf=sorted(glob.glob('/tmp/ait_logits/chunk_*.npy'))
def logsoftmax(x):
    x=x-x.max(-1,keepdims=True); e=np.exp(x); return np.log(e/e.sum(-1,keepdims=True)+1e-12)
LOG=[logsoftmax(np.load(f).astype(np.float32)[:,cols]) for f in logf]
ref=open('/tmp/ref.txt').read()
def co(s):
    s=unicodedata.normalize('NFC',s).replace('ॐ',' ')
    s=re.sub(r'[०-९।॥ऽ]',' ',s); s=re.sub(r'[^ऀ-ॿ\s]',' ',s)
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
def rates(r,h):
    r=co(r); h=co(h); rc,hc=r.replace(' ',''),h.replace(' ','')
    return ed(rc,hc)/max(1,len(rc))*100, ed(r.split(),h.split())/max(1,len(r.split()))*100
bl=len(labels)-1
def greedy(lp):
    ids=lp.argmax(-1); out=[]; prev=-1
    for i in ids:
        if i!=prev and i!=bl: out.append(labels[i])
        prev=i
    return ''.join(out).replace('▁',' ').strip()
g=' '.join(greedy(l) for l in LOG); cer,wer=rates(ref,g)
print(f'greedy on sa-slice (sanity)  CER {cer:.2f} WER {wer:.2f}',flush=True)
def dec(a,b,lm):
    d=build_ctcdecoder(labels, kenlm_model_path=lm, alpha=a, beta=b)
    return ' '.join(d.decode(l, beam_width=100) for l in LOG)
best=None
for name,a,b,lm in [('beam no-LM',0,0,None),('+LM a0.3',0.3,0.5,'/tmp/sa.bin'),
                    ('+LM a0.6',0.6,0.5,'/tmp/sa.bin'),('+LM a1.0',1.0,0.5,'/tmp/sa.bin'),
                    ('+LM a1.5',1.5,0.5,'/tmp/sa.bin')]:
    t=dec(a,b,lm); cer,wer=rates(ref,t); print(f'{name:12s} CER {cer:.2f} WER {wer:.2f}',flush=True)
    if best is None or cer<best[0]: best=(cer,name,t)
json.dump({'text':best[2],'cfg':best[1],'cer':best[0]},open('/tmp/ait_lm_out.json','w'),ensure_ascii=False)
print('BEST',best[1],round(best[0],2),flush=True)
