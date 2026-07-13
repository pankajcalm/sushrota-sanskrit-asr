#!/usr/bin/env python3
"""Diagnostic: for real annotated clips, dump per-akshara GOP AND, for each flagged akshara,
the real token the model prefers over the target at its aligned frames. Reveals whether flags
are nasal-equivalence (target ं but model emits म / vice-versa), alignment slips, or real."""
import os, sys, json, re, unicodedata
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as na
ROOT = "/home/ece/BigDisk/Prathosh/ASR"; OFF, V, BL = 4096, 256, 5632; NEG = -1e30
KEEP = re.compile(r'[^ऀ-ॿ\s]'); DROP = re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]'); NASAL = re.compile(r'म्(?=\s|[।॥]|$)')
def dedup(s):
    o=[]
    for c in s:
        if o and c==o[-1] and unicodedata.category(c) in ('Mn','Mc'): continue
        o.append(c)
    return ''.join(o)
def norm(s):
    s=unicodedata.normalize('NFC',s); s=NASAL.sub('ं',dedup(s)); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
def lse(x,ax): m=x.max(ax,keepdims=True); return m+np.log(np.exp(x-m).sum(ax,keepdims=True))
def is_base(ch):
    o=ord(ch); return (0x0905<=o<=0x0939) or (0x0958<=o<=0x0961) or (0x0972<=o<=0x097F)
M=na.models.EncDecHybridRNNTCTCBPEModel.restore_from(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo",map_location="cuda:0").eval()
SUB=M.tokenizer.tokenizers_dict["sa"]; LAB=json.load(open(f"{ROOT}/data/eval_logits/labels.json"))
_s2c={LAB[d]:d+1 for d in range(min(V,len(LAB)))}
SIB=np.full(V+1,-1,int)
for d in range(min(V,len(LAB))):
    s=LAB[d]; sib=s[1:] if s.startswith('▁') else '▁'+s; SIB[d+1]=_s2c.get(sib,-1)
NASAL_SURF={'ं','ँ','म्','न्','ण्','ञ्','ङ्'}
NASALCOLS=[c for s,c in _s2c.items() if s in NASAL_SURF]
EQUIV={}
for d in range(min(V,len(LAB))):
    c=d+1; eq={c}
    if SIB[c]>=0: eq.add(int(SIB[c]))
    if LAB[d] in NASAL_SURF: eq.update(NASALCOLS)
    EQUIV[c]=np.array(sorted(eq),int)
def surf(col): return LAB[col-1] if 1<=col<=len(LAB) else '?'
def post(wav):
    sig=torch.tensor(wav).unsqueeze(0).cuda(); sl=torch.tensor([len(wav)]).cuda()
    with torch.no_grad():
        enc,_=M.forward(input_signal=sig,input_signal_length=sl); lp=M.ctc_decoder(encoder_output=enc)[0].cpu().numpy()
    cols=[BL]+list(range(OFF,OFF+V)); P=lp[:,cols]; return P-lse(P,1)
def cluster(t):
    chars=[]
    for j,s in enumerate(SUB.text_to_tokens(t)):
        for ch in s: chars.append((' ',-1) if ch=='▁' else (ch,j))
    out=[]; cur=""; tks=[]; pv=False
    def fl(we):
        nonlocal cur,tks
        if cur: out.append({"text":cur,"toks":sorted(set(tks))}); cur="";tks=[]
    for ch,tj in chars:
        if ch==' ': fl(True); pv=False; continue
        if is_base(ch) and cur and not pv: fl(False); cur=ch; tks=[tj] if tj>=0 else []
        else:
            cur+=ch
            if tj>=0: tks.append(tj)
        pv=(ord(ch)==0x094D)
    fl(True)
    m=[]
    for a in out:
        if m and a["text"].endswith("्"): m[-1]["text"]+=a["text"]; m[-1]["toks"]=sorted(set(m[-1]["toks"])|set(a["toks"]))
        else: m.append(a)
    return m
def align(P,ids):
    T=P.shape[0]; cols=[d+1 for d in ids if 0<=d<V]; L=len(cols)
    if L==0 or T<L: return None
    ext=[0]
    for c in cols: ext+=[c,0]
    ext=np.array(ext); S=len(ext); topcol=P[:,1:].max(1); topidx=P[:,1:].argmax(1)+1
    flp=P[:,ext].astype(float,copy=True)
    for k in range(S):
        c=ext[k]
        if c!=0: flp[:,k]=P[:,EQUIV[c]].max(1)
    allow=np.zeros(S,bool); allow[2:]=(ext[2:]!=0)&(ext[2:]!=ext[:-2])
    dp=np.full((T,S),NEG); bp=np.zeros((T,S),np.int32); dp[0,0]=flp[0,0]
    if S>1: dp[0,1]=flp[0,1]
    for t in range(1,T):
        stay=dp[t-1]; prev=np.full(S,NEG); prev[1:]=dp[t-1,:-1]; skip=np.full(S,NEG); skip[2:]=dp[t-1,:-2]; skip[~allow]=NEG
        cand=np.stack([stay,prev,skip]); arg=cand.argmax(0); dp[t]=cand.max(0)+flp[t]; bp[t]=np.arange(S)-arg
    s=S-1 if dp[T-1,S-1]>=dp[T-1,S-2] else S-2
    if dp[T-1,s]<=NEG/2: return None
    path=np.empty(T,np.int32)
    for t in range(T-1,-1,-1): path[t]=s; s=bp[t,s]
    per=[]
    for j in range(L):
        fr=np.where(path==2*j+1)[0]
        if len(fr)==0: per.append({"gop":NEG,"nf":0,"comp":"-","tcol":cols[j]}); continue
        vals=flp[fr,2*j+1]-topcol[fr]; wf=fr[int(vals.argmin())]
        per.append({"gop":float(vals.mean()),"nf":len(fr),"tcol":cols[j],
                    "comp":surf(topidx[wf]),"tgt":surf(cols[j]),
                    "tlp":float(P[wf,cols[j]]),"clp":float(topcol[wf])})
    return per
# scan annotated clips; show aksharas with anusvara or flagged
last={}
for l in open(f"{ROOT}/data/epgp/annot/annot_refs.jsonl"):
    try: r=json.loads(l)
    except: continue
    last[r["id"]]=r
shown=0
for r in last.values():
    if not r.get("text","").strip() or r.get("unclear"): continue
    p=f"{ROOT}/data/epgp/annot_clips/{r['id']}.wav"
    if not os.path.exists(p): continue
    t=norm(r["text"])
    if 'ं' not in t: continue
    wav,_=sf.read(p,dtype="float32");
    if wav.ndim>1: wav=wav.mean(1)
    wav=np.concatenate([np.zeros(4800,np.float32),wav.astype(np.float32),np.zeros(4800,np.float32)])
    ids=SUB.text_to_ids(t); per=align(post(wav),ids)
    if not per: continue
    aks=cluster(t); gd={j:per[j] for j in range(len(per))}
    print(f"\n{r['id']}  {t[:50]}")
    for a in aks:
        vals=[(j,gd[j]) for j in a["toks"] if j in gd and gd[j]["nf"]>0]
        if not vals: continue
        j,g=min(vals,key=lambda x:x[1]["gop"]); mn=g["gop"]
        anu='ं' in a["text"]
        if mn<-0.5 or anu:
            tag='ANUSVARA' if anu else ''
            print(f"   {a['text']:8s} gop={mn:6.2f} nf={g['nf']:2d} tgt={g.get('tgt','?')!r} vs top={g['comp']!r} (tlp={g.get('tlp',0):.2f} clp={g.get('clp',0):.2f}) {tag}")
    shown+=1
    if shown>=12: break
print("\nDEBUG DONE",flush=True)
