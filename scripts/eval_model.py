import os, json, re, unicodedata, argparse
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as nemo_asr
ROOT='/home/ece/BigDisk/Prathosh/ASR'
ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--tag',default='')
A=ap.parse_args()
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s); return re.sub(r'\s+',' ',s).strip()
def ed(a,b):
    n,mm=len(a),len(b)
    if n==0:return mm
    if mm==0:return n
    p=list(range(mm+1))
    for i in range(1,n+1):
        c=[i]+[0]*mm; ai=a[i-1]
        for j in range(1,mm+1): c[j]=min(p[j]+1,c[j-1]+1,p[j-1]+(ai!=b[j-1]))
        p=c
    return p[mm]
def spaceless_word_errors(hyp, ref):
    """Sandhi/segmentation-free word error: strip spaces from both, align at char level,
    a ref word is 'bad' if any char is subbed/deleted (strict also charges insertions to the
    preceding word). Removes the word-boundary degree of freedom that inflates Sanskrit WER.
    Returns (n_words, bad_strict, bad_subdel, n_ins)."""
    rw=ref.split()
    if not rw: return (0,0,0,0)
    R=[]; owner=[]
    for wi,w in enumerate(rw):
        for ch in w: R.append(ch); owner.append(wi)
    H=list(''.join(hyp.split())); n,mm=len(R),len(H)
    dp=[[0]*(mm+1) for _ in range(n+1)]; bt=[['']*(mm+1) for _ in range(n+1)]
    for i in range(1,n+1): dp[i][0]=i; bt[i][0]='D'
    for j in range(1,mm+1): dp[0][j]=j; bt[0][j]='I'
    for i in range(1,n+1):
        Ri=R[i-1]
        for j in range(1,mm+1):
            if Ri==H[j-1]: dp[i][j]=dp[i-1][j-1]; bt[i][j]='M'
            else:
                sub,dele,ins=dp[i-1][j-1]+1,dp[i-1][j]+1,dp[i][j-1]+1
                best=min(sub,dele,ins); dp[i][j]=best
                bt[i][j]='S' if best==sub else ('D' if best==dele else 'I')
    ds=[False]*len(rw); dsd=[False]*len(rw); ni=0; i,j=n,mm
    while i>0 or j>0:
        op=bt[i][j]
        if op=='M': i-=1; j-=1
        elif op=='S': ds[owner[i-1]]=True; dsd[owner[i-1]]=True; i-=1; j-=1
        elif op=='D': ds[owner[i-1]]=True; dsd[owner[i-1]]=True; i-=1
        else:
            w=owner[i-1] if i>=1 else owner[0]; ds[w]=True; ni+=1; j-=1
    return (len(rw), sum(ds), sum(dsd), ni)
dev='cuda' if torch.cuda.is_available() else 'cpu'
m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(A.model, map_location=dev); m.eval()
LABELS=json.load(open(f'{ROOT}/data/eval_logits/labels.json')); OFF,V,BLANK=4096,256,5632
def lse(a,ax): mx=a.max(ax,keepdims=True); return mx+np.log(np.exp(a-mx).sum(ax,keepdims=True))
def greedy(wav):
    sig=torch.tensor(wav).unsqueeze(0).to(dev); sl=torch.tensor([len(wav)]).to(dev)
    with torch.no_grad():
        enc,_=m.forward(input_signal=sig,input_signal_length=sl)
        lp=m.ctc_decoder(encoder_output=enc)[0].detach().cpu().numpy()
    cols=[BLANK]+list(range(OFF,OFF+V)); P=lp[:,cols]; P=P-lse(P,1); ids=P.argmax(1)
    out=[]; prev=-1
    for i in ids:
        i=int(i)
        if i!=prev and i!=0: out.append(LABELS[i-1])
        prev=i
    return ''.join(out).replace('▁',' ').strip()
rows=[json.loads(l) for l in open(A.manifest)]
ce=cn=we=wn=0; W=BADS=BADSD=INS=0
for r in rows:
    wav,sr=sf.read(r['audio_filepath'],dtype='float32')
    if wav.ndim>1: wav=wav.mean(1)
    hn=norm(greedy(wav)); rn=norm(r['text'])
    ce+=ed(hn.replace(' ',''),rn.replace(' ','')); cn+=len(rn.replace(' ',''))
    we+=ed(hn.split(),rn.split()); wn+=len(rn.split())
    nw,bs,bsd,ni=spaceless_word_errors(hn,rn); W+=nw; BADS+=bs; BADSD+=bsd; INS+=ni
snw=100*BADSD/W; snws=100*BADS/W; swer=100*we/wn
print(f'{A.tag}: CER {100*ce/cn:.2f}%  WER {swer:.2f}%  SN-WER {snw:.2f}%..{snws:.2f}%  ({len(rows)} clips, seg-share {100*(swer-snw)/swer:.0f}-{100*(swer-snws)/swer:.0f}%)')
