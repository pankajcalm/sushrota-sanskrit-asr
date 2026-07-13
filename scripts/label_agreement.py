"""Agreement-based pseudo-labeling of e-PG train clips.
Keep clips where v5 (IndicConformer-CTC) and Whisper-medium-ft AGREE (word-level),
gated by v5 confidence and corpus-validity. Label = v5 transcript. GPU."""
import os, json, glob, re, unicodedata, random, argparse
import numpy as np, soundfile as sf, torch, difflib
random.seed(0)
ROOT='/home/ece/BigDisk/Prathosh/ASR'; EP=f'{ROOT}/data/epgp'
ap=argparse.ArgumentParser()
ap.add_argument('--per_video', type=int, default=100)
ap.add_argument('--agree', type=float, default=0.80)     # min word-level agreement v5 vs whisper
ap.add_argument('--conf', type=float, default=0.55)      # min mean v5 word confidence
ap.add_argument('--corpus', type=float, default=0.60)    # min fraction of words in lexicon
ap.add_argument('--out', default=f'{EP}/manifest_epgp_pilot.jsonl')
A=ap.parse_args()
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
def ed(a,b):
    n,m=len(a),len(b)
    if n==0: return m
    if m==0: return n
    p=list(range(m+1))
    for i in range(1,n+1):
        c=[i]+[0]*m; ai=a[i-1]
        for j in range(1,m+1): c[j]=min(p[j]+1,c[j-1]+1,p[j-1]+(ai!=b[j-1]))
        p=c
    return p[m]
import pickle
LEX=pickle.load(open(f'{ROOT}/data/lexicon.pkl','rb'))
# ---- sample clips per video ----
clips=glob.glob(f'{EP}/train_clips/*.wav')
byv={}
for c in clips: byv.setdefault(os.path.basename(c).rsplit('_',1)[0],[]).append(c)
samp=[]
for v,cs in byv.items():
    random.shuffle(cs); samp+=cs[:A.per_video]
random.shuffle(samp)
print(f'scoring {len(samp)} clips from {len(byv)} videos', flush=True)
# ---- models on GPU ----
import nemo.collections.asr as nemo_asr
m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(f'{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo', map_location='cuda'); m.eval()
LABELS=json.load(open(f'{ROOT}/data/eval_logits/labels.json')); OFF,V,BLANK=4096,256,5632
def lse(a,ax): mx=a.max(ax,keepdims=True); return mx+np.log(np.exp(a-mx).sum(ax,keepdims=True))
def v5_decode(wav):
    sig=torch.tensor(wav).unsqueeze(0).cuda(); sl=torch.tensor([len(wav)]).cuda()
    with torch.no_grad():
        enc,_=m.forward(input_signal=sig,input_signal_length=sl)
        lp=m.ctc_decoder(encoder_output=enc)[0].detach().cpu().numpy()
    cols=[BLANK]+list(range(OFF,OFF+V)); P=lp[:,cols]; P=P-lse(P,1)
    ids=P.argmax(1); toks=[]; prev=-1
    for t,i in enumerate(ids):
        i=int(i)
        if i!=prev and i!=0: toks.append((int(i)-1, float(P[t,i])))
        prev=i
    text=''.join(LABELS[k] for k,_ in toks).replace('▁',' ').strip()
    conf=float(np.exp(np.mean([lp for _,lp in toks]))) if toks else 0.0
    return text, conf
from transformers import pipeline
asr=pipeline('automatic-speech-recognition', model=f'{ROOT}/exp/whisper_med_v5', device=0, torch_dtype=torch.float16, chunk_length_s=30)
def load(fp):
    x,sr=sf.read(fp,dtype='float32');  return x.mean(1) if x.ndim>1 else x
kept=[]; scored=0
BATCH=16
for i in range(0,len(samp),BATCH):
    batch=samp[i:i+BATCH]; wavs=[load(c) for c in batch]
    who=asr(wavs, batch_size=BATCH, generate_kwargs=dict(language='sanskrit',task='transcribe'))
    for c,wo in zip(batch,wavs and who):
        scored+=1
        v5t,conf=v5_decode(load(c)); wht=wo['text']
        vw=norm(v5t).split(); ww=norm(wht).split()
        if not vw: continue
        agree=1-ed(vw,ww)/max(len(vw),len(ww),1)
        corpus=sum(1 for w in vw if w in LEX)/len(vw)
        if agree>=A.agree and conf>=A.conf and corpus>=A.corpus:
            dur=len(load(c))/16000
            kept.append({'audio_filepath':c,'text':norm(v5t),'duration':round(dur,2),'lang':'sa',
                         'video':os.path.basename(c).rsplit('_',1)[0],'agree':round(agree,2),'conf':round(conf,2)})
    if (i//BATCH)%10==0: print(f'  {scored}/{len(samp)} scored, {len(kept)} kept', flush=True)
with open(A.out,'w') as f:
    for k in kept: f.write(json.dumps(k,ensure_ascii=False)+'\n')
hrs=sum(k['duration'] for k in kept)/3600
print(f'DONE scored {scored}, kept {len(kept)} ({100*len(kept)/max(scored,1):.0f}%), {hrs:.2f} h -> {A.out}')
