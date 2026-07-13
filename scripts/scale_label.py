"""Scaled clean-labeling: v5 whole-clip transcript per train clip, gated by confidence
+ corpus-validity (drops garbage/silence). Higher yield than agreement. GPU."""
import os, json, glob, re, unicodedata, argparse, pickle
import numpy as np, soundfile as sf, torch
ROOT='/home/ece/BigDisk/Prathosh/ASR'; EP=f'{ROOT}/data/epgp'
ap=argparse.ArgumentParser()
ap.add_argument('--conf', type=float, default=0.55); ap.add_argument('--corpus', type=float, default=0.70)
ap.add_argument('--out', default=f'{EP}/manifest_epgp_full.jsonl')
A=ap.parse_args()
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s); return re.sub(r'\s+',' ',s).strip()
LEX=pickle.load(open(f'{ROOT}/data/lexicon.pkl','rb'))
import nemo.collections.asr as nemo_asr
m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(f'{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo', map_location='cuda'); m.eval()
LABELS=json.load(open(f'{ROOT}/data/eval_logits/labels.json')); OFF,V,BLANK=4096,256,5632
def lse(a,ax): mx=a.max(ax,keepdims=True); return mx+np.log(np.exp(a-mx).sum(ax,keepdims=True))
def v5(wav):
    sig=torch.tensor(wav).unsqueeze(0).cuda(); sl=torch.tensor([len(wav)]).cuda()
    with torch.no_grad():
        enc,_=m.forward(input_signal=sig,input_signal_length=sl)
        lp=m.ctc_decoder(encoder_output=enc)[0].detach().cpu().numpy()
    cols=[BLANK]+list(range(OFF,OFF+V)); P=lp[:,cols]; P=P-lse(P,1); ids=P.argmax(1)
    out=[]; lps=[]; prev=-1
    for t,i in enumerate(ids):
        i=int(i)
        if i!=prev and i!=0: out.append(LABELS[i-1]); lps.append(float(P[t,i]))
        prev=i
    text=''.join(out).replace('▁',' ').strip()
    conf=float(np.exp(np.mean(lps))) if lps else 0.0
    return text,conf
clips=sorted(glob.glob(f'{EP}/train_clips/*.wav'))
print(f'labeling {len(clips)} clips', flush=True)
kept=[]; n=0
for c in clips:
    n+=1
    wav,sr=sf.read(c,dtype='float32');
    if wav.ndim>1: wav=wav.mean(1)
    d=len(wav)/16000
    if d<1.5 or d>16: continue
    text,conf=v5(wav); vt=norm(text).split()
    if len(vt)<3: continue
    corp=sum(1 for w in vt if w in LEX)/len(vt)
    if conf>=A.conf and corp>=A.corpus:
        kept.append({'audio_filepath':c,'text':norm(text),'duration':round(d,2),'lang':'sa',
                     'video':os.path.basename(c).rsplit('_',1)[0]})
    if n%2000==0: print(f'  {n}/{len(clips)}, {len(kept)} kept', flush=True)
with open(A.out,'w') as f:
    for k in kept: f.write(json.dumps(k,ensure_ascii=False)+'\n')
import collections; sp=collections.Counter(k['video'] for k in kept)
print(f'DONE kept {len(kept)}/{len(clips)} ({100*len(kept)/len(clips):.0f}%), {sum(k["duration"] for k in kept)/3600:.2f} h, {len(sp)} speakers -> {A.out}')
