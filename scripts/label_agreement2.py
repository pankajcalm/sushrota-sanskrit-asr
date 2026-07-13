"""Word-level agreement re-cut: keep the spans where v5 & Whisper-ft agree, slice
audio to those spans (v5 word timestamps), label = v5 text. Much higher yield."""
import os, json, glob, re, unicodedata, random, argparse
import numpy as np, soundfile as sf, torch, difflib
random.seed(0)
ROOT='/home/ece/BigDisk/Prathosh/ASR'; EP=f'{ROOT}/data/epgp'
ap=argparse.ArgumentParser()
ap.add_argument('--per_video', type=int, default=120)
ap.add_argument('--conf', type=float, default=0.55)
ap.add_argument('--corpus', type=float, default=0.6)
ap.add_argument('--minwords', type=int, default=3)
ap.add_argument('--outdir', default=f'{EP}/lab_clips')
ap.add_argument('--out', default=f'{EP}/manifest_epgp_pilot.jsonl')
A=ap.parse_args(); os.makedirs(A.outdir, exist_ok=True)
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
import pickle; LEX=pickle.load(open(f'{ROOT}/data/lexicon.pkl','rb'))
clips=glob.glob(f'{EP}/train_clips/*.wav'); byv={}
for c in clips: byv.setdefault(os.path.basename(c).rsplit('_',1)[0],[]).append(c)
samp=[]
for v,cs in byv.items(): random.shuffle(cs); samp+=cs[:A.per_video]
random.shuffle(samp)
print(f'scoring {len(samp)} clips from {len(byv)} videos', flush=True)
import nemo.collections.asr as nemo_asr
m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(f'{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo', map_location='cuda'); m.eval()
LABELS=json.load(open(f'{ROOT}/data/eval_logits/labels.json')); OFF,V,BLANK=4096,256,5632
def lse(a,ax): mx=a.max(ax,keepdims=True); return mx+np.log(np.exp(a-mx).sum(ax,keepdims=True))
def v5_words(wav):
    sig=torch.tensor(wav).unsqueeze(0).cuda(); sl=torch.tensor([len(wav)]).cuda()
    with torch.no_grad():
        enc,_=m.forward(input_signal=sig,input_signal_length=sl)
        lp=m.ctc_decoder(encoder_output=enc)[0].detach().cpu().numpy()
    cols=[BLANK]+list(range(OFF,OFF+V)); P=lp[:,cols]; P=P-lse(P,1)
    T=P.shape[0]; spf=len(wav)/max(T,1); ids=P.argmax(1)
    em=[]; prev=-1
    for t,i in enumerate(ids):
        i=int(i)
        if i!=prev and i!=0: em.append((t,int(i)-1,float(P[t,i])))
        prev=i
    words=[]; cur=[]
    def flush(endf):
        if cur:
            txt=''.join(LABELS[k] for _,k,_ in cur).replace('▁',' ').strip()
            words.append({'t':norm(txt),'s':cur[0][0],'e':endf,'c':float(np.exp(np.mean([lp for _,_,lp in cur])))})
    for (t,k,lp) in em:
        if LABELS[k].startswith('▁') and cur: flush(t); cur=[]
        cur.append((t,k,lp))
    flush(T)
    return [w for w in words if w['t']], spf
from transformers import pipeline
asr=pipeline('automatic-speech-recognition', model=f'{ROOT}/exp/whisper_med_v5', device=0, torch_dtype=torch.float16, chunk_length_s=30)
def load(fp):
    x,sr=sf.read(fp,dtype='float32'); return x.mean(1) if x.ndim>1 else x
kept=[]; scored=0; nrun=0
for bi in range(0,len(samp),16):
    batch=samp[bi:bi+16]; wavs=[load(c) for c in batch]
    who=asr(wavs, batch_size=16, generate_kwargs=dict(language='sanskrit',task='transcribe'))
    for c,wav,wo in zip(batch,wavs,who):
        scored+=1
        vw,spf=v5_words(wav); vt=[w['t'] for w in vw]; ww=norm(wo['text']).split()
        if not vt: continue
        for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(a=vt,b=ww,autojunk=False).get_opcodes():
            if tag!='equal' or (i2-i1)<A.minwords: continue
            run=vw[i1:i2]; text=' '.join(w['t'] for w in run)
            conf=float(np.mean([w['c'] for w in run]))
            corp=sum(1 for w in run if w['t'] in LEX)/len(run)
            if conf<A.conf or corp<A.corpus: continue
            s=max(0,int(run[0]['s']*spf)-1600); e=min(len(wav),int(run[-1]['e']*spf)+1600)
            dur=(e-s)/16000
            if dur<1.0 or dur>18: continue
            op=f"{A.outdir}/{os.path.basename(c)[:-4]}_r{i1}.wav"; sf.write(op, wav[s:e], 16000)
            kept.append({'audio_filepath':op,'text':text,'duration':round(dur,2),'lang':'sa',
                         'video':os.path.basename(c).rsplit('_',1)[0],'conf':round(conf,2)}); nrun+=1
    if (bi//16)%10==0: print(f'  {scored}/{len(samp)} scored, {nrun} runs kept', flush=True)
with open(A.out,'w') as f:
    for k in kept: f.write(json.dumps(k,ensure_ascii=False)+'\n')
hrs=sum(k['duration'] for k in kept)/3600
print(f'DONE scored {scored}, {nrun} agreed-runs, {hrs:.2f} h -> {A.out}')
