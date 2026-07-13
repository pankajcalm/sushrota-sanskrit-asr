#!/usr/bin/env python3
"""Heavy multi-condition augmentation to manufacture pseudo-speakers + channels from
few-speaker data. Per clip, a randomized chain: speed-perturb (resample) + pitch-shift
+ codec round-trip + additive noise + gain. Replaces the light aug (1x) or multiplies (Nx).
Runs in nemo_ai4b (torch/torchaudio). No sox/MUSAN needed."""
import os, json, re, zlib, random, subprocess, tempfile, argparse
import numpy as np, soundfile as sf, torch, torchaudio.functional as AF
ROOT="/home/ece/BigDisk/Prathosh/ASR"; SRC=f"{ROOT}/data/tts16k"; OUT=f"{ROOT}/data/tts16k_heavy"
CODECS=[("opus","12k"),("opus","16k"),("opus","24k"),("mp3","32k"),("mp3","48k"),(None,None)]

def pink(n, rng):
    w=rng.standard_normal(n); f=np.fft.rfft(w)
    s=1.0/np.sqrt(np.arange(1,len(f)+1)); return np.fft.irfft(f*s, n)

def heavy(wav, cid, sr, tmp):
    rng=np.random.default_rng(zlib.crc32(cid.encode())); R=random.Random(zlib.crc32((cid+"r").encode()))
    x,_=sf.read(wav); x=x.astype(np.float32)
    if x.ndim>1: x=x.mean(1)
    # 1) speed-perturb (tempo+pitch) via linear resample  -> pseudo-speaker
    if R.random()<0.8:
        s=R.uniform(0.9,1.1); idx=np.arange(0,len(x),s); x=np.interp(idx,np.arange(len(x)),x).astype(np.float32)
    # 2) pitch-shift (independent) -> pseudo-speaker
    if R.random()<0.7:
        n_steps=R.uniform(-3,3)
        x=AF.pitch_shift(torch.tensor(x).unsqueeze(0), sr, n_steps=n_steps).squeeze(0).numpy()
    # 3) additive noise (pink) at random SNR -> channel
    if R.random()<0.7:
        snr=R.uniform(5,25); sigp=float(np.mean(x**2))+1e-9; npw=sigp/(10**(snr/10))
        nz=pink(len(x),rng); nz=nz/ (np.sqrt(np.mean(nz**2))+1e-9)*np.sqrt(npw); x=x+nz.astype(np.float32)
    # 4) gain jitter
    x=x*(10**(R.uniform(-5,3)/20)); pk=float(np.max(np.abs(x)))+1e-9
    if pk>0.99: x=x*(0.99/pk)
    x=np.clip(x,-1,1).astype(np.float32)
    # 5) codec round-trip -> channel
    kind,br=CODECS[R.randrange(len(CODECS))]
    src=f"{tmp}/a.wav"; sf.write(src,x,sr)
    if kind is not None:
        comp=f"{tmp}/c."+("ogg" if kind=="opus" else "mp3"); enc="libopus" if kind=="opus" else "libmp3lame"
        subprocess.run(["ffmpeg","-y","-v","error","-i",src,"-c:a",enc,"-b:a",br,"-ar","16000","-ac","1",comp],capture_output=True)
        dec=f"{tmp}/d.wav"; subprocess.run(["ffmpeg","-y","-v","error","-i",comp,"-ar","16000","-ac","1","-c:a","pcm_s16le",dec],capture_output=True)
        if os.path.exists(dec): x,_=sf.read(dec)
    return np.clip(np.asarray(x,dtype=np.float32),-1,1), f"{kind or 'clean'}{br or ''}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--speaker",required=True); ap.add_argument("--insuffix",default="6.0h")
    ap.add_argument("--variants",type=int,default=1); a=ap.parse_args()
    man=[json.loads(l) for l in open(f"{SRC}/manifest_{a.speaker}_{a.insuffix}.jsonl")]
    os.makedirs(f"{OUT}/{a.speaker}",exist_ok=True); rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        for r in man:
            for v in range(a.variants):
                cid=f"{r['src']}_h{v}"; outp=f"{OUT}/{a.speaker}/{cid}.wav"
                y,how=heavy(r["audio_filepath"], cid, 16000, tmp); sf.write(outp,y,16000)
                nr=dict(r); nr["audio_filepath"]=outp; nr["src"]=cid; nr["aug"]="heavy:"+how; rows.append(nr)
    mp=f"{SRC}/manifest_{a.speaker}_{a.insuffix}_heavy.jsonl"
    with open(mp,"w") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"{a.speaker}: {len(rows)} heavy-aug clips ({a.variants}x) -> {mp}", flush=True)

if __name__=="__main__": main()
