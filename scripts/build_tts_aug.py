#!/usr/bin/env python3
"""Channel-match the clean studio TTS to the crowd-sourced recitation/voice-note domain.
Per clip (deterministic seed): additive noise @ random SNR + gain jitter + codec round-trip
(Opus/MP3 at varied low bitrate, some left clean). Replaces TTS 1:1 (keeps 16% balance).
Emits data/tts16k_aug/<spk>/*.wav + manifest_<spk>_1.0h_aug.jsonl.
"""
import os, json, glob, zlib, random, subprocess, tempfile
import numpy as np, soundfile as sf
ROOT="/home/ece/BigDisk/Prathosh/ASR"
SRC=f"{ROOT}/data/tts16k"; OUT=f"{ROOT}/data/tts16k_aug"
CODECS=[("opus","12k"),("opus","16k"),("opus","24k"),("mp3","32k"),("mp3","48k"),
        (None,None),(None,None)]   # ~30% left uncompressed -> varied like recitation

def codec_roundtrip(y, sr, kind, br, tmp):
    src=f"{tmp}/a.wav"; sf.write(src, y, sr)
    if kind is None:
        z,_=sf.read(src); return z
    comp=f"{tmp}/c." + ("ogg" if kind=="opus" else "mp3")
    enc="libopus" if kind=="opus" else "libmp3lame"
    subprocess.run(["ffmpeg","-y","-v","error","-i",src,"-c:a",enc,"-b:a",br,
                    "-ar","16000","-ac","1",comp], capture_output=True)
    dec=f"{tmp}/d.wav"
    subprocess.run(["ffmpeg","-y","-v","error","-i",comp,"-ar","16000","-ac","1",
                    "-c:a","pcm_s16le",dec], capture_output=True)
    if not os.path.exists(dec): z,_=sf.read(src); return z
    z,_=sf.read(dec); return z

def augment(wav, cid, tmp):
    rng=random.Random(zlib.crc32(cid.encode()))       # stable per-clip
    x,sr=sf.read(wav)
    if x.ndim>1: x=x.mean(1)
    x=x.astype(np.float32)
    snr=rng.uniform(10,28)
    sigp=float(np.mean(x**2))+1e-9
    npow=sigp/(10**(snr/10))
    x=x+np.random.default_rng(zlib.crc32((cid+"n").encode())).normal(0,np.sqrt(npow),x.shape).astype(np.float32)
    x=x*(10**(rng.uniform(-4,2)/20))                  # gain jitter
    peak=float(np.max(np.abs(x)))+1e-9
    if peak>0.99: x=x*(0.99/peak)
    kind,br=CODECS[rng.randrange(len(CODECS))]
    y=codec_roundtrip(x, sr, kind, br, tmp)
    return np.clip(y,-1,1).astype(np.float32), sr, f"{kind or 'clean'}{br or ''}@snr{snr:.0f}"

def main():
    import argparse
    from collections import Counter
    ap=argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True); ap.add_argument("--insuffix", default="1.0h")
    a=ap.parse_args()
    spk=a.speaker
    man=[json.loads(l) for l in open(f"{SRC}/manifest_{spk}_{a.insuffix}.jsonl")]
    os.makedirs(f"{OUT}/{spk}", exist_ok=True)
    rows=[]; mix=Counter()
    with tempfile.TemporaryDirectory() as tmp:
        for r in man:
            cid=r["src"]; outp=f"{OUT}/{spk}/{cid}.wav"
            y,sr,how=augment(r["audio_filepath"], cid, tmp)
            sf.write(outp, y, sr)
            nr=dict(r); nr["audio_filepath"]=outp; nr["aug"]=how
            rows.append(nr); mix[how.split("@")[0]]+=1
    mp=f"{SRC}/manifest_{spk}_{a.insuffix}_aug.jsonl"
    with open(mp,"w") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"{spk}: {len(rows)} clips augmented -> {mp}")
    print("  codec mix:", dict(mix.most_common()))

if __name__=="__main__": main()
