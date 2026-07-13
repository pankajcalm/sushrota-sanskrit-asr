import os, json, argparse
import soundfile as sf
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
ROOT='/home/ece/BigDisk/Prathosh/ASR/data/epgp'
ap=argparse.ArgumentParser()
ap.add_argument('--ids', required=True); ap.add_argument('--outdir', required=True); ap.add_argument('--manifest', required=True)
A=ap.parse_args()
os.makedirs(A.outdir, exist_ok=True)
m=load_silero_vad()
ids=[l.strip() for l in open(A.ids) if l.strip()]
n=0; tot=0.0
with open(A.manifest,'w') as mf:
  for vid in ids:
    p=f'{ROOT}/wav16k/{vid}.wav'
    if not os.path.exists(p): continue
    wav=read_audio(p, sampling_rate=16000)
    ts=get_speech_timestamps(wav, m, sampling_rate=16000, max_speech_duration_s=15, min_silence_duration_ms=300, threshold=0.5)
    arr=wav.numpy()
    for i,seg in enumerate(ts):
      s,e=seg['start'],seg['end']; dur=(e-s)/16000
      if dur<1.0 or dur>20: continue
      cp=f'{A.outdir}/{vid}_{i:04d}.wav'
      sf.write(cp, arr[s:e], 16000)
      mf.write(json.dumps({'audio_filepath':cp,'video':vid,'start':round(s/16000,2),'dur':round(dur,2)})+'\n')
      n+=1; tot+=dur
    print(f'  {vid}: done', flush=True)
print(f'SEGDONE segmented {n} clips, {tot/3600:.2f} h from {len(ids)} videos')
