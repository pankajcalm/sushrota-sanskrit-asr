import soundfile as sf, glob, os, json
import numpy as np
d='/home/ece/BigDisk/Prathosh/ASR/data/flywheel'
segs={}
for l in open(f'{d}/events.jsonl'):
    try: r=json.loads(l)
    except: continue
    if r.get('kind')=='segment' and r.get('audio'):
        segs[os.path.basename(r['audio'])]=r.get('raw','')
files=sorted(glob.glob(f'{d}/audio/*.wav'))
print(f"{'file':22} {'sr':>6} {'dur':>5} {'rms':>6}  raw_output")
for f in files:
    x,sr=sf.read(f)
    if x.ndim>1: x=x.mean(1)
    r=float(np.sqrt((x.astype('float64')**2).mean())) if len(x) else 0.0
    print(f"{os.path.basename(f):22} {sr:6d} {len(x)/sr:5.1f} {r:6.3f}  {segs.get(os.path.basename(f),'(no seg raw)')[:46]}")
print(f"\n{len(files)} audio clips; {len(segs)} with logged raw")
