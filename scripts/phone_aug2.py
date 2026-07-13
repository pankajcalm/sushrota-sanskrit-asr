import json, os, subprocess, argparse
from concurrent.futures import ThreadPoolExecutor
ap=argparse.ArgumentParser(); ap.add_argument('--inmanifest',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--outmanifest',required=True)
A=ap.parse_args(); os.makedirs(A.outdir,exist_ok=True); DN=open(os.devnull,'w')
rows=[json.loads(l) for l in open(A.inmanifest)]
def aug(r):
    ap_=r['audio_filepath']; op=f"{A.outdir}/{os.path.basename(ap_)}"; o=f"{op[:-4]}.opus"
    subprocess.run(['ffmpeg','-y','-i',ap_,'-af','highpass=f=120,acompressor=threshold=-18dB:ratio=3:attack=5:release=50,volume=1.2','-c:a','libopus','-b:a','16k',o],stderr=DN,stdout=DN)
    subprocess.run(['ffmpeg','-y','-i',o,'-ar','16000','-ac','1',op],stderr=DN,stdout=DN)
    if os.path.exists(o): os.remove(o)
    return {'audio_filepath':op,'text':r['text'],'duration':r['duration'],'lang':'sa'} if os.path.exists(op) else None
with ThreadPoolExecutor(max_workers=8) as ex:
    res=[r for r in ex.map(aug, rows) if r]
with open(A.outmanifest,'w') as f:
    for r in res: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(f'PHONEAUG {len(res)}/{len(rows)}')
