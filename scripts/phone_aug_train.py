import json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
ROOT='/home/ece/BigDisk/Prathosh/ASR'; EP=f'{ROOT}/data/epgp'; OUT=f'{EP}/phone_train_clips'
os.makedirs(OUT, exist_ok=True); DN=open(os.devnull,'w')
rows=[json.loads(l) for l in open(f'{EP}/manifest_epgp_train.jsonl')]
def aug(r):
    ap=r['audio_filepath']; op=f"{OUT}/{os.path.basename(ap)}"; o=f"{op[:-4]}.opus"
    subprocess.run(['ffmpeg','-y','-i',ap,'-af','highpass=f=120,acompressor=threshold=-18dB:ratio=3:attack=5:release=50,volume=1.2','-c:a','libopus','-b:a','16k',o],stderr=DN,stdout=DN)
    subprocess.run(['ffmpeg','-y','-i',o,'-ar','16000','-ac','1',op],stderr=DN,stdout=DN)
    if os.path.exists(o): os.remove(o)
    return {'audio_filepath':op,'text':r['text'],'duration':r['duration'],'lang':'sa'} if os.path.exists(op) else None
with ThreadPoolExecutor(max_workers=8) as ex:
    res=[r for r in ex.map(aug, rows) if r]
with open(f'{EP}/manifest_epgp_train_phone.jsonl','w') as f:
    for r in res: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(f'PHONEAUG {len(res)}/{len(rows)} clips -> manifest_epgp_train_phone.jsonl')
