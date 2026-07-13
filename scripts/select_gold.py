import json, glob, random, os, requests, soundfile as sf
random.seed(0)
ROOT='/home/ece/BigDisk/Prathosh/ASR/data/epgp'
clips=glob.glob(f'{ROOT}/eval_clips/*.wav')
byspk={}
for c in clips:
    vid=os.path.basename(c).rsplit('_',1)[0]
    byspk.setdefault(vid,[]).append(c)
gold=[]; PER=7.5*60
for vid,cs in byspk.items():
    random.shuffle(cs); tot=0
    for c in cs:
        try: d=sf.info(c).duration
        except: continue
        if d<2 or d>15: continue
        gold.append({'video':vid,'audio_filepath':c,'dur':round(d,2)}); tot+=d
        if tot>=PER: break
random.shuffle(gold)
for i,g in enumerate(gold):
    g['id']=f'g{i:04d}'; g['blank']=(i%10==0)
for g in gold:
    if g['blank']: g['prefill']=''; continue
    try:
        with open(g['audio_filepath'],'rb') as f:
            r=requests.post('http://127.0.0.1:8000/transcribe', files={'audio':f}, data={'interim':'true'}, timeout=30)
        g['prefill']=r.json().get('raw_text','')
    except Exception: g['prefill']=''
with open(f'{ROOT}/manifest_gold.jsonl','w') as mf:
    for g in gold: mf.write(json.dumps(g,ensure_ascii=False)+'\n')
tot=sum(g['dur'] for g in gold); blanks=sum(1 for g in gold if g['blank'])
print(f'GOLD {len(gold)} clips, {tot/60:.1f} min, {len(byspk)} speakers, {blanks} blank')
