import json, requests
ROOT='/home/ece/BigDisk/Prathosh/ASR/data/epgp'
rows=[json.loads(l) for l in open(f'{ROOT}/manifest_gold.jsonl')]
n=0
for g in rows:
    if not g.get('prefill','').strip():
        try:
            with open(g['audio_filepath'],'rb') as f:
                r=requests.post('http://127.0.0.1:8000/transcribe', files={'audio':f}, data={'interim':'true'}, timeout=30)
            g['prefill']=r.json().get('raw_text','')
        except Exception: g['prefill']=''
        g['blank']=False; n+=1
with open(f'{ROOT}/manifest_gold.jsonl','w') as mf:
    for g in rows: mf.write(json.dumps(g,ensure_ascii=False)+'\n')
empty=sum(1 for g in rows if not g.get('prefill','').strip())
print(f'FILLED {n} previously-blank clips; remaining empty: {empty}; total {len(rows)}')
