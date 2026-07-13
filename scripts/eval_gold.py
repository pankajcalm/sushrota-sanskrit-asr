"""Zero-shot CER of v5 on the human-corrected e-PG gold set (content-only)."""
import json, re, unicodedata, requests, glob
ROOT='/home/ece/BigDisk/Prathosh/ASR'; EP=f'{ROOT}/data/epgp'
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
def ed(a,b):
    n,m=len(a),len(b)
    if n==0:return m
    if m==0:return n
    p=list(range(m+1))
    for i in range(1,n+1):
        c=[i]+[0]*m;ai=a[i-1]
        for j in range(1,m+1):c[j]=min(p[j]+1,c[j-1]+1,p[j-1]+(ai!=b[j-1]))
        p=c
    return p[m]
gold={json.loads(l)['id']:json.loads(l)['text'] for l in open(f'{EP}/gold_refs.jsonl')}
man={json.loads(l)['id']:json.loads(l)['audio_filepath'] for l in open(f'{EP}/manifest_gold.jsonl')}
ce=cn=we=wn=0; n=0; skipped=0
byspk_ce={}; byspk_cn={}
for gid,ref in gold.items():
    rn=norm(ref)
    if rn=='' or ref.strip()=='[x]': skipped+=1; continue
    ap=man.get(gid)
    if not ap: continue
    with open(ap,'rb') as f:
        hyp=requests.post('http://127.0.0.1:8000/transcribe',files={'audio':f},data={'interim':'true'},timeout=30).json().get('raw_text','')
    hn=norm(hyp); hc,rc=hn.replace(' ',''),rn.replace(' ','')
    ce+=ed(hc,rc); cn+=len(rc); we+=ed(hn.split(),rn.split()); wn+=len(rn.split()); n+=1
    sp=gid  # per-clip; group by video via manifest
print(f'== v5 zero-shot on e-PG gold ({n} clips, {skipped} skipped) ==')
print(f'   content-only CER {100*ce/cn:.2f}%   WER {100*we/wn:.2f}%')
