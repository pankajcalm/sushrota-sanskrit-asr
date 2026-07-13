"""Phone/voice-note channel simulation on the gold clips + v5 CER vs same gold refs."""
import json, re, unicodedata, requests, subprocess, os
ROOT='/home/ece/BigDisk/Prathosh/ASR'; EP=f'{ROOT}/data/epgp'; PH=f'{EP}/phone_gold'; os.makedirs(PH,exist_ok=True)
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
DN=open(os.devnull,'w')
def phone(ap,op):
    o=f'{op[:-4]}.opus'
    subprocess.run(['ffmpeg','-y','-i',ap,'-af','highpass=f=120,acompressor=threshold=-18dB:ratio=3:attack=5:release=50,volume=1.2','-c:a','libopus','-b:a','16k',o],stderr=DN,stdout=DN)
    subprocess.run(['ffmpeg','-y','-i',o,'-ar','16000','-ac','1',op],stderr=DN,stdout=DN)
    return os.path.exists(op)
gold={json.loads(l)['id']:json.loads(l)['text'] for l in open(f'{EP}/gold_refs.jsonl')}
man={json.loads(l)['id']:json.loads(l)['audio_filepath'] for l in open(f'{EP}/manifest_gold.jsonl')}
ce=cn=we=wn=n=fail=0
for gid,ref in gold.items():
    rn=norm(ref)
    if rn=='' or ref.strip()=='[x]': continue
    ap=man.get(gid)
    if not ap: continue
    op=f'{PH}/{gid}.wav'
    if not phone(ap,op): fail+=1; continue
    with open(op,'rb') as f:
        hyp=requests.post('http://127.0.0.1:8000/transcribe',files={'audio':f},data={'interim':'true'},timeout=30).json().get('raw_text','')
    hn=norm(hyp); ce+=ed(hn.replace(' ',''),rn.replace(' ','')); cn+=len(rn.replace(' ',''))
    we+=ed(hn.split(),rn.split()); wn+=len(rn.split()); n+=1
print(f'== v5 on PHONE-augmented e-PG gold ({n} clips, {fail} ffmpeg-fail) ==')
print(f'   content-only CER {100*ce/cn:.2f}%   WER {100*we/wn:.2f}%   (clean was 3.59% / 12.97%)')
