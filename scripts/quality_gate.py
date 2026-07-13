#!/usr/bin/env python3
"""Quality-gate v2: per-work, svara-stripped, short-blocks filtered.
Reports CER (AM-weakness confounded) AND overlap-ratio (difflib) which measures whether the
audio actually CONTAINS the label — the real alignment check. high overlap+high CER = aligned
but hard (KEEP, learning headroom); low overlap = misaligned (REJECT)."""
import json, re, unicodedata, difflib
from collections import defaultdict
import nemo.collections.asr as nemo_asr
ROOT="/home/ece/BigDisk/Prathosh/ASR"
MAN=f"{ROOT}/data/disk_utts/manifest_disk.jsonl"
MODEL=f"{ROOT}/exp/ft_ctc_v2/ft_ctc_ep20.nemo"
KEEP=re.compile(r'[^ऀ-ॿ\s]')
DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')  # + Vedic svara/accents
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
def edist(a,b):
    n,m=len(a),len(b)
    if n==0:return m
    if m==0:return n
    p=list(range(m+1))
    for i in range(1,n+1):
        c=[i]+[0]*m;ai=a[i-1]
        for j in range(1,m+1):c[j]=min(p[j]+1,c[j-1]+1,p[j-1]+(ai!=b[j-1]))
        p=c
    return p[m]
man=[json.loads(l) for l in open(MAN)]
byw=defaultdict(list)
for u in man:
    if len(norm(u["text"]).replace(" ",""))>=10:   # skip tiny colophon/marker blocks
        byw[u["work"]].append(u)
sample=[]
for w,us in byw.items():
    step=max(1,len(us)//12); sample+=us[::step][:12]
m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL,map_location="cuda"); m.eval()
m.change_decoding_strategy(decoder_type="ctc")
res=m.transcribe([u["audio_filepath"] for u in sample],batch_size=16,language_id="sa")
if isinstance(res,tuple):res=res[0]
hyps=[norm(h if isinstance(h,str) else getattr(h,"text",str(h))) for h in res]
Ce=defaultdict(int);Cn=defaultdict(int);Ov=defaultdict(list)
for u,h in zip(sample,hyps):
    r=norm(u["text"]);rc,hc=r.replace(" ",""),h.replace(" ","")
    Ce[u["work"]]+=edist(hc,rc);Cn[u["work"]]+=len(rc)
    ov=difflib.SequenceMatcher(a=hc,b=rc,autojunk=False).ratio()
    Ov[u["work"]].append(ov)
print("== quality gate v2 (svara-stripped, short-filtered) ==")
print("%-40s %5s %6s %7s"%("work","n","CER%","overlap"))
for w in sorted(byw):
    ov=sum(Ov[w])/max(1,len(Ov[w]))
    print("%-40s %5d %5.0f%% %6.2f"%(w[:40], sum(1 for u in sample if u['work']==w), 100*Ce[w]/max(1,Cn[w]), ov))
