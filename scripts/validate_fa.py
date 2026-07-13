#!/usr/bin/env python3
"""Validate forced-aligned dataset: transcribe a sample of cut clips, CER vs label."""
import json, re, unicodedata
import nemo.collections.asr as nemo_asr
ROOT="/home/ece/BigDisk/Prathosh/ASR"
MODEL=("/home/ece/.cache/torch/NeMo/NeMo_2.7.3/hf_hub_cache/ai4bharat/"
       "indicconformer_stt_sa_hybrid_ctc_rnnt_large/c82246cc7136e7f1be8df3090e3a07d9/"
       "indicconformer_stt_sa_hybrid_rnnt_large.nemo")
import sys
MANIFEST=sys.argv[1] if len(sys.argv)>1 else f"{ROOT}/data/utts_fa/manifest_fa_train_ex10.jsonl"
man=[json.loads(l) for l in open(MANIFEST)]
# deterministic spread across speakers: every Nth
sample=man[::max(1,len(man)//40)][:40]
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
def cer(a,b):
    a,b=a.replace(" ",""),b.replace(" ","")
    if not b: return 1.0
    n,m=len(a),len(b); prev=list(range(m+1))
    for i in range(1,n+1):
        cur=[i]+[0]*m
        for j in range(1,m+1): cur[j]=min(prev[j]+1,cur[j-1]+1,prev[j-1]+(a[i-1]!=b[j-1]))
        prev=cur
    return prev[m]/max(1,m)
m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL,map_location="cuda"); m.eval()
m.change_decoding_strategy(decoder_type="ctc")
paths=[u["audio_filepath"] for u in sample]
res=m.transcribe(paths,batch_size=16,language_id="sa")
if isinstance(res,tuple): res=res[0]
hyps=[norm(h if isinstance(h,str) else getattr(h,"text",str(h))) for h in res]
cers=[]
for u,h in zip(sample,hyps):
    c=cer(h,norm(u["text"])); cers.append(c)
print(f"sampled {len(sample)} clips across speakers")
for u,h,c in sorted(zip(sample,hyps,cers),key=lambda x:-x[2])[:8]:
    print(f"  CER {c*100:5.1f}%  {u['src'][:30]}\n     HYP {h[:64]}\n     LAB {norm(u['text'])[:64]}")
import statistics as st
print(f"\nraw     : mean CER {100*st.mean(cers):.2f}%  median {100*st.median(cers):.2f}%  "
      f"<=15%: {sum(c<=.15 for c in cers)}/{len(cers)}")
# anusvara-folded CER (collapse orthographic ं <-> म् which are phonetically identical)
pn=lambda s: norm(s).replace('ं','म्')
pcers=[cer(pn(h),pn(u["text"])) for u,h in zip(sample,hyps)]
print(f"anusvara: mean CER {100*st.mean(pcers):.2f}%  median {100*st.median(pcers):.2f}%  "
      f"<=15%: {sum(c<=.15 for c in pcers)}/{len(pcers)}")
