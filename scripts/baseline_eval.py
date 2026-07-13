#!/usr/bin/env python3
"""Zero-shot baseline: IndicConformer CTC (greedy) on the sk10 held-out eval.
Content-only CER/WER (dandas ।॥, digits, avagraha ऽ, om ॐ, non-Devanagari all stripped).
Micro-averaged (corpus CER = total edit dist / total ref chars) + per-speaker."""
import json, re, unicodedata, sys, argparse
from collections import defaultdict
import nemo.collections.asr as nemo_asr

ROOT="/home/ece/BigDisk/Prathosh/ASR"
MANIFEST=f"{ROOT}/data/utts_fa/manifest_fa_eval_sk10.jsonl"
BASE=("/home/ece/.cache/torch/NeMo/NeMo_2.7.3/hf_hub_cache/ai4bharat/"
      "indicconformer_stt_sa_hybrid_ctc_rnnt_large/c82246cc7136e7f1be8df3090e3a07d9/"
      "indicconformer_stt_sa_hybrid_rnnt_large.nemo")
_ap=argparse.ArgumentParser(); _ap.add_argument("--model", default=BASE); _ap.add_argument("--tag", default="base")
_ap.add_argument("--manifest", default=MANIFEST)
_A=_ap.parse_args()
MODEL=_A.model; MANIFEST=_A.manifest
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ]')
def norm(s):
    s=unicodedata.normalize('NFC', s); s=DROP.sub(' ', s); s=KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
def edist(a, b):
    n, m=len(a), len(b)
    if n==0: return m
    if m==0: return n
    prev=list(range(m+1))
    for i in range(1, n+1):
        cur=[i]+[0]*m; ai=a[i-1]
        for j in range(1, m+1): cur[j]=min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ai!=b[j-1]))
        prev=cur
    return prev[m]

man=[json.loads(l) for l in open(MANIFEST)]
print(f"eval utts: {len(man)}", flush=True)
m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL, map_location="cuda"); m.eval()
m.change_decoding_strategy(decoder_type="ctc")
paths=[u["audio_filepath"] for u in man]
res=m.transcribe(paths, batch_size=32, language_id="sa")
if isinstance(res, tuple): res=res[0]
hyps=[norm(h if isinstance(h,str) else getattr(h,"text",str(h))) for h in res]

# accumulate micro CER/WER overall + per speaker
Ce=defaultdict(int); Cn=defaultdict(int); We=defaultdict(int); Wn=defaultdict(int); U=defaultdict(int)
worst=[]
for u, h in zip(man, hyps):
    sp=u["speaker"]; ref=norm(u["text"])
    rc, hc=ref.replace(" ",""), h.replace(" ","")
    ce=edist(hc, rc); we=edist(h.split(), ref.split())
    Ce[sp]+=ce; Cn[sp]+=len(rc); We[sp]+=we; Wn[sp]+=len(ref.split()); U[sp]+=1
    Ce["_ALL"]+=ce; Cn["_ALL"]+=len(rc); We["_ALL"]+=we; Wn["_ALL"]+=len(ref.split())
    if len(rc)>15: worst.append((ce/max(1,len(rc)), sp, h, ref))

def pct(e,n): return 100*e/max(1,n)
print("\n== PER-SPEAKER (content-only, micro CER/WER) ==")
print(f"  {'speaker':16} {'utts':>4} {'CER%':>7} {'WER%':>7}")
for sp in sorted([k for k in U], key=lambda s: pct(Ce[s],Cn[s])):
    print(f"  {sp:16} {U[sp]:4d} {pct(Ce[sp],Cn[sp]):7.2f} {pct(We[sp],Wn[sp]):7.2f}")
print(f"\n== OVERALL (micro) ==  CER {pct(Ce['_ALL'],Cn['_ALL']):.2f}%  "
      f"WER {pct(We['_ALL'],Wn['_ALL']):.2f}%   ({len(man)} utts)")
print("\n== worst 6 clips ==")
for c,sp,h,r in sorted(worst, reverse=True)[:6]:
    print(f"  {c*100:5.1f}% [{sp}]\n    HYP {h[:70]}\n    REF {r[:70]}")
json.dump({"overall_cer":pct(Ce['_ALL'],Cn['_ALL']), "overall_wer":pct(We['_ALL'],Wn['_ALL']),
           "per_speaker":{sp:{"utts":U[sp],"cer":pct(Ce[sp],Cn[sp]),"wer":pct(We[sp],Wn[sp])} for sp in U}},
          open(f"{ROOT}/data/eval_sk10_{_A.tag}.json","w"), ensure_ascii=False, indent=1)
print(f"\n[{_A.tag}] model={MODEL.split('/')[-1]}  saved -> data/eval_sk10_{_A.tag}.json", flush=True)
