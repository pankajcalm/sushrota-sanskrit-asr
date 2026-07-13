#!/usr/bin/env python3
import json, argparse
from collections import defaultdict
ROOT="/home/ece/BigDisk/Prathosh/ASR"
ap=argparse.ArgumentParser()
ap.add_argument("--venkat", default=f"{ROOT}/data/tts16k/manifest_venkat_1.0h_aug.jsonl")
ap.add_argument("--prathosh", default=f"{ROOT}/data/tts16k/manifest_prathosh_1.0h_aug.jsonl")
ap.add_argument("--out", default=f"{ROOT}/data/train_manifest_v1.jsonl")
A=ap.parse_args()
srcs=[("recit_bhagavatam", f"{ROOT}/data/utts_fa/manifest_fa_train_ex10.jsonl"),
      ("recit_anuvyakhyana", f"{ROOT}/data/utts_fa/manifest_fa_nonbhp_anuv.jsonl"),
      ("recit_dvadasha_vsn", f"{ROOT}/data/utts_fa/manifest_fa_nonbhp_dv_vsn.jsonl"),
      ("recit_vayustuti", f"{ROOT}/data/utts_fa/manifest_fa_nonbhp_vayu.jsonl"),
      ("recit_tantrasara", f"{ROOT}/data/utts_fa/manifest_fa_nonbhp_tantra.jsonl"),
      ("tts_venkat", A.venkat),
      ("tts_prathosh", A.prathosh)]
out=open(A.out,"w")
by=defaultdict(float); n=defaultdict(int); spk=defaultdict(float); tot=0.0
for tag,mf in srcs:
    for l in open(mf):
        r=json.loads(l)
        rec={"audio_filepath":r["audio_filepath"],"text":r["text"],
             "duration":r["duration"],"speaker":r.get("speaker"),"source":tag}
        out.write(json.dumps(rec,ensure_ascii=False)+"\n")
        by[tag]+=r["duration"]; n[tag]+=1; tot+=r["duration"]; spk[r.get("speaker")]+=r["duration"]
out.close()
print("== TRAIN MANIFEST v1: data/train_manifest_v1.jsonl ==")
for tag,_ in srcs:
    print("  %-14s %5d utts  %5.2f h  (%.0f%%)" % (tag, n[tag], by[tag]/3600, 100*by[tag]/tot))
print("  %-14s %5d utts  %5.2f h" % ("TOTAL", sum(n.values()), tot/3600))
top=sorted(spk.items(), key=lambda x:-x[1])[:5]
print("top speakers:", ", ".join("%s %.2fh(%.0f%%)"%(s,d/3600,100*d/tot) for s,d in top))
print("held-out eval: data/utts_fa/manifest_fa_eval_sk10.jsonl (2.74h, disjoint speakers+text)")
