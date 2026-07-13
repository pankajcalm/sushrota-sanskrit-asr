#!/usr/bin/env python3
"""Build train_ft5 = existing recitation + capped TTS + capped disk data (new prose speakers).
Cap dominant disk speakers; hold out disk_vtn_tu entirely as an unseen-prose eval."""
import json, random, argparse
from collections import defaultdict
ROOT="/home/ece/BigDisk/Prathosh/ASR"
_ap=argparse.ArgumentParser()
_ap.add_argument("--gita-cap", type=float, default=3.0)
_ap.add_argument("--upa-cap", type=float, default=3.0)
_ap.add_argument("--out", default=f"{ROOT}/data/train_ft5.jsonl")
_A=_ap.parse_args()
RECIT=["manifest_fa_train_ex10","manifest_fa_nonbhp_anuv","manifest_fa_nonbhp_dv_vsn",
       "manifest_fa_nonbhp_vayu","manifest_fa_nonbhp_tantra"]
TTS=[f"{ROOT}/data/tts16k/manifest_venkat_1.0h_aug.jsonl",
     f"{ROOT}/data/tts16k/manifest_prathosh_1.0h_aug.jsonl"]
DISK=f"{ROOT}/data/disk_utts/manifest_disk.jsonl"
CAP={"disk_gita_rig":_A.gita_cap, "disk_upanishad":_A.upa_cap, "disk_gtn":99.0, "disk_vtn_tu":0.0}  # 0 => held out (eval)
rng=random.Random(0)
train=[]; bysrc=defaultdict(float)
def add(r, src):
    train.append({"audio_filepath":r["audio_filepath"],"text":r["text"],"duration":r["duration"],"lang":"sa"})
    bysrc[src]+=r["duration"]
for name in RECIT:
    for l in open(f"{ROOT}/data/utts_fa/{name}.jsonl"): add(json.loads(l), "recit")
for mf in TTS:
    tag="tts_"+("venkat" if "venkat" in mf else "prathosh")
    for l in open(mf): add(json.loads(l), tag)
disk=[json.loads(l) for l in open(DISK)]
byspk=defaultdict(list)
for r in disk: byspk[r["speaker"]].append(r)
prose_eval=[]
for spk, rows in byspk.items():
    cap=CAP.get(spk, 99.0)
    if cap==0.0:
        prose_eval+=rows; continue
    byw=defaultdict(list)
    for r in rows: byw[r["work"]].append(r)
    for w in byw.values(): rng.shuffle(w)
    order=sorted(byw, key=lambda k:-len(byw[k])); idx={k:0 for k in byw}; tot=0.0
    while tot<cap*3600 and any(idx[k]<len(byw[k]) for k in byw):
        for k in order:
            if idx[k]<len(byw[k]):
                r=byw[k][idx[k]]; idx[k]+=1; add(r, spk); tot+=r["duration"]
                if tot>=cap*3600: break
rng.shuffle(train)
with open(_A.out,"w") as f:
    for r in train: f.write(json.dumps(r,ensure_ascii=False)+"\n")
with open(f"{ROOT}/data/utts_fa/manifest_prose_eval_vtn.jsonl","w") as f:
    for r in prose_eval: f.write(json.dumps(dict(audio_filepath=r["audio_filepath"],text=r["text"],
        duration=r["duration"],speaker=r["speaker"]),ensure_ascii=False)+"\n")
tot=sum(r["duration"] for r in train)/3600
print("== train_ft5 composition ==")
for s,d in sorted(bysrc.items(),key=lambda x:-x[1]): print("  %-16s %5.2f h (%.0f%%)"%(s,d/3600,100*d/3600/tot))
print("  %-16s %5.2f h  (%d utts)"%("TOTAL",tot,len(train)))
print("held-out prose eval (vtn_tu): %.2f h, %d utts -> manifest_prose_eval_vtn.jsonl"%(
      sum(r["duration"] for r in prose_eval)/3600, len(prose_eval)))
