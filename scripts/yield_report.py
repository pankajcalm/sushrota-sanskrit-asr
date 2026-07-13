#!/usr/bin/env python3
"""Per-speaker / per-file yield report from allsegs + files_map."""
import json
from collections import defaultdict
ROOT = "/home/ece/BigDisk/Prathosh/ASR"
allsegs = [json.loads(l) for l in open(f"{ROOT}/data/utts/allsegs_bhagavatam_ex10.jsonl")]
fmap = {r["id"]: r for r in json.load(open(f"{ROOT}/data/norm16k/files_map.json"))}

TH = 0.25
raw = defaultdict(float); kept = defaultdict(float)
raw_f = defaultdict(float); kept_f = defaultdict(float); spk_of = {}
for a in allsegs:
    sp = a["speaker"]; raw[sp] += a["dur"]; raw_f[a["src"]] += a["dur"]; spk_of[a["src"]] = sp
    if a["dur"] >= 1.0 and a["cer_ref"] <= TH:
        kept[sp] += a["dur"]; kept_f[a["src"]] += a["dur"]

print(f"== PER-SPEAKER yield @ CER<={TH} (bhagavatam sk1-4) ==")
print(f"  {'speaker':14} {'kept_h':>7} {'seg_h':>7} {'yield':>6}")
for sp in sorted(raw, key=lambda s: -kept[s]):
    y = 100*kept[sp]/max(0.01, raw[sp])
    print(f"  {sp:14} {kept[sp]/3600:7.2f} {raw[sp]/3600:7.2f} {y:5.0f}%")
print(f"  {'TOTAL':14} {sum(kept.values())/3600:7.2f} {sum(raw.values())/3600:7.2f} "
      f"{100*sum(kept.values())/max(1,sum(raw.values())):5.0f}%")

print(f"\n== LOW-YIELD FILES (<35%) — candidates for review/repetition-style ==")
for f in sorted(kept_f, key=lambda x: kept_f[x]/max(1,raw_f[x])):
    y = 100*kept_f[f]/max(1, raw_f[f])
    if y < 35:
        print(f"  {y:3.0f}%  {kept_f[f]/60:5.1f}/{raw_f[f]/60:4.1f}m  {f}")
