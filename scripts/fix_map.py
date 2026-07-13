#!/usr/bin/env python3
"""Patch files_map.json: correct mis-parsed skandha/adhyaya + work labels.
Curated by eyeballing all 94 filenames (more reliable than regex here)."""
import json
P = "/home/ece/BigDisk/Prathosh/ASR/data/norm16k/files_map.json"
rows = json.load(open(P))

# orig filename -> (skandha, adhyaya)  for the mis-parsed / unparsed bhagavatam files
FIX = {
    "Giri_4th Skandha 1st Adhyaya.mp3": (4, "1"),
    "Giri_4th Skandha 2nd Adhyaya.mp3": (4, "2"),
    "Giri_4th Skandha 3rd Adhyaya.mp3": (4, "3"),
    "Giri_4th Skandha 4th Adhyaya.mp3": (4, "4"),
    "Akhil_skandah_3.adhayaya_17.Shloka_1-10.mp3":  (3, "17"),
    "Akhil_skandah_3.adhayaya_17.Shloka_11-18.mp3": (3, "17"),
    "Akhil_skandah_3.adhayaya_17.Shloka_19-37.mp3": (3, "17"),
    "Akhil_skandah_3.adhayaya_18.mp3": (3, "18"),
    "Akhil_skandah_3.adhayaya_19.mp3": (3, "19"),
    "Akhil_skandah_3.adhayaya_20.mp3": (3, "20"),
    "Vinay_3 skndha 1 adhyaya.m4a": (3, "1"),
    "Vinay_3 skndha 2 adhyaya.m4a": (3, "2"),
    "Vinay_3skandha 3&4 adhyaya.m4a": (3, "3-4"),
    "Sanjeevachar_Bhagavatam 10-1,2,3,4.mp3": (10, "1-4"),
}
WORK_FIX = {"Venu_vishnu_sahahasra_nama.mp3": "vishnu_sahasranama"}

n = 0
for r in rows:
    if r["orig"] in FIX:
        r["skandha"], r["adhyaya"] = FIX[r["orig"]]; n += 1
    if r["orig"] in WORK_FIX:
        r["work"] = WORK_FIX[r["orig"]]; n += 1

json.dump(rows, open(P, "w"), ensure_ascii=False, indent=2)

# report residual gaps
bad = [r for r in rows if r["work"] == "bhagavatam" and (r["skandha"] is None or r["adhyaya"] is None)]
print(f"applied {n} fixes. residual bhagavatam gaps: {len(bad)}")
for r in bad: print("  GAP", r["orig"])

from collections import defaultdict
cov = defaultdict(lambda: [0, 0.0])
for r in rows:
    if r["work"] == "bhagavatam":
        cov[r["skandha"]][0] += 1; cov[r["skandha"]][1] += r["duration_s"]
print("== bhagavatam coverage by skandha ==")
for sk in sorted(cov, key=lambda x: (x is None, x)):
    print(f"  skandha {sk}: {cov[sk][0]} files  {cov[sk][1]/3600:.2f} h")
