#!/usr/bin/env python3
"""Build non-BhP alignment jobs for dvadasha-stotra + vishnu-sahasranama from files_map.
Maps each audio file to its chapter (handles Kannada numerals in Vijay filenames)."""
import os, re, json
ROOT="/home/ece/BigDisk/Prathosh/ASR"; NB=f"{ROOT}/data/texts/nonbhp"
KAN={'೦':'0','೧':'1','೨':'2','೩':'3','೪':'4','೫':'5','೬':'6','೭':'7','೮':'8','೯':'9'}
def chapter(orig):
    base=os.path.splitext(orig)[0]
    nums=re.findall(r'[0-9೦-೯]+', base)
    if not nums: return None
    return int(''.join(KAN.get(c,c) for c in nums[-1]))
rows=json.load(open(f"{ROOT}/data/norm16k/files_map.json"))
jobs=[]
for r in rows:
    if not r["ok"]: continue
    w=r["work"]
    if w=="vishnu_sahasranama":
        jobs.append(dict(id=r["id"], wavs=[r["wav"]], verses_path=f"{NB}/vishnu_sahasranama.json",
                         speaker=r["speaker"], work=w, chapter=0))
    elif w=="dvadasha_stotra":
        ch=chapter(r["orig"]); vp=f"{NB}/dvadasha_stotra_{ch}.json"
        if ch is None or not os.path.exists(vp):
            print("SKIP (no chapter/text):", r["orig"], "ch=", ch); continue
        jobs.append(dict(id=r["id"], wavs=[r["wav"]], verses_path=vp,
                         speaker=r["speaker"], work=w, chapter=ch))
json.dump(jobs, open(f"{ROOT}/data/nonbhp_jobs.json","w"), ensure_ascii=False, indent=1)
print(f"{len(jobs)} jobs written")
for j in jobs: print(f"  {j['id'][:36]:36} -> {j['work']} ch{j['chapter']}")
