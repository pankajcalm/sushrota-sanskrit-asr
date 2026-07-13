import json
P=[json.loads(l) for l in open("data/epgp/annot/manifest_hardneg.jsonl")]
P.sort(key=lambda x:-x["dis"])
print("=== HARDEST picked (dis~0.9) ===")
for r in P[:6]:
    print(f"[dis {r['dis']}] v5: {r['draft'][:66]}")
    print(f"           wh: {r['wh'][:66]}")
print("\n=== MID picked (dis~0.55-0.62) ===")
for r in [x for x in P if 0.55<=x['dis']<=0.62][:3]:
    print(f"[dis {r['dis']}] v5: {r['draft'][:66]}")
    print(f"           wh: {r['wh'][:66]}")
S=[json.loads(l) for l in open("data/epgp/annot/shortlist_scored.jsonl")]
exc=[r for r in S if r['dis']>0.9]
print(f"\n=== EXCLUDED dis>0.9 ({len(exc)} of {len(S)} shortlist) — expect English/noise ===")
for r in sorted(exc,key=lambda x:-x['dis'])[:6]:
    print(f"[dis {round(r['dis'],2)}] v5: {r['draft'][:62]}")
print(f"\nspeakers covered: {len(set(r['video'] for r in P))}  | avg picked/spk: {len(P)/len(set(r['video'] for r in P)):.1f}")
