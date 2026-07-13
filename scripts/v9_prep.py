#!/usr/bin/env python3
"""Compose v9 data: dedup+filter human annotations, join gold, hold out 3 annotation speakers
as a de-anchored eval, write clean-train + heldout manifests."""
import json, os, collections
import soundfile as sf
ROOT = "/home/ece/BigDisk/Prathosh/ASR"
def dur(p):
    try: return round(sf.info(p).duration, 2)
    except Exception: return 0.0

# annotations: last save per id, valid = non-empty text & not unclear
last = {}
for l in open(f"{ROOT}/data/epgp/annot/annot_refs.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    last[r["id"]] = r
ann = []
for r in last.values():
    if not r.get("text", "").strip() or r.get("unclear"): continue
    p = f"{ROOT}/data/epgp/annot_clips/{r['id']}.wav"
    if os.path.exists(p):
        ann.append({"audio_filepath": p, "text": r["text"].strip(), "duration": dur(p),
                    "spk": r["id"].rsplit("_", 1)[0]})

# hold out the 3 annotation speakers with the most clips -> de-anchored, speaker-held-out eval
cnt = collections.Counter(r["spk"] for r in ann)
held_spk = [s for s, _ in cnt.most_common(5)]
held = [r for r in ann if r["spk"] in held_spk]
ann_train = [r for r in ann if r["spk"] not in held_spk]

# gold (8 eval speakers) -> all into training; join gold_refs(id->text) with manifest_gold(id->audio)
gmap = {json.loads(l)["id"]: json.loads(l) for l in open(f"{ROOT}/data/epgp/manifest_gold.jsonl")}
gold = []
for l in open(f"{ROOT}/data/epgp/gold_refs.jsonl"):
    r = json.loads(l); g = gmap.get(r["id"])
    if not g: continue
    ap = g["audio_filepath"]
    if r.get("text", "").strip() and os.path.exists(ap):
        gold.append({"audio_filepath": ap, "text": r["text"].strip(), "duration": g.get("dur", dur(ap))})

def w(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({"audio_filepath": r["audio_filepath"], "text": r["text"],
                                "duration": r["duration"], "lang": "sa"}, ensure_ascii=False) + "\n")
clean = ann_train + gold
w(f"{ROOT}/data/epgp/v9_clean.jsonl", clean)
w(f"{ROOT}/data/epgp/v9_heldout.jsonl", held)
print(f"clean-train {len(clean)} (ann {len(ann_train)} + gold {len(gold)}) | heldout {len(held)} clips from {held_spk}")
print(f"annotation speakers total {len(cnt)}")
