#!/usr/bin/env python3
"""Formalized split. TRAIN = ALL valid annotations (67 train speakers). EVAL = gold (8 e-PG
eval speakers, held out — pulled from training). chant/prose eval manifests already exist."""
import json, os
import soundfile as sf
ROOT = "/home/ece/BigDisk/Prathosh/ASR"
def dur(p):
    try: return round(sf.info(p).duration, 2)
    except Exception: return 0.0
# annotations -> ALL valid -> TRAIN (dedup last save per id, non-empty, not unclear)
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
        ann.append({"audio_filepath": p, "text": r["text"].strip(), "duration": dur(p), "lang": "sa"})
# gold -> EVAL (join gold_refs id->text with manifest_gold id->audio)
gmap = {json.loads(l)["id"]: json.loads(l) for l in open(f"{ROOT}/data/epgp/manifest_gold.jsonl")}
gold = []
for l in open(f"{ROOT}/data/epgp/gold_refs.jsonl"):
    r = json.loads(l); g = gmap.get(r["id"])
    if g and r.get("text", "").strip() and os.path.exists(g["audio_filepath"]):
        gold.append({"audio_filepath": g["audio_filepath"], "text": r["text"].strip(),
                     "duration": g.get("dur", 0), "lang": "sa"})
def w(path, rows):
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
w(f"{ROOT}/data/epgp/v9v3_clean.jsonl", ann)
w(f"{ROOT}/data/epgp/eval_gold.jsonl", gold)
tr_spk = len(set(os.path.basename(r["audio_filepath"]).rsplit("_", 1)[0] for r in ann))
print(f"TRAIN annotations {len(ann)} clips / {tr_spk} speakers | EVAL gold {len(gold)} clips")
