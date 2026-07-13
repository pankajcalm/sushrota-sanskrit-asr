#!/usr/bin/env python3
"""Smoke-test /prep (metre->pāda split) and /score (GOP + free-decode 'heard')."""
import json, os, re, unicodedata, requests
B = "http://localhost:8010"
def prep(tag, text):
    j = requests.post(B + "/prep", data={"text": text}).json()
    print("\n### PREP: %s" % tag)
    if "error" in j: print("  ERR", j["error"]); return
    for p in j["padas"]:
        print("  [v%d %-16s] %2d ak | %s" % (p["verse"], p["metre"], len(p["aksharas"]), p["text"]))

# 1. anuṣṭubh (Gītā 1.1) — expect 4 pādas of 8, metre 'anuṣṭubh'
prep("anuṣṭubh (Gītā 1.1)",
     "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।\nमामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥")
# 2a. anuṣṭubh half (word-final halanta म् must not break syllable count)
prep("anuṣṭubh half + halanta", "शुक्लांबरधरं विष्णुं शशिवर्णं चतुर्भुजम् ।")
# 2b. śārdūlavikrīḍita (19/pāda) — expect 4 pādas + metre name via signature
prep("śārdūlavikrīḍita",
     "या कुन्देन्दुतुषारहारधवला या शुभ्रवस्त्रावृता ।\nया वीणावरदण्डमण्डितकरा या श्वेतपद्मासना ॥")
# 3. IAST input, anuṣṭubh
prep("IAST anuṣṭubh",
     "vāgarthāviva saṃpṛktau vāgarthapratipattaye ।\njagataḥ pitarau vande pārvatīparameśvarau ॥")
# 4. prose -> gadya chunks on daṇḍa
prep("prose", "अत्र इति शब्दः कस्यचिद् वचनम् । यथा उक्तं तथैव भवति ।")

# 5. /score on a known-correct clip: confirm % high + 'heard' present
KEEP = re.compile(r'[^ऀ-ॿ\s]'); DROP = re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s = unicodedata.normalize('NFC', s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
last = {}
for l in open("data/epgp/annot/annot_refs.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    last[r["id"]] = r
for r in last.values():
    if not r.get("text", "").strip() or r.get("unclear"): continue
    p = "data/epgp/annot_clips/%s.wav" % r["id"]
    if os.path.exists(p) and len(norm(r["text"]).split()) >= 5:
        t = norm(r["text"])
        j = requests.post(B + "/score", files={"audio": open(p, "rb")}, data={"text": t}).json()
        print("\n### SCORE: %s  %s%%  red=%s" % (r["id"], j.get("percent"), j.get("n_red")))
        print("  REF  :", j.get("reference", "")[:60])
        print("  HEARD:", j.get("heard", "")[:60])
        break
