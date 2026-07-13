#!/usr/bin/env python3
"""Slice the timestamped disk recordings into (wav-segment, text) pairs — no forced align.
Per wav: session blocks {id,start_ms,end_ms} + source-text content[id].text -> cut + label."""
import os, re, json, glob, unicodedata
import numpy as np, soundfile as sf
ROOT="/home/ece/BigDisk/Prathosh/ASR"; DISK=f"{ROOT}/data/disk_asr"
WAVDIR=f"{DISK}/wavs"; OUT=f"{ROOT}/data/disk_utts"; os.makedirs(OUT, exist_ok=True)

# wav stem -> session file basename (verified)
WAV2SESSION={
 "Geeta_Bhashya_10_18":"Geetabhashya-10-18.json",
 "GeethaBhashya_1-3":"GB-1-3-adhyaya.json",
 "GeethaBhashya_4_9":"GB-4-9 .json",
 "Geetha_tatparya_Adhayaya_1_1":"GTN_ND_timestamps_partial_20260519_203657.json",
 "geetha_tat_A3,4_20-5-26":"GTN_ND_A3,4_20-5-26.json",
 "VTN-Santi-hi":"VTN-santibhede sarvagamaH.json",
 "VTN_part2":"Vtn-2.json",
 "isha_ub_18-5-26":"isha_ub_18-5-26.json",
 "katha_ub_18-5-26":"katha_ub_18-5-26.json",
 "manduka_ub_18-5-26":"manduka_ub_18-5-26.json",
 "rgbhashya_12-5-26":"rgbhashya_12-5-26.json",
 "rgbhashya_14-5-26":"rgbhashya_14-5-26.json",
 "rgbhashya_19-5-26":"rgbhashya_19-5-26.json",
 "shatprashna_ub_19-5-26":"shatprashna_ub_19-5-26.json",
 "taittiriya_ub_till_brahmavalli_22-5-26":"taittiriya_ub_till_brahmavalli_22_5_26.json",
 "talavakara_ub_19-5-26":"talavakara_ub_19-5-26.json",
 "tatvodyota":"Tatvodyota-complete.json",
}
def spk(stem):
    if stem.startswith(("Geeta_Bhashya","GeethaBhashya","rgbhashya")): return "disk_gita_rig"
    if stem.startswith(("VTN","tatvodyota")): return "disk_vtn_tu"
    if stem.startswith(("Geetha_tatparya","geetha_tat")): return "disk_gtn"
    return "disk_upanishad"

KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ]')
def norm(s):
    s=unicodedata.normalize('NFC', s); s=DROP.sub(' ', s); s=KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
def nb(s): return re.sub(r'[^a-z0-9]','',s.lower())

# index every json on disk by normalized basename
jidx={}
for f in glob.glob(f"{DISK}/**/*.json", recursive=True): jidx.setdefault(nb(os.path.basename(f)), f)

def resolve(name):
    return jidx.get(nb(name)) or jidx.get(nb(os.path.basename(name)))

def blk_text(entry):
    out=[]
    for e in (entry if isinstance(entry,list) else [entry]):
        if isinstance(e,dict) and e.get("text"): out.append(" ".join(e["text"]))
    return norm(" ".join(out))

from collections import defaultdict
OFF=json.load(open(f"{OUT}/offsets.json")) if os.path.exists(f"{OUT}/offsets.json") else {}
manifest=[]; byspk=defaultdict(float); rep=[]
for stem, sess_name in WAV2SESSION.items():
    O=OFF.get(stem,{}).get("O",0.0)
    wav=f"{WAVDIR}/{stem}.wav"
    if not os.path.exists(wav): rep.append((stem,"NO_WAV",0,0)); continue
    sf_path=resolve(sess_name)
    if not sf_path: rep.append((stem,"NO_SESSION",0,0)); continue
    sess=json.load(open(sf_path)); src=sess.get("source_file","")
    tf=resolve(src); content=json.load(open(tf)).get("content",{}) if tf else {}
    audio,sr=sf.read(wav)
    if audio.ndim>1: audio=audio.mean(1)
    segdir=f"{OUT}/{spk(stem)}"; os.makedirs(segdir, exist_ok=True)
    kept=0; kdur=0.0
    for i,blk in enumerate(sess["blocks"]):
        if not isinstance(blk,dict) or blk.get("end_ms") is None: continue
        text=blk_text(content.get(blk["id"]))
        s=blk["start_ms"]/1000.0+O; e=blk["end_ms"]/1000.0+O; d=e-s
        if not text or d<0.5 or d>30: continue
        s=max(0.0,s); e=min(len(audio)/sr,e)
        p=f"{segdir}/{stem}_{i:04d}.wav"
        sf.write(p, audio[int(s*sr):int(e*sr)], sr)
        manifest.append(dict(audio_filepath=p, text=text, duration=round(d,2),
                             speaker=spk(stem), work=stem, src=f"{stem}_{i}"))
        kept+=1; kdur+=d
    byspk[spk(stem)]+=kdur; rep.append((stem, os.path.basename(sf_path), kept, kdur/60))
with open(f"{OUT}/manifest_disk.jsonl","w") as f:
    for u in manifest: f.write(json.dumps(u,ensure_ascii=False)+"\n")
print("%-38s %-26s %5s %8s"%("wav","session","kept","min"))
for stem,s,k,m in rep: print("%-38s %-26s %5s %7.1fm"%(stem[:38], str(s)[:26], k, m))
print("\n== per-speaker-group ==")
for s,d in byspk.items(): print(f"  {s:18} {d/3600:.2f} h")
print(f"TOTAL: {sum(u['duration'] for u in manifest)/3600:.2f} h, {len(manifest)} utts -> {OUT}/manifest_disk.jsonl")
