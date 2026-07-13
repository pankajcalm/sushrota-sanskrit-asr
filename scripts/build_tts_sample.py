#!/usr/bin/env python3
"""Build a stratified N-hour ASR manifest from a TTS speaker's recordings.
Reusable across speakers. Emits 16k-mono wavs + NeMo manifest.
  --speaker venkat   -> from server training_master.jsonl (Devanagari text_original)
  --speaker prathosh -> from uploaded data/tts_src/prathosh_*.jsonl (SLP1 align_text -> Deva)
"""
import os, re, json, glob, random, subprocess, unicodedata, argparse
from collections import Counter
ROOT="/home/ece/BigDisk/Prathosh/ASR"
TTS="/home/ece/Prathosh/sanskrit-tts/data"
OUT=f"{ROOT}/data/tts16k"
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ]')
def norm(s):
    s=unicodedata.normalize('NFC', s); s=DROP.sub(' ', s); s=KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def load_venkat():
    out=[]
    for r in (json.loads(l) for l in open(f"{TTS}/training_master.jsonl")):
        if str(r.get("tts_exclude")).lower()=="true": continue
        out.append(dict(audio=f"{TTS}/wavs/{r['path']}", text=norm(r.get("text_original","")),
                        dur=float(r["duration_s"]), meter=(r.get("vrtta") or "unknown"),
                        cid=os.path.splitext(r["path"])[0]))
    return out

def load_prathosh():
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    out=[]
    for mf in sorted(glob.glob(f"{ROOT}/data/tts_src/prathosh_*.jsonl")):
        for r in (json.loads(l) for l in open(mf)):
            txt=r.get("align_text") or ""
            if not txt: continue
            deva=transliterate(txt, sanscript.SLP1, sanscript.DEVANAGARI)
            ap=r["audio_path"]
            if not os.path.exists(ap):                       # tolerate prathosh/ vs prathosh/wavs/
                alt=ap.replace("/prathosh/wavs/", "/prathosh/")
                ap=alt if os.path.exists(alt) else ap
            out.append(dict(audio=ap, text=norm(deva),
                            dur=float(r.get("duration") or r.get("dur") or 0),
                            meter=(r.get("meter") or "unknown"), cid=r["clip_id"]))
    return out

def stratified(clips, hours):
    target=hours*3600; rng=random.Random(0); groups={}
    for c in clips:
        if not (1.0<=c["dur"]<=20.0) or not c["text"]: continue
        groups.setdefault(c["meter"], []).append(c)
    for g in groups.values(): rng.shuffle(g)
    order=sorted(groups, key=lambda k:-len(groups[k])); idx={k:0 for k in groups}
    sel=[]; tot=0.0
    while tot<target and any(idx[k]<len(groups[k]) for k in groups):
        for k in order:                                      # round-robin -> meter diversity
            if idx[k]<len(groups[k]):
                c=groups[k][idx[k]]; idx[k]+=1; sel.append(c); tot+=c["dur"]
                if tot>=target: break
    return sel, tot

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True); ap.add_argument("--hours", type=float, default=1.0)
    a=ap.parse_args()
    clips={"venkat":load_venkat, "prathosh":load_prathosh}[a.speaker]()
    print(f"{a.speaker}: {len(clips)} usable source clips, "
          f"{sum(c['dur'] for c in clips)/3600:.2f}h available", flush=True)
    sel, tot=stratified(clips, a.hours)
    spkdir=f"{OUT}/{a.speaker}"; os.makedirs(spkdir, exist_ok=True)
    man=[]; mc=Counter(); miss=0
    for c in sel:
        p=f"{spkdir}/{c['cid']}.wav"
        subprocess.run(["ffmpeg","-y","-v","error","-i",c["audio"],"-ac","1","-ar","16000",
                        "-c:a","pcm_s16le",p], capture_output=True)
        if not os.path.exists(p): miss+=1; continue
        man.append(dict(audio_filepath=p, text=c["text"], duration=round(c["dur"],2),
                        speaker=a.speaker, work="tts", meter=c["meter"], src=c["cid"]))
        mc[c["meter"]]+=1
    mpath=f"{OUT}/manifest_{a.speaker}_{a.hours}h.jsonl"
    with open(mpath,"w") as f:
        for u in man: f.write(json.dumps(u, ensure_ascii=False)+"\n")
    print(f"KEPT {len(man)} clips, {sum(u['duration'] for u in man)/3600:.2f}h "
          f"(missing wav: {miss}) -> {mpath}", flush=True)
    print("meter mix:", dict(mc.most_common()), flush=True)

if __name__=="__main__": main()
