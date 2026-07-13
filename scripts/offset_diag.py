#!/usr/bin/env python3
"""Diagnose whether a mis-aligned file has a CONSTANT timestamp offset. For several blocks,
slice at start+delta and find the delta maximizing overlap with the label. Consistent best
delta across blocks => constant offset (easy fix); scattered => drift (forced-align instead)."""
import json, re, glob, os, unicodedata, difflib
import numpy as np, soundfile as sf
import nemo.collections.asr as nemo_asr
ROOT="/home/ece/BigDisk/Prathosh/ASR"; DISK=f"{ROOT}/data/disk_asr"
MODEL=f"{ROOT}/exp/ft_ctc_v2/ft_ctc_ep20.nemo"
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s);s=DROP.sub(' ',s);s=KEEP.sub(' ',s);return re.sub(r'\s+',' ',s).strip()
def nb(s): return re.sub(r'[^a-z0-9]','',s.lower())
jidx={}
for f in glob.glob(f"{DISK}/**/*.json",recursive=True): jidx.setdefault(nb(os.path.basename(f)),f)
JOBS={"Geeta_Bhashya_10_18":"Geetabhashya-10-18.json","katha_ub_18-5-26":"katha_ub_18-5-26.json",
      "VTN_part2":"Vtn-2.json"}
m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL,map_location="cuda"); m.eval()
m.change_decoding_strategy(decoder_type="ctc")
DELTAS=[-8,-6,-4,-3,-2,-1,0,1,2,3,4]
for stem,sn in JOBS.items():
    sess=json.load(open(jidx[nb(sn)])); src=sess.get("source_file","")
    content=json.load(open(jidx[nb(src)])).get("content",{}) if nb(src) in jidx else {}
    audio,sr=sf.read(f"{DISK}/wavs/{stem}.wav")
    if audio.ndim>1: audio=audio.mean(1)
    blks=[b for b in sess["blocks"] if b.get("end_ms") and content.get(b["id"])]
    picks=blks[len(blks)//4::max(1,len(blks)//5)][:4]   # 4 spread-out blocks
    print(f"\n=== {stem} (probe {len(picks)} blocks) ===")
    for b in picks:
        lab=norm(" ".join(content[b["id"]][0]["text"])); s0=b["start_ms"]/1000; e0=b["end_ms"]/1000
        segs=[];
        for dl in DELTAS:
            s=max(0,s0+dl); e=min(len(audio)/sr,e0+dl)
            segs.append(audio[int(s*sr):int(e*sr)])
        res=m.transcribe(segs,batch_size=16,language_id="sa")
        if isinstance(res,tuple):res=res[0]
        ov=[difflib.SequenceMatcher(a=norm(h if isinstance(h,str) else getattr(h,'text',str(h))).replace(' ',''),b=lab.replace(' ',''),autojunk=False).ratio() for h in res]
        best=DELTAS[int(np.argmax(ov))]
        print(f"  blk@{s0:6.1f}s bestΔ={best:+d}s ov={max(ov):.2f}  (Δ0 ov={ov[DELTAS.index(0)]:.2f})  overlaps="+ " ".join(f"{d:+d}:{o:.2f}" for d,o in zip(DELTAS,ov)))
