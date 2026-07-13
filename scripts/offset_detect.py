#!/usr/bin/env python3
"""Detect per-file constant timestamp offset O (wav_time = ts_time + O). Block-0 start_ms=0
maps to real time O in the wav (leading intro). Find O by AM-matching an early block across
candidate offsets, then VALIDATE the same O on a late block (constant vs drift). Writes
data/disk_utts/offsets.json {stem: {O, anchor_ov, valid_ov, ok}}."""
import json, re, glob, os, unicodedata, difflib, tempfile
import numpy as np, soundfile as sf
import nemo.collections.asr as nemo_asr
ROOT="/home/ece/BigDisk/Prathosh/ASR"; DISK=f"{ROOT}/data/disk_asr"
MODEL=f"{ROOT}/exp/ft_ctc_v2/ft_ctc_ep20.nemo"
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s);s=DROP.sub(' ',s);s=KEEP.sub(' ',s);return re.sub(r'\s+',' ',s).strip()
def nb(s): return re.sub(r'[^a-z0-9]','',s.lower())
def ov(a,b): return difflib.SequenceMatcher(a=a.replace(' ',''),b=b.replace(' ',''),autojunk=False).ratio()
jidx={}
for f in glob.glob(f"{DISK}/**/*.json",recursive=True): jidx.setdefault(nb(os.path.basename(f)),f)
# wav->session map (hardcoded; do NOT import the extractor — its slice loop runs on import)
W2S={
 "Geeta_Bhashya_10_18":"Geetabhashya-10-18.json","GeethaBhashya_1-3":"GB-1-3-adhyaya.json",
 "GeethaBhashya_4_9":"GB-4-9 .json","Geetha_tatparya_Adhayaya_1_1":"GTN_ND_timestamps_partial_20260519_203657.json",
 "geetha_tat_A3,4_20-5-26":"GTN_ND_A3,4_20-5-26.json","VTN-Santi-hi":"VTN-santibhede sarvagamaH.json",
 "VTN_part2":"Vtn-2.json","isha_ub_18-5-26":"isha_ub_18-5-26.json","katha_ub_18-5-26":"katha_ub_18-5-26.json",
 "manduka_ub_18-5-26":"manduka_ub_18-5-26.json","rgbhashya_12-5-26":"rgbhashya_12-5-26.json",
 "rgbhashya_14-5-26":"rgbhashya_14-5-26.json","rgbhashya_19-5-26":"rgbhashya_19-5-26.json",
 "shatprashna_ub_19-5-26":"shatprashna_ub_19-5-26.json",
 "taittiriya_ub_till_brahmavalli_22-5-26":"taittiriya_ub_till_brahmavalli_22_5_26.json",
 "talavakara_ub_19-5-26":"talavakara_ub_19-5-26.json","tatvodyota":"Tatvodyota-complete.json"}
m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL,map_location="cuda"); m.eval()
m.change_decoding_strategy(decoder_type="ctc")
def tr(segs):
    with tempfile.TemporaryDirectory() as td:
        paths=[]
        for i,s in enumerate(segs):
            p=f"{td}/{i}.wav"; sf.write(p, np.clip(s,-1,1).astype('float32'), 16000); paths.append(p)
        r=m.transcribe(paths,batch_size=24,language_id="sa")
    r=r[0] if isinstance(r,tuple) else r
    return [norm(h if isinstance(h,str) else getattr(h,'text',str(h))) for h in r]
def blk_text(content,bid):
    e=content.get(bid);
    return norm(" ".join(e[0]["text"])) if e else ""
out={}
for stem,sn in W2S.items():
    sess=json.load(open(jidx[nb(sn)])); content=json.load(open(jidx[nb(sess.get("source_file",""))])).get("content",{}) if nb(sess.get("source_file","")) in jidx else {}
    audio,sr=sf.read(f"{DISK}/wavs/{stem}.wav"); audio=audio.mean(1) if audio.ndim>1 else audio; dur=len(audio)/sr
    blks=[b for b in sess["blocks"] if b.get("end_ms") and len(blk_text(content,b["id"]))>=20]
    if len(blks)<3: out[stem]={"O":0,"ok":False,"note":"few anchors"}; continue
    anc=blks[1]; lab=blk_text(content,anc["id"]); s0=anc["start_ms"]/1000; d0=(anc["end_ms"]-anc["start_ms"])/1000
    # coarse then fine offset search
    def search(grid):
        segs=[audio[int(max(0,(s0+D))*sr):int(min(dur,(s0+D+d0))*sr)] for D in grid]
        o=[ov(h,lab) for h in tr(segs)]; i=int(np.argmax(o)); return grid[i],o[i]
    Dc,_=search(list(range(0,125,5)))
    O,ao=search([Dc+x for x in range(-4,5)])
    # validate on a late block
    lb=blks[int(len(blks)*0.8)]; llab=blk_text(content,lb["id"]); ls=lb["start_ms"]/1000; ld=(lb["end_ms"]-lb["start_ms"])/1000
    vseg=[audio[int((ls+O)*sr):int((ls+O+ld)*sr)]]; vo=ov(tr(vseg)[0],llab)
    ok=bool(ao>=0.5 and vo>=0.45)
    out[stem]={"O":float(round(float(O),2)),"anchor_ov":float(round(float(ao),2)),
               "valid_ov":float(round(float(vo),2)),"ok":ok}
    print(f"  {stem[:38]:38} O={O:+5.1f}s anchor_ov={ao:.2f} valid_ov={vo:.2f} {'OK' if ok else 'CHECK'}",flush=True)
json.dump(out,open(f"{ROOT}/data/disk_utts/offsets.json","w"),indent=1)
print("saved offsets.json")
