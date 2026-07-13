#!/usr/bin/env python3
"""Stage 2 (dec env, has ctc-segmentation): forced-align canonical verses to audio.
Reads data/ctcseg/<id>.{npy,json} -> per-verse (start,end,confidence) -> cut wavs
-> data/utts_fa/<id>/verse_*.wav + manifest_fa_<tag>.jsonl + yield table.
"""
import os, re, json, glob, argparse, subprocess
import numpy as np
from ctc_segmentation import (CtcSegmentationParameters, ctc_segmentation,
                              determine_utterance_segments, prepare_token_list)

ROOT="/home/ece/BigDisk/Prathosh/ASR"
SEG=f"{ROOT}/data/ctcseg"; OUT=f"{ROOT}/data/utts_fa"; os.makedirs(OUT, exist_ok=True)
CHAR_LIST=json.load(open(f"{SEG}/char_list.json"))

def align_file(jid, min_score, min_dur, max_dur):
    meta=json.load(open(f"{SEG}/{jid}.json"))
    lp=np.load(f"{SEG}/{jid}.npy")
    utts=meta["utterances"]; toks=[np.array(t, dtype=int) for t in meta["tokens"]]
    cfg=CtcSegmentationParameters(); cfg.char_list=CHAR_LIST
    cfg.index_duration=meta["index_duration"]; cfg.blank=0
    gt, utt_begin=prepare_token_list(cfg, toks)
    timings, char_probs, _=ctc_segmentation(cfg, lp, gt)
    segs=determine_utterance_segments(cfg, utt_begin, char_probs, timings, utts)  # [(s,e,score)]
    segdir=f"{OUT}/{jid}"; os.makedirs(segdir, exist_ok=True)
    kept=[]; allrec=[]
    for i,(s,e,score) in enumerate(segs):
        d=e-s; allrec.append(dict(src=jid, verse=i, dur=round(d,2), score=round(float(score),3)))
        if score>=min_score and min_dur<=d<=max_dur and utts[i]:
            p=f"{segdir}/verse_{i:04d}.wav"
            subprocess.run(["ffmpeg","-y","-v","error","-i",meta["wav"],"-ss",f"{s:.3f}",
                            "-to",f"{e:.3f}","-ar","16000","-ac","1","-c:a","pcm_s16le",p],
                           capture_output=True)
            kept.append(dict(audio_filepath=p, text=utts[i], duration=round(d,2),
                             speaker=meta["speaker"], skandha=meta["skandha"],
                             adhyaya=meta["adhyaya"], work=meta["work"], src=jid,
                             score=round(float(score),3)))
    return kept, allrec, meta["dur"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ids"); ap.add_argument("--tag", default="run")
    ap.add_argument("--min-score", type=float, default=-2.0)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--only-sk", type=int)
    a=ap.parse_args()
    jids=sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{SEG}/*.json")
                if os.path.basename(f)!="char_list.json")
    if a.ids: jids=[j for j in jids if j.startswith(tuple(a.ids.split(",")))]
    if a.only_sk is not None:
        jids=[j for j in jids if json.load(open(f"{SEG}/{j}.json")).get("skandha")==a.only_sk]
    manifest=[]; allsegs=[]
    for jid in jids:
        try:
            kept, allrec, dur=align_file(jid, a.min_score, a.min_dur, a.max_dur)
        except Exception as ex:
            print(f"  ERR {jid}: {ex}", flush=True); continue
        manifest+=kept; allsegs+=allrec
        kd=sum(u["duration"] for u in kept)
        print(f"  {jid[:44]:44} verses={len(allrec):3d} kept={len(kept):3d} "
              f"{kd/60:5.1f}m  {100*kd/max(1,dur):3.0f}% of {dur/60:.1f}m", flush=True)
    mpath=f"{OUT}/manifest_fa_{a.tag}.jsonl"
    with open(mpath,"w") as f:
        for u in manifest: f.write(json.dumps(u, ensure_ascii=False)+"\n")
    with open(f"{OUT}/allsegs_fa_{a.tag}.jsonl","w") as f:
        for u in allsegs: f.write(json.dumps(u, ensure_ascii=False)+"\n")
    print("\n== yield vs min-score ==", flush=True)
    for th in (-0.5,-1.0,-1.5,-2.0,-3.0):
        h=sum(x["dur"] for x in allsegs if x["score"]>=th and a.min_dur<=x["dur"]<=a.max_dur)/3600
        print(f"  score>={th:>5}:  {h:.2f} h", flush=True)
    print(f"\nKEPT(@{a.min_score}) {len(manifest)} utts, "
          f"{sum(u['duration'] for u in manifest)/3600:.2f} h -> {mpath}", flush=True)

if __name__=="__main__": main()
