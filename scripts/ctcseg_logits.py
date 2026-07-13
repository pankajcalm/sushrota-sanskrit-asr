#!/usr/bin/env python3
"""Stage 1 (nemo_ai4b): windowed CTC log-probs + tokenized canonical verses.
Emits data/ctcseg/<id>.npy  ([T,257] logprobs, col0=blank, 1..256=sa tokens)
      data/ctcseg/<id>.json {index_duration, wav, dur, utterances:[text], tokens:[[ids]]}
"""
import os, re, json, glob, unicodedata, argparse
import numpy as np, torch, soundfile as sf
import nemo.collections.asr as nemo_asr

ROOT="/home/ece/BigDisk/Prathosh/ASR"
MAP=f"{ROOT}/data/norm16k/files_map.json"; TEXTS=f"{ROOT}/data/texts/bhagavatam"
OUT=f"{ROOT}/data/ctcseg"; os.makedirs(OUT, exist_ok=True)
MODEL=("/home/ece/.cache/torch/NeMo/NeMo_2.7.3/hf_hub_cache/ai4bharat/"
       "indicconformer_stt_sa_hybrid_ctc_rnnt_large/c82246cc7136e7f1be8df3090e3a07d9/"
       "indicconformer_stt_sa_hybrid_rnnt_large.nemo")
OFF, V, BLANK = 4096, 256, 5632
COLS = [BLANK] + list(range(OFF, OFF+V))          # blank first (ctc_segmentation default)
W, OV = 28.0, 0.0                                  # non-overlapping windows (drift-free stitch)

KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ]')
def norm(s):
    s=unicodedata.normalize('NFC', s); s=DROP.sub(' ', s); s=KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
def adh_set(spec):
    s=str(spec)
    if '-' in s:
        a,b=s.split('-')[0], s.split('-')[-1]
        if a.isdigit() and b.isdigit(): return set(range(int(a), int(b)+1))
    return {int(x) for x in s.split('-') if x.isdigit()}
def canonical_verses(sk, adh, vrange=None):
    p=f"{TEXTS}/skandha_{sk}.json"
    if not os.path.exists(p): return []
    d=json.load(open(p)); want=adh_set(adh); items=[]
    for v in d["content"].values():
        for e in v:
            if e.get("adhyaya") in want:
                vn=e["verse"] if isinstance(e.get("verse"), int) else 10**6
                if vrange and not (vrange[0] <= vn <= vrange[1]): continue
                t=norm(" ".join(e.get("text", [])))
                if t: items.append((e["adhyaya"], vn, t))
    items.sort(key=lambda x:(x[0], x[1]))
    return [t for _,_,t in items]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ids"); ap.add_argument("--work", default="bhagavatam")
    ap.add_argument("--only-sk", type=int); ap.add_argument("--exclude-sk", type=int)
    ap.add_argument("--model", default=MODEL)
    a=ap.parse_args()
    rows=[r for r in json.load(open(MAP)) if r["ok"] and r["work"]==a.work]
    if a.only_sk is not None: rows=[r for r in rows if r["skandha"]==a.only_sk]
    if a.exclude_sk is not None: rows=[r for r in rows if r["skandha"]!=a.exclude_sk]
    if a.ids: rows=[r for r in rows if r["id"].startswith(tuple(a.ids.split(",")))]
    print(f"stage1: {len(rows)} files", flush=True)

    m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(a.model, map_location="cuda"); m.eval()
    def logprobs(clip):
        sig=torch.tensor(clip).unsqueeze(0).cuda(); sl=torch.tensor([clip.shape[0]]).cuda()
        with torch.no_grad():
            enc,_=m.forward(input_signal=sig, input_signal_length=sl)
            lp=m.ctc_decoder(encoder_output=enc)[0].cpu().numpy()      # [T,5633]
        x=lp[:, COLS]; x=x-x.max(-1, keepdims=True); e=np.exp(x)
        return np.log(e/e.sum(-1, keepdims=True)+1e-12).astype(np.float32)

    for r in rows:
        mm=re.search(r'shloka[_ ]*(\d+)[-_ ]+(\d+)', r["orig"].lower())
        vrange=(int(mm.group(1)), int(mm.group(2))) if mm else None
        verses=canonical_verses(r["skandha"], r["adhyaya"], vrange)
        if not verses: print("  NO_REF", r["id"], flush=True); continue
        audio, sr=sf.read(r["wav"]); dur=len(audio)/sr
        step=W-OV
        wins=[]; s=0.0
        while s < dur - 1e-3:
            wins.append([s, min(dur, s+W)]); s+=step
        if len(wins) >= 2 and (wins[-1][1]-wins[-1][0]) < 4.0:   # absorb tiny tail
            wins[-2][1]=wins[-1][1]; wins.pop()
        parts=[]; idur=None
        for wi,(ws,we) in enumerate(wins):
            clip=audio[int(ws*sr):int(we*sr)].astype(np.float32)
            lp=logprobs(clip); tw=lp.shape[0]; idur=(we-ws)/tw
            fps=1.0/idur
            lo=0 if wi==0 else int((OV/2)*fps)
            hi=tw if wi==len(wins)-1 else tw-int((OV/2)*fps)
            parts.append(lp[lo:hi])
        full=np.concatenate(parts, 0)
        idur=dur/full.shape[0]             # exact uniform frame spacing (no drift)
        tokens=[[i+1 for i in m.tokenizer.text_to_ids(t, "sa")] for t in verses]  # +1: col0=blank
        np.save(f"{OUT}/{r['id']}.npy", full)
        json.dump(dict(index_duration=float(idur), wav=r["wav"], dur=dur,
                       speaker=r["speaker"], skandha=r["skandha"], adhyaya=r["adhyaya"],
                       work=r["work"], utterances=verses, tokens=tokens),
                  open(f"{OUT}/{r['id']}.json", "w"), ensure_ascii=False)
        print(f"  {r['id'][:44]:44} T={full.shape[0]:5d} verses={len(verses):3d} {dur/60:5.1f}m", flush=True)
    # save sa char_list once
    sv=json.load(open(f"{ROOT}/lm/sa_vocab.json"))
    json.dump(["<blank>"]+sv["tokens"], open(f"{OUT}/char_list.json", "w"), ensure_ascii=False)
    print("stage1 done", flush=True)

if __name__=="__main__": main()
