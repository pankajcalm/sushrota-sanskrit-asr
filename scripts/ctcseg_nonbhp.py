#!/usr/bin/env python3
"""Stage 1 for non-Bhagavatam works (config-driven). Concatenates a job's wavs,
computes windowed CTC logprobs, tokenizes canonical verses, writes data/ctcseg/<id>.{npy,json}
in the SAME format as ctcseg_logits so ctcseg_align.py processes them unchanged.

--jobs <json>: [{id, wavs:[abs paths], verses_path, speaker, work}]  (verses_path = json list of Deva verse strings)
"""
import os, json, argparse
import numpy as np, torch, soundfile as sf
import nemo.collections.asr as nemo_asr

ROOT="/home/ece/BigDisk/Prathosh/ASR"
OUT=f"{ROOT}/data/ctcseg"; CONCAT=f"{ROOT}/data/norm16k_concat"
os.makedirs(OUT, exist_ok=True); os.makedirs(CONCAT, exist_ok=True)
MODEL=("/home/ece/.cache/torch/NeMo/NeMo_2.7.3/hf_hub_cache/ai4bharat/"
       "indicconformer_stt_sa_hybrid_ctc_rnnt_large/c82246cc7136e7f1be8df3090e3a07d9/"
       "indicconformer_stt_sa_hybrid_rnnt_large.nemo")
OFF,V,BLANK=4096,256,5632; COLS=[BLANK]+list(range(OFF,OFF+V)); W=28.0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--jobs", required=True)
    ap.add_argument("--model", default=MODEL); a=ap.parse_args()
    jobs=json.load(open(a.jobs))
    m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(a.model, map_location="cuda"); m.eval()
    def logprobs(clip):
        sig=torch.tensor(clip).unsqueeze(0).cuda(); sl=torch.tensor([clip.shape[0]]).cuda()
        with torch.no_grad():
            enc,_=m.forward(input_signal=sig, input_signal_length=sl)
            lp=m.ctc_decoder(encoder_output=enc)[0].cpu().numpy()
        x=lp[:,COLS]; x=x-x.max(-1,keepdims=True); e=np.exp(x)
        return np.log(e/e.sum(-1,keepdims=True)+1e-12).astype(np.float32)
    sr=16000
    for j in jobs:
        # concat wavs in order
        segs=[sf.read(w)[0].astype(np.float32) for w in j["wavs"]]
        audio=np.concatenate(segs) if len(segs)>1 else segs[0]
        dur=len(audio)/sr
        cwav=f"{CONCAT}/{j['id']}.wav"; sf.write(cwav, audio, sr)
        # windowed logprobs (non-overlapping, drift-free)
        parts=[]; s=0.0
        wins=[]
        while s < dur-1e-3: wins.append((s,min(dur,s+W))); s+=W
        if len(wins)>=2 and (wins[-1][1]-wins[-1][0])<4.0:
            wins[-2]=(wins[-2][0],wins[-1][1]); wins.pop()
        for ws,we in wins:
            parts.append(logprobs(audio[int(ws*sr):int(we*sr)]))
        full=np.concatenate(parts,0); idur=dur/full.shape[0]
        verses=json.load(open(j["verses_path"]))
        tokens=[[i+1 for i in m.tokenizer.text_to_ids(t,"sa")] for t in verses]
        np.save(f"{OUT}/{j['id']}.npy", full)
        json.dump(dict(index_duration=float(idur), wav=cwav, dur=dur, speaker=j["speaker"],
                       skandha=None, adhyaya=j.get("chapter"), work=j["work"],
                       utterances=verses, tokens=tokens),
                  open(f"{OUT}/{j['id']}.json","w"), ensure_ascii=False)
        print(f"  {j['id'][:44]:44} T={full.shape[0]:5d} verses={len(verses):3d} {dur/60:5.1f}m", flush=True)
    sv=json.load(open(f"{ROOT}/lm/sa_vocab.json"))
    json.dump(["<blank>"]+sv["tokens"], open(f"{OUT}/char_list.json","w"), ensure_ascii=False)
    print("nonbhp stage1 done", flush=True)

if __name__=="__main__": main()
